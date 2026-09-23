"""
Build a tile pool from a directory of HiRISE DTMs.

The DTMs are never loaded into memory: every candidate tile is read straight off disk with a
rasterio windowed read, so this script costs a few tens of KB of RAM regardless of how many
DTMs, or how large, are scanned. The result is a single .npy array of shape
(n_tiles, tile_size, tile_size) plus a JSON file recording where each tile came from.

This makes the .IMG files a build-time input rather than a runtime dependency: download a batch
of DTMs, fold their tiles into a pool, delete the .IMG files, repeat. The pool size is what you
choose here, not a function of how much terrain went into it.

Usage:
    python build_tile_pool.py --dtm-dir DTMs/training --out tile_pools/tiles_training.npy
    python build_tile_pool.py --dtm-dir DTMs/testing  --out tile_pools/tiles_testing.npy --n-tiles 4000
"""

import argparse
import json
import os
import time

import numpy as np
import rasterio
from rasterio.windows import Window


def grid_origins(height, width, tile_size, rng):
    """
    Every non-overlapping tile position in a raster, in random order.

    Tiles are taken from a regular lattice so that no two tiles of the same DTM overlap (which
    would put near-duplicate terrain in the pool), with a random offset per DTM so the lattice
    is not always anchored at the same pixel.
    """
    offset_y = int(rng.integers(max(1, (height % tile_size) + 1)))
    offset_x = int(rng.integers(max(1, (width % tile_size) + 1)))

    rows = np.arange(offset_y, height - tile_size + 1, tile_size)
    cols = np.arange(offset_x, width - tile_size + 1, tile_size)
    origins = np.array([(r, c) for r in rows for c in cols], dtype=np.int64)
    rng.shuffle(origins)
    return origins


def read_tile(src, row, col, tile_size, nodata):
    """Read one tile and mark every invalid pixel as +inf, the way HiriseDTM does."""
    tile = src.read(1, window=Window(col, row, tile_size, tile_size)).astype(np.float32)

    invalid = ~np.isfinite(tile)
    if nodata is not None:
        invalid |= (tile == nodata)

    if invalid.any():
        tile[invalid] = np.inf

    return tile, invalid.mean()


def collect_from_dtm(src, origins, start, quota, tile_size, max_nodata, min_relief, pool, filled):
    """
    Walk this DTM's shuffled tile positions from `start` and write accepted tiles into the pool.

    :return: (n_accepted, next_start, stats)
    """
    stats = {"tried": 0, "rejected_nodata": 0, "rejected_flat": 0}
    accepted = 0
    index = start
    nodata = src.nodata

    while accepted < quota and index < len(origins):
        row, col = origins[index]
        index += 1
        stats["tried"] += 1

        tile, nodata_fraction = read_tile(src, int(row), int(col), tile_size, nodata)

        if nodata_fraction > max_nodata:
            stats["rejected_nodata"] += 1
            continue

        finite = tile[np.isfinite(tile)]
        if finite.size == 0 or finite.std() < min_relief:
            # a perfectly flat tile makes the task trivial and contributes no gradient signal
            stats["rejected_flat"] += 1
            continue

        pool[filled + accepted] = tile
        accepted += 1

    return accepted, index, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtm-dir", required=True, help="directory holding the .IMG DTMs to scan")
    parser.add_argument("--out", required=True, help="destination .npy file for the pool")
    parser.add_argument("--n-tiles", type=int, default=30000,
                        help="how many tiles to collect (default: 30000)")
    parser.add_argument("--tile-size", type=int, default=64,
                        help="tile side in pixels; must cover map_size + 2*fov_distance (default: 64)")
    parser.add_argument("--max-nodata", type=float, default=0.0,
                        help="largest fraction of nodata pixels a tile may contain (default: 0.0, none)")
    parser.add_argument("--min-relief", type=float, default=0.05,
                        help="reject tiles whose altitude standard deviation is below this, in metres")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for tile position sampling (default: 42)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    dtm_files = sorted(f for f in os.listdir(args.dtm_dir) if f.upper().endswith(".IMG"))
    if not dtm_files:
        raise SystemExit(f"No .IMG file found in {args.dtm_dir}")

    print(f"Scanning {len(dtm_files)} DTMs in {args.dtm_dir} for {args.n_tiles} tiles "
          f"of {args.tile_size}x{args.tile_size}\n")

    parent_dir = os.path.dirname(args.out)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # written straight to disk so peak memory stays at one tile, whatever the pool size
    pool = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float32,
                                     shape=(args.n_tiles, args.tile_size, args.tile_size))

    sources = []
    provenance = []
    handles, origins, cursors = [], [], []

    started = time.time()
    try:
        for name in dtm_files:
            src = rasterio.open(os.path.join(args.dtm_dir, name))
            handles.append(src)
            origins.append(grid_origins(src.shape[0], src.shape[1], args.tile_size, rng))
            cursors.append(0)
            sources.append({"file": name, "shape": list(src.shape),
                            "tile_positions": len(origins[-1])})

        filled = 0
        # the first pass gives every DTM the same quota, so a large raster cannot dominate the
        # pool; a second pass hands the leftover budget to the DTMs that still have positions
        for pass_number in (1, 2):
            if filled >= args.n_tiles:
                break

            available = [i for i in range(len(handles)) if cursors[i] < len(origins[i])]
            if not available:
                break

            quota = max(1, (args.n_tiles - filled) // len(available))

            for i in available:
                if filled >= args.n_tiles:
                    break

                take = min(quota, args.n_tiles - filled)
                accepted, cursors[i], stats = collect_from_dtm(
                    handles[i], origins[i], cursors[i], take, args.tile_size,
                    args.max_nodata, args.min_relief, pool, filled)

                provenance.extend([sources[i]["file"]] * accepted)
                filled += accepted

                if pass_number == 1:
                    print(f"  {sources[i]['file'][:44]:46s} {accepted:6d} tiles  "
                          f"(tried {stats['tried']}, rejected {stats['rejected_nodata']} nodata / "
                          f"{stats['rejected_flat']} flat)")
    finally:
        for src in handles:
            src.close()

    del pool

    if filled < args.n_tiles:
        # the requested pool could not be filled: rewrite it at its real size
        print(f"\nOnly {filled} tiles passed the filters, trimming the pool to that size.")
        trimmed = np.array(np.load(args.out, mmap_mode="r")[:filled])
        np.save(args.out, trimmed)
        del trimmed

    pool = np.load(args.out, mmap_mode="r")
    step = max(1, len(pool) // 2000)
    relief = np.array([pool[i][np.isfinite(pool[i])].std() for i in range(0, len(pool), step)])

    meta = {
        "n_tiles": int(len(pool)),
        "tile_size": args.tile_size,
        "dtm_dir": args.dtm_dir,
        "max_nodata": args.max_nodata,
        "min_relief": args.min_relief,
        "seed": args.seed,
        "sources": sources,
        "tiles_per_source": {name: provenance.count(name) for name in dtm_files},
        "provenance": provenance,
    }
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{len(pool)} tiles -> {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)")
    print(f"provenance -> {meta_path}")
    print(f"relief (altitude std): median {np.median(relief):.2f} m, "
          f"5th pct {np.percentile(relief, 5):.2f} m, 95th pct {np.percentile(relief, 95):.2f} m")
    print(f"build time: {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
