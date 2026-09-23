import json
import os

import numpy as np

from hirise_dtm import HiriseDTM


class TilePool:
    """
    A pool of terrain tiles extracted offline from a set of HiRISE DTMs by build_tile_pool.py.

    The pool is a single .npy array of shape (n_tiles, tile_size, tile_size) opened in
    memory-mapped mode, so every worker process maps the same file and the operating system
    keeps one shared copy in its page cache. Adding more DTMs to the pool therefore costs
    nothing at training time: the pool size is fixed when it is built, not by how many DTMs
    went into it.

    Each sample() returns one tile wrapped in a HiriseDTM, so the environment keeps using the
    exact same geometry helpers (get_portion_of_map, get_fov_mask, get_possible_moves,
    get_adjacency_list) it uses for a full DTM.

    :param path: path to the .npy file produced by build_tile_pool.py.
    :param augment: whether to apply a random rotation/flip to each sampled tile.
    """

    def __init__(self, path: str | os.PathLike, augment: bool = True):
        self.path = str(path)
        self.augment = augment
        self.tiles = np.load(self.path, mmap_mode='r')

        if self.tiles.ndim != 3 or self.tiles.shape[1] != self.tiles.shape[2]:
            raise ValueError(f"Expected a (n_tiles, tile_size, tile_size) array, got {self.tiles.shape}")

        self.n_tiles = self.tiles.shape[0]
        self.tile_size = self.tiles.shape[1]

        meta_path = os.path.splitext(self.path)[0] + "_meta.json"
        self.metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

    def __len__(self):
        return self.n_tiles

    def __repr__(self):
        sources = len(self.metadata.get("sources", []))
        return (f"TilePool({self.n_tiles} tiles of {self.tile_size}x{self.tile_size}"
                f"{f' from {sources} DTMs' if sources else ''}, "
                f"{self.tiles.nbytes / 1e6:.0f} MB memory-mapped)")

    def sample(self, rng=None):
        """
        Draw one random tile and return it as a HiriseDTM.

        With augment enabled, one of the 8 symmetries of the square (4 rotations, optionally
        mirrored) is applied, which multiplies the effective number of distinct maps by 8.

        :param rng: a numpy Generator; falls back to the global numpy random state.
        :return: a HiriseDTM wrapping a copy of the tile.
        """
        if rng is None:
            rng = np.random

        tile = self.tiles[int(rng.integers(self.n_tiles))]

        if self.augment:
            k = int(rng.integers(4))
            if k:
                tile = np.rot90(tile, k)
            if rng.integers(2):
                tile = np.fliplr(tile)

        # HiriseDTM copies the array, which also makes the rotated view contiguous again
        return HiriseDTM(img=tile)
