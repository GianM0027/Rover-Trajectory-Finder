import os
from collections import deque

import rasterio
import numpy as np
from typing import Tuple, Dict
from matplotlib import pyplot as plt

plt.style.use('default')  # Stile classico con sfondo bianco


class HiriseDTM:
    """
    This class takes as input the path to a local HiRISE .IMG file, converts it into a NumPy array,
    and provides a set of utility functions for working with it.

    :param img_path: Path to a local HiRISE .IMG file.
    """

    def __init__(self, img_path: str | os.PathLike):
        with rasterio.open(img_path) as src:
            data = src.read(1)      # first band
            nodata = src.nodata     # check nodata value

        if nodata is not None:
            data = data.astype(float)
            data[data == nodata] = np.inf  # infinitely tall wall at map borders

        self.numpy_image = data
        self.img_path = img_path
        self.file_name = os.path.split(img_path)[-1].replace(".IMG", "")
        self.metadata = self._get_metadata()

    def get_portion_of_map(self, size, max_percentage_inf=0, random_rotation=False):
        # Extracts a size x size portion of the image, avoiding too many np.inf
        img_height, img_width = self.numpy_image.shape[:2]

        while True:
            # pick random top-left corner
            x = np.random.randint(0, img_width - size + 1)
            y = np.random.randint(0, img_height - size + 1)

            # extract portion
            image_subset = self.numpy_image[y:y + size, x:x + size]

            # count infinities
            num_inf = np.sum(np.isinf(image_subset))
            if num_inf <= max_percentage_inf * (size * size):
                break

        if random_rotation:
            # todo: data augmentation sulle porzioni della mappa, effettua una rotazione random dell'immagine
            pass

        # return image portion and its coordinates as (row,column)=(y,x)
        return image_subset, (y, x)

    @classmethod
    def _which_pixels_are_visible(cls, altitudes):
        """
        Given a line of pixels of length "fov_distance", where the rover stands in the first one,
        compute which ones are visible from there.
        """
        visibles = [False] * len(altitudes)
        visibles[0] = True  # rover sees the pixel it's in

        rover_altitude = altitudes[0]
        max_slope = float("-inf")

        for distance in range(1, len(altitudes)):
            if altitudes[distance] == np.inf:
                visibles[distance] = True
                continue

            slope = (altitudes[distance] - rover_altitude) / distance
            if slope >= max_slope:
                visibles[distance] = True
                max_slope = slope

        return visibles

    def get_fov_mask(self, position, fov_distance, action_to_direction):
        mask_size = (fov_distance * 2) + 1
        fov_mask = np.zeros((mask_size, mask_size))

        center = np.array([fov_distance, fov_distance])

        for _, action_direction in action_to_direction.items():
            idx_list = []
            for distance in range(fov_distance + 1):
                idx = np.array(position) + action_direction * distance

                # map borders control
                if not (0 <= idx[0] < self.numpy_image.shape[0] and 0 <= idx[1] < self.numpy_image.shape[1]):
                    break
                idx_list.append(idx)

            if not idx_list:
                continue

            altitudes = [self.numpy_image[tuple(idx)] for idx in idx_list]
            visible_pixels = self._which_pixels_are_visible(altitudes)

            for idx, visible_pixel in zip(idx_list, visible_pixels):
                idx_to_update = center + (idx - position)
                if 0 <= idx_to_update[0] < mask_size and 0 <= idx_to_update[1] < mask_size:
                    fov_mask[tuple(idx_to_update)] = visible_pixel

        return fov_mask

    def get_possible_moves(self, position, moves, max_step, max_drop, local_map_size, local_map_position):
        """
        Given a point (y, x), returns a 3x3 boolean matrix of possible moves.
        - 1 = rover can move there
        - 0 = rover cannot move there
        """
        possible_moves = np.ones((3, 3), dtype=bool)
        y, x = position
        current_altitude = self.numpy_image[y, x]

        for _, move in moves.items():
            moves_idx = np.array((1, 1)) + move         # map move to 3x3 possible_moves matrix index
            new_y, new_x = np.array(position) + move

            # Out of bounds check for y
            if (new_y < local_map_position[0] or new_x < local_map_position[1] or
                    new_y >= local_map_position[0]+local_map_size or new_x >= local_map_position[1]+local_map_size):
                possible_moves[moves_idx[0], moves_idx[1]] = 0
                continue

            new_altitude = self.numpy_image[new_y, new_x]

            # Invalid terrain
            if new_altitude == np.inf:
                possible_moves[moves_idx[0], moves_idx[1]] = 0
                continue

            # Too steep to climb
            if new_altitude - current_altitude > max_step:
                possible_moves[moves_idx[0], moves_idx[1]] = 0

            # Too steep downward drop
            if current_altitude - new_altitude > max_drop:
                possible_moves[moves_idx[0], moves_idx[1]] = 0

        return possible_moves

    def get_adjacency_list(self, moves, max_step, max_drop, local_map_size, local_map_position):
        adjacency_list = {}

        for global_y in range(local_map_position[0], local_map_position[0] + local_map_size):
            for global_x in range(local_map_position[1], local_map_position[1] + local_map_size):
                local_y = global_y - local_map_position[0]
                local_x = global_x - local_map_position[1]

                possible_moves = self.get_possible_moves(
                    (global_y, global_x),
                    moves,
                    max_step,
                    max_drop,
                    local_map_size,
                    local_map_position
                )

                neighbors = []
                center = np.array((1, 1))
                for move in moves.values():
                    idx = tuple(center + move)
                    if possible_moves[idx]:
                        neighbors.append((local_y + move[0], local_x + move[1]))

                adjacency_list[(local_y, local_x)] = neighbors

        return adjacency_list



    def get_lowest_highest_altitude(self):
        return np.nanmin(self.numpy_image), np.nanmax(self.numpy_image)

    def plot_dtm(self, dtm = None, figsize: Tuple = (12, 12)) -> None:
        """
        :param dtm: dtm to plot (optional), if set to None, the whole map will be plotted.
        :param figsize: plot figure map_size.

        Shows the DTM numpy_image in a matplotlib figure.
        """
        img_to_plot = dtm if dtm else self.numpy_image
        plt.figure(figsize=figsize)
        plt.imshow(img_to_plot, cmap="terrain")
        plt.colorbar(label="Elevation (m)")
        plt.title("HiRISE DTM")
        plt.show()

    def _get_metadata(self) -> Dict:
        """
        Returns the metadata of a HiRISE .IMG file, given its tile name in the format
        'aabcd_xxxxxx_xxxx_yyyyyy_yyyy_Vnn'.

        For details on the naming convention, see: https://www.uahirise.org/dtm/about.php.

        :return: A dictionary containing the metadata of the HiRISE .IMG file.
        """
        unk = "unknown"
        aabcd, xxxxxx, xxxx, yyyyyy, yyyy, Vnn = self.file_name.split("_")

        product_type = "DTM" if aabcd[:2] == "DT" else unk

        file_type = "Areoid Elevations" if aabcd[2] == "E" else unk

        projection = "Equirectangular" if aabcd[3] == "E" else "Polar Stereographic" if aabcd[3] == "P" else unk

        grid_spacing = 0.25 if aabcd[4] == "A" else \
                       0.5 if aabcd[4] == "B" else \
                       1.0 if aabcd[4] == "C" else \
                       2 if aabcd[4] == "D" else unk

        producing_institution = "USGS" if Vnn[0] == "U" else \
                                "University of Arizona" if Vnn[0] == "A" else \
                                "CalTech" if Vnn[0] == "C" else \
                                "NASA Ames" if Vnn[0] == "N" else \
                                "JPL" if Vnn[0] == "J" else \
                                "Ohio State" if Vnn[0] == "O" else \
                                "Planetary Science Institute" if Vnn[0] == "P" else unk

        metadata = {"product_type": product_type,
                    "file_type": file_type,
                    "projection": projection,
                    "grid_spacing": grid_spacing,
                    "orbit_and_latitude_1": (xxxxxx, xxxx),
                    "orbit_and_latitude_2": (yyyyyy, yyyy),
                    "producing_institution": producing_institution,
                    "version_number": Vnn[1:]}

        return metadata
