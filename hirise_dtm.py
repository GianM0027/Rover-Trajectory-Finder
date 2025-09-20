import os

import numpy as np
import rasterio
from typing import Tuple, Dict
from matplotlib import pyplot as plt


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
            data[data == nodata] = float("nan")

        self.img_path = img_path
        self.numpy_image = data
        self.file_name = os.path.split(img_path)[-1].replace(".IMG", "")
        self.metadata = self._get_metadata()

    def get_portion_of_map(self, size):
        # todo: metodo che estrae una sezione dell'immagine di grandezza (size x size), evitando i NaN
        # todo: il metodo restituisce anche le coordinate top-left dalle quali è stata ricavata la porzione di mappa
        #       IMPORTANTE: la top left position è una tupla (x, y) dove x è la colonna e y è la riga

        return np.zeros((size, size)), (0, 0)

    def get_field_of_view(self, position, fov_distance):
        # todo: metodo che dato un punto di coordinate (x,y) restituisce la matrice booleana che evidenzia i pixel che
        #       il rover vede da quella posizione

        return np.ones((fov_distance, fov_distance))

    def get_possible_moves(self, position):
        # todo: metodo che dato un punto di coordinate (x,y) restituisce la matrice booleana che evidenzia i pixel su cui
        #       il rover può spostarsi da quella posizione.
        #       Per esempio:
        #       - Se a destra l'elevazione è di poco minore di quella su cui si trova il rover, allora ci si può spostare
        #       - Se a sinistra c'è una forte elevazione rispetto a dove sta il rover (un muro) allora non ci si può spostare
        #       - E così via... Alla fine avremo una matrice 3x3 booleana che indica se il rover può spostarsi di una posizione
        #         su quel pixel (1) oppure no (0)

        return np.ones((3, 3))

    def get_lowest_highest_altitude(self):
        return np.nanmin(self.numpy_image), np.nanmax(self.numpy_image)

    def plot_dtm(self, figsize: Tuple = (12, 12)) -> None:
        """
        :param figsize: plot figure map_size.
        Shows the DTM numpy_image in a matplotlib figure.
        """
        # todo: migliorare la funzione di plot per essere più nitida, ma anche per plottare una sezione della mappa invece
        #       della mappa intera
        plt.figure(figsize=figsize)
        plt.imshow(self.numpy_image, cmap="terrain")
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
