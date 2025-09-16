import os
import rasterio
from matplotlib import pyplot as plt


class HiriseDTM:
    """
    This class takes as input the path to a HiRISE .IMG file, converts it into a NumPy array,
    and provides a set of utility functions for working with it.
    """

    def __init__(self, img_path):
        with rasterio.open(img_path) as src:
            data = src.read(1)  # first band
            nodata = src.nodata  # check nodata value

        if nodata is not None:
            data = data.astype(float)
            data[data == nodata] = float("nan")

        self.img_path = img_path
        self.file_name = os.path.split(img_path)[-1].replace(".IMG", "")
        self.numpy_data = data
        self.metadata = self._get_metadata()

    def plot_dtm(self, figsize=(12, 12)):
        """
        :param figsize: plot figure size.
        Shows the DTM image in a matplotlib figure.
        """
        plt.figure(figsize=figsize)
        plt.imshow(self.numpy_data, cmap="terrain")
        plt.colorbar(label="Elevation (m)")
        plt.title("HiRISE DTM")
        plt.show()

    def _get_metadata(self):
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