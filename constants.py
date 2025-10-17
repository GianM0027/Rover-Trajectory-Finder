import os
from math import sqrt

WEIGHTS_DIR = 'weights'
TRAINING_INFO_DIR = 'training_info'


def get_weights_path(map_size, step=""):
    if step != "":
        step = f"_step{step}"
    cnn_weights_path = os.path.join(WEIGHTS_DIR, f'cnn_weights_{map_size}x{map_size}{step}.h5')
    full_weights_path = os.path.join(WEIGHTS_DIR, f'full_weights_{map_size}x{map_size}{step}.h5')
    return cnn_weights_path, full_weights_path


def get_training_info_path(map_size, step=""):
    if step != "":
        step = f"_step{step}"
    training_info_path = os.path.join(TRAINING_INFO_DIR, f"episodes_summary_{map_size}x{map_size}{step}.json")
    training_losses_path = os.path.join(TRAINING_INFO_DIR, f"losses_{map_size}x{map_size}{step}.json")
    training_parameters_path = os.path.join(TRAINING_INFO_DIR, f"training_parameters_{map_size}x{map_size}{step}.json")
    return training_info_path, training_losses_path, training_parameters_path


def get_fovDistance_maxSteps(map_size):
    fov_distance = map_size // 5
    max_number_of_steps = int(15 * (map_size * sqrt(map_size)))
    return fov_distance, max_number_of_steps
