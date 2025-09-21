import time

import numpy as np

from custom_environment import GridMarsEnv
from hirise_dtm import HiriseDTM
from matplotlib import pyplot as plt

filepath = "DTMs/DTEED_082989_1630_083055_1630_A01.IMG"
dtm_file = HiriseDTM(filepath)

grid_mars_env = GridMarsEnv(dtm_file, render_mode="human", map_size=10, fov_distance=2)
obs, info = grid_mars_env.reset()

terminated = False
while not terminated:
    action = np.random.randint(8)
    observation, reward, terminated, truncated, info = grid_mars_env.step(action, verbose=True)
    time.sleep(1)