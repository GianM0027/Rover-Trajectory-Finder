import time
import numpy as np
from hirise_dtm import HiriseDTM
from custom_environment import GridMarsEnv

#todo: add seed for environment generation and initialization

filepath = "DTMs/DTEED_082989_1630_083055_1630_A01.IMG"
dtm_file = HiriseDTM(filepath)

grid_mars_env = GridMarsEnv(dtm_file,
                            render_mode="human",
                            map_size=15,
                            fov_distance=2,
                            rover_max_step=0.3,
                            rover_max_drop=0.5)
obs, info = grid_mars_env.reset()

terminated = False
while not terminated:
    action = np.random.randint(8)
    observation, reward, terminated, truncated, info = grid_mars_env.step(action, verbose=True)
    time.sleep(1)