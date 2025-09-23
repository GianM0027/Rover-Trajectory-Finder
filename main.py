import numpy as np
from hirise_dtm import HiriseDTM
from custom_environment import GridMarsEnv

# todo: valuta se aggiungere un'ulteriore copia della mappa, che per ogni pixel "visto", tiene in memoria info sulla sua altitudine

SEED = 42
np.random.seed(SEED)

filepath = "DTMs/DTEEC_016460_2230_016170_2230_G01.IMG"
dtm_file = HiriseDTM(filepath)

grid_mars_env = GridMarsEnv(dtm_file,
                            render_mode="human",
                            map_size=50,
                            fov_distance=5,
                            rover_max_step=1,
                            rover_max_drop=1,
                            rover_max_number_of_steps=1000)
obs, info = grid_mars_env.reset(seed=SEED)

terminated = False
while not terminated:
    action = np.random.randint(8)
    observation, reward, terminated, truncated, info = grid_mars_env.step(action, verbose=True)
