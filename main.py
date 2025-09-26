import numpy as np
from hirise_dtm import HiriseDTM
from custom_environment import GridMarsEnv
from agent import Agent

SEED = 42
np.random.seed(SEED)

filepath = "DTMs/DTEEC_016460_2230_016170_2230_G01.IMG"
dtm_file = HiriseDTM(filepath)

grid_mars_env = GridMarsEnv(dtm_file,
                            render_mode="human",
                            map_size=10,
                            fov_distance=2,
                            rover_max_step=1,
                            rover_max_drop=1,
                            rover_max_number_of_steps=15)

agent = Agent(mars_environment=grid_mars_env, seed=SEED)
agent.run_simulation()



print('test')