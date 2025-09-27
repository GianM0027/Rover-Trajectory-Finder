import numpy as np
import os
import torch
from hirise_dtm import HiriseDTM
from custom_environment import GridMarsEnv
from agent import Agent
from policy_network import PolicyNetwork

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

SEED = 42
np.random.seed(SEED)
weights_path = os.path.join('weights', 'weights.h5')
config = {
           "input_channels": 3,
           "backbone": [
               {"type": "conv", "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},
               {"type": "pool", "mode": "max", "kernel_size": 2},
               {"type": "conv", "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},
               {"type": "pool", "mode": "max", "kernel_size": 2}
           ],
           "head_action": [
               {"type": "fc", "out_features": 128, "activation": "relu"},
               {"type": "fc", "out_features": 8}
           ],
           "head_value": [
               {"type": "fc", "out_features": 64, "activation": "relu"},
               {"type": "fc", "out_features": 1}
           ]
       }

filepath = "DTMs/DTEEC_016460_2230_016170_2230_G01.IMG"
dtm_file = HiriseDTM(filepath)
TRAIN = True
map_size = 10
fov_distance = 3

grid_mars_env = GridMarsEnv(dtm_file,
                            render_mode="no", # human
                            map_size=map_size,
                            fov_distance=fov_distance,
                            rover_max_step=1,
                            rover_max_drop=1,
                            rover_max_number_of_steps=100)

policy_network = PolicyNetwork(config)
if not TRAIN:
    policy_network(torch.randn(1, 3, map_size, map_size))
    policy_network.load_weights(weights_path)

agent = Agent(mars_environment=grid_mars_env, policy_network=policy_network, seed=SEED)

if TRAIN:
    agent.train(training_episodes=3000,
                batch_size=512,
                minibatch_size=16,
                epochs=1,
                weights_path=weights_path,
                device=device)
else:
    agent.run_simulation(use_policy_network=True, device=device)
