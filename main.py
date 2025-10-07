import numpy as np
import os
import torch
from hirise_dtm import HiriseDTM
from agent import Agent
from policy_network import PolicyNetwork
from impala import ImpalaModel

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")


TRAINING_SEED = 42
np.random.seed(TRAINING_SEED)

weights_path = os.path.join('weights', 'weights.h5')
filepath = os.path.join("DTMs", "DTEED_082989_1630_083055_1630_A01.IMG")
training_info_path = os.path.join("training_info", "episodes_summary.json")
training_losses_path = os.path.join("training_info", "losses.json")
training_parameters_path = os.path.join("training_info", "training_parameters.json")

config = {
   "input_channels": 8,
   "backbone": [
       {"type": "conv", "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},
       {"type": "pool", "mode": "max", "kernel_size": 2},
       {"type": "conv", "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},
       {"type": "pool", "mode": "max", "kernel_size": 2},
       {"type": "conv", "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},
       #{"type": "pool", "mode": "max", "kernel_size": 2}
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


dtm_file = HiriseDTM(filepath)

TRAIN = True
map_size = 20
fov_distance = map_size // 5
max_number_of_steps = map_size*10
max_step_height = 10
max_drop_height = 10
frame_skip_len = 2

# policy_network = PolicyNetwork(config)
policy_network = ImpalaModel()

agent = Agent(policy_network=policy_network,
              fov_distance=fov_distance,
              map_size=map_size,
              max_number_of_steps=max_number_of_steps,
              dtm_file=dtm_file,
              max_step_height=max_step_height,
              max_drop_height=max_drop_height)

if TRAIN:
    agent.seed = TRAINING_SEED
    agent.train(training_episodes=1000,
                batch_size=256,
                minibatch_size=128,
                epochs=2,
                frame_skip_len=frame_skip_len,
                weights_path=weights_path,
                training_info_path=training_info_path,
                training_losses_path=training_losses_path,
                training_parameters_path=training_parameters_path,
                device=device,
                step_verbose=False,
                c2=0.01,
                learning_rate=1e-6)
else:
    policy_network(torch.randn(1, 4*frame_skip_len, map_size, map_size))
    agent.policy_network.load_weights(weights_path)
    agent.run_simulation(use_policy_network=True, frame_skip_len=frame_skip_len, device=device, verbose=True, sample_action=True)
