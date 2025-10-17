import torch
from hirise_dtm import HiriseDTM
from agent import Agent
from impala import ImpalaModel
import gymnasium as gym
from custom_environment import GridMarsEnv
from constants import *

TRAIN = False
FREEZE_CNN = True
TRAINING_TIMESTEPS = 6000000
LEARNING_RATE = 5e-5

map_size = 20
max_step_height = 1
max_drop_height = 1
fov_distance, max_number_of_steps = get_fovDistance_maxSteps(map_size)

n_environments = 16
batch_size = n_environments * 128
minibatch_size = batch_size//8

cnn_weights_path, full_weights_path = get_weights_path(map_size)
training_info_path, training_losses_path, training_parameters_path = get_training_info_path(map_size)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    filepath = os.path.join("DTMs", "DTEED_055378_2190_055444_2190_A01.IMG")
    dtm_file = HiriseDTM(filepath)

    policy_network = ImpalaModel(input_channels=4)
    policy_network(torch.randn(1, 4, map_size, map_size))
    policy_network.load_feature_extractor_weights(os.path.join('weights', f'cnn_weights_30x30.h5'))

    if FREEZE_CNN:
        for name, param in policy_network.named_parameters():
            if name.startswith(('conv_block1', 'conv_block2', 'conv_block3')):
                param.requires_grad = False

    if TRAIN:
        mars_environments = gym.vector.AsyncVectorEnv([
            lambda: GridMarsEnv(dtm=dtm_file,
                                map_size=map_size,
                                fov_distance=fov_distance,
                                rover_max_step=max_step_height,
                                rover_max_drop=max_drop_height,
                                rover_max_number_of_steps=max_number_of_steps)
            for _ in range(n_environments)
            ],
            shared_memory=False
        )
    else:
        mars_environments = None

    agent = Agent(n_environments=n_environments,
                  policy_network=policy_network,
                  fov_distance=fov_distance,
                  map_size=map_size,
                  max_number_of_steps=max_number_of_steps,
                  dtm_file=dtm_file,
                  max_step_height=max_step_height,
                  max_drop_height=max_drop_height)

    if TRAIN:
        agent.train(environments=mars_environments,
                    training_steps=TRAINING_TIMESTEPS,
                    batch_size=batch_size,
                    minibatch_size=minibatch_size,
                    epochs=3,
                    cnn_weights_path=cnn_weights_path,
                    full_weights_path=full_weights_path,
                    training_info_path=training_info_path,
                    training_losses_path=training_losses_path,
                    training_parameters_path=training_parameters_path,
                    device=device,
                    learning_rate=LEARNING_RATE,
                    save_interval=100000,
                    c2=0.01)
    else:
        policy_network(torch.randn(1, 4, map_size, map_size))
        agent.policy_network.load_full_weights(full_weights_path)
        agent.run_simulation(use_policy_network=True, device=device, verbose=True, sample_action=True)
