import torch
from tile_pool import TilePool
from agent import Agent
from impala import ImpalaModel
import gymnasium as gym
from custom_environment import GridMarsEnv
from constants import *

FREEZE_CNN = True
RELOAD_WEIGHTS = False
RELOAD_FULL_WEIGHTS = False
MAP_SIZE_TO_RELOAD = 20

map_size = 20
TRAINING_TIMESTEPS = 6000000
LEARNING_RATE = 5e-5

max_step_height = 0.3
max_drop_height = 0.3
fov_distance, max_number_of_steps = get_fovDistance_maxSteps(map_size)

n_environments = 16
batch_size = n_environments * 128
minibatch_size = batch_size // 8

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

    tile_pool_path = get_tile_pool_path("training")

    policy_network = ImpalaModel(input_channels=OBSERVATION_CHANNELS)
    if RELOAD_WEIGHTS:
        policy_network(torch.randn(1, OBSERVATION_CHANNELS, map_size, map_size))
        if RELOAD_FULL_WEIGHTS:
            weights_signature = "full"
        else:
            weights_signature = "cnn"
        policy_network.load_full_weights(os.path.join(WEIGHTS_DIR,
                                                      f'{weights_signature}_weights_{MAP_SIZE_TO_RELOAD}x{MAP_SIZE_TO_RELOAD}.h5'),
                                         device=device)

    if FREEZE_CNN:
        for name, param in policy_network.named_parameters():
            if name.startswith(('conv_block1', 'conv_block2', 'conv_block3')):
                param.requires_grad = False

    mars_environments = gym.vector.AsyncVectorEnv([
        lambda: GridMarsEnv(dtm=TilePool(tile_pool_path),
                            map_size=map_size,
                            fov_distance=fov_distance,
                            rover_max_step=max_step_height,
                            rover_max_drop=max_drop_height,
                            rover_max_number_of_steps=max_number_of_steps)
        for _ in range(n_environments)
    ],
        shared_memory=False
    )

    agent = Agent(n_environments=n_environments,
                  policy_network=policy_network,
                  fov_distance=fov_distance,
                  map_size=map_size,
                  max_number_of_steps=max_number_of_steps,
                  tile_pool_path=tile_pool_path,
                  max_step_height=max_step_height,
                  max_drop_height=max_drop_height)

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
