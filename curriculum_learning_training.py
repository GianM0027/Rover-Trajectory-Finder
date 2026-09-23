import torch
from constants import *
from agent import Agent
from impala import ImpalaModel

curriculum_learning_config = {
    # Step 0 learns to navigate on small maps. 20x20 saturates around 200k steps and drifts
    # slightly worse afterwards (91.2% -> 83.4% success between 200k and 2M), so there is no
    # point extending it: the weights saved at the end would be worse than the ones already
    # overwritten halfway through.
    0: {
        "training_timesteps": 3e5,
        "map_size": 20,
        "learning_rate": 5e-5,
        "freeze_cnn": False,
        "c1": 0.05,
        "c2": 0.01,
        "weights_to_reload": None,
        "training_seed": 42
    },

    # Step 1 moves the same policy onto 40x40 maps. Only the convolutional weights are
    # reloaded: the dense layer has a fixed input width (3x3x32 = 288 values at 20x20,
    # 5x5x32 = 800 at 40x40) so it cannot transfer, while the trunk reading the terrain can.
    1: {
        "training_timesteps": 6e5,
        "map_size": 40,
        "learning_rate": 5e-5,
        "freeze_cnn": False,
        "c1": 0.05,
        "c2": 0.01,
        "weights_to_reload": os.path.join(WEIGHTS_DIR, 'cnn_weights_20x20_step0.h5'),
        "training_seed": 84
    }
}

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    tile_pool_path = get_tile_pool_path("training")

    max_step_height = 0.3
    max_drop_height = 0.3
    n_environments = 32

    policy_network = ImpalaModel(input_channels=OBSERVATION_CHANNELS)
    agent = Agent(n_environments=n_environments,
                  policy_network=policy_network,
                  tile_pool_path=tile_pool_path,
                  max_step_height=max_step_height,
                  max_drop_height=max_drop_height)

    agent.curriculum_learning_train(config=curriculum_learning_config, device=device)
