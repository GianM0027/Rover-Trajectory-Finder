import torch
from constants import *
from agent import Agent
from impala import ImpalaModel
from hirise_dtm import HiriseDTM

curriculum_learning_config = {
    0: {
        "training_timesteps": 2.5e6,
        "map_size": 10,
        "learning_rate": 5e-5,
        "freeze_cnn": False,
        "c2": 0.01,
        "weights_to_reload": None,
        "training_seed": 42
    },

    1: {
        "training_timesteps": 6e6,
        "map_size": 20,
        "learning_rate": 5e-5,
        "freeze_cnn": True,
        "c2": 0.05,
        "weights_to_reload": os.path.join('weights', f'cnn_weights_10x10_step0.h5'),
        "training_seed": 42*2
    },

    2: {
        "training_timesteps": 10e6,
        "map_size": 20,
        "learning_rate": 7e-6,
        "freeze_cnn": False,
        "c2": 0.01,
        "weights_to_reload": os.path.join('weights', f'full_weights_20x20_step1.h5'),
        "training_seed": 42*3
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

    filepath = os.path.join("DTMs", "DTEED_055378_2190_055444_2190_A01.IMG")
    dtm_file = HiriseDTM(filepath)

    max_step_height = 1
    max_drop_height = 1
    n_environments = 16

    policy_network = ImpalaModel(input_channels=4)
    agent = Agent(n_environments=n_environments,
                  policy_network=policy_network,
                  dtm_file=dtm_file,
                  max_step_height=max_step_height,
                  max_drop_height=max_drop_height)

    agent.curriculum_learning_train(config=curriculum_learning_config, device=device)
