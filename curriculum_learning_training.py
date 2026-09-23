import torch
from constants import *
from agent import Agent
from impala import ImpalaModel

curriculum_learning_config = {
    # Step 0 impara la navigazione su mappe piccole. 20x20 satura verso i 200k passi e da li
    # peggiora leggermente (91.2% -> 83.4% di successo fra i 200k e i 2M), quindi non serve
    # allungarlo: i pesi salvati alla fine sarebbero peggiori di quelli a meta' corsa.
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

    # Step 1 porta la stessa policy su mappe 40x40. Si ricaricano solo i pesi convoluzionali:
    # la dense layer ha ingresso fisso (3x3x32 = 288 valori a 20x20, 5x5x32 = 800 a 40x40) e
    # quindi non e' trasferibile, mentre il tronco che legge il terreno lo e'.
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
