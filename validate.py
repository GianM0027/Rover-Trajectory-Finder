import torch
from agent import Agent
from impala import ImpalaModel
from constants import *

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    SAVE_RESULTS = True
    RANDOM_POLICY = False
    SAMPLE_ACTION = True   # argmax leaves the rover looping between cells: 60.5% vs 92.5%
    map_size = 20
    step = 0

    n_environments = 16
    max_step_height = 0.3
    max_drop_height = 0.3
    fov_distance, max_number_of_steps = get_fovDistance_maxSteps(map_size)

    _, full_weights_path = get_weights_path(map_size, step)

    policy_network = None
    if not RANDOM_POLICY:
        policy_network = ImpalaModel(input_channels=OBSERVATION_CHANNELS)
        policy_network(torch.randn(1, OBSERVATION_CHANNELS, map_size, map_size))
        policy_network.load_full_weights(full_weights_path, device=device)

    # Validation runs on the held-out tile pool, built from the DTMs in DTMs/testing
    tile_pool_path = get_tile_pool_path("testing")

    agent = Agent(n_environments=n_environments,
                  policy_network=policy_network,
                  fov_distance=fov_distance,
                  map_size=map_size,
                  max_number_of_steps=max_number_of_steps,
                  tile_pool_path=tile_pool_path,
                  max_step_height=max_step_height,
                  max_drop_height=max_drop_height,
                  seed=1)

    if SAVE_RESULTS:
        validation_info_path = get_validation_info_path(map_size=map_size, random_policy=RANDOM_POLICY)
        agent.validate(num_episodes=1000,
                       device=device,
                       policy_network=policy_network,
                       validation_info_path=validation_info_path,
                       sample_action=SAMPLE_ACTION)
    else:
        agent.run_simulation(use_policy_network=not RANDOM_POLICY,
                             device=device,
                             sample_action=SAMPLE_ACTION)
