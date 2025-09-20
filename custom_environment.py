from typing import Optional
import numpy as np
import gymnasium as gym

from hirise_dtm import HiriseDTM


class GridMarsEnv(gym.Env):
    """
    GridMarsEnv: A Gymnasium environment simulating a Mars rover navigating a gridworld.

    The rover knows its landing position and the target location but can only perceive its
    local surroundings through a limited field of view (FOV). The environment provides a
    matrix of altitude values within the FOV and a corresponding mask indicating which
    cells are observable.

    Observation Space (Dict):
        - "agent": 2D coordinates [x, y] of the rover.
        - "target": 2D coordinates [x, y] of the destination.
        - "local_map": (fov_distance x fov_distance) matrix of altitude values in the rover's FOV.
        - "mask": (fov_distance x fov_distance) binary matrix indicating which cells are observable in the local_map.

    Action Space (Discrete):
        - 8 possible movements corresponding to the cardinal and diagonal directions.

    :param dtm: a HiriseDTM object containing the terrain data.
    :param map_size: size of the gridworld (map_size x map_size).
    :param fov_distance: size of the rover's field of view (fov_distance x fov_distance).
    """

    def __init__(self, dtm: HiriseDTM, map_size: int = 200, fov_distance: int = 20, render_mode: str = 'rgb_array'):
        # Retrieving min and max altitude from map
        self._dtm = dtm
        min_altitude, max_altitude = self._dtm.get_lowest_highest_altitude()

        # map size and fov distance
        self.map_size = map_size
        self._fov_distance = fov_distance

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_global_location = np.array([-1, -1], dtype=np.int32)
        self._agent_relative_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Initialize agent field of view - will be properly set in reset()
        # setting mask to 0 as "uninitialized" state
        self._local_map_position = np.array([-1, -1], dtype=np.int32)
        self._local_map = np.zeros([fov_distance, fov_distance], dtype=np.float32)
        self._mask = np.zeros([fov_distance, fov_distance], dtype=np.bool_)

        # Define what the agent can observe
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, self.map_size - 1, shape=(2,), dtype=np.int32),    # [x, y] agent coordinates
                "target": gym.spaces.Box(0, self.map_size - 1, shape=(2,), dtype=np.int32),   # [x, y] goal coordinates

                # matrix with field of view, along with the mask that states whether the agent can see that pixel
                "local_map": gym.spaces.Box(min_altitude, max_altitude, shape=(fov_distance, fov_distance), dtype=np.float32),
                "mask": gym.spaces.MultiBinary((fov_distance, fov_distance))
            }
        )

        # Define what actions are available (8 directions)
        self.action_space = gym.spaces.Discrete(8)

        # Map action numbers to actual movements on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),        # Move right
            1: np.array([0, 1]),        # Move up
            2: np.array([-1, 0]),       # Move left
            3: np.array([0, -1]),       # Move down
            4: np.array([1, 1]),        # Move right-up
            5: np.array([-1, 1]),       # Move left-up
            6: np.array([1, -1]),       # Move right-down
            7: np.array([-1, -1]),      # Move left-down
        }

        # render mode for visualisation
        self.render_mode = render_mode

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        :param seed: Random seed for reproducible episodes
        :param options: Additional configuration (unused in this example)

        :return: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Select a random portion of the DTM map to use as an environment map
        self._local_map, self._local_map_position = self._dtm.get_portion_of_map(self.map_size)

        # Randomly place the agent anywhere on the grid
        self._agent_relative_location = self.np_random.integers(0, self.map_size, size=2, dtype=int)

        # Compute global position of agent
        self._agent_global_location = self._get_agent_global_position()

        # Retrieve the mask corresponding to the visible areas given the global agent position
        self._mask = self._dtm.get_field_of_view(self._agent_global_location, self._fov_distance)

        # Randomly place target, ensuring it is different from agent position
        self._target_location = self._agent_relative_location
        while np.array_equal(self._target_location, self._agent_relative_location):
            self._target_location = self.np_random.integers(
                0, self.map_size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: int):
        """
        Execute one timestep within the environment.

        :param action: The action to take (0-8 for directions)

        :return: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-8) to a movement direction
        direction = self._action_to_direction[action]

        # Boolean matrix 3x3 indicating which positions the agent can move to and which are forbidden (wall, big step, etc...)
        movements_allowed = self._dtm.get_possible_moves(self._agent_relative_location)

        # Convert the direction (dx, dy) into an index in the 3x3 movements_allowed matrix, to retrieve whether that direction is allowed
        move_index = (1 + direction[0], 1 + direction[1])

        # Check if the move is allowed
        if movements_allowed[move_index]:
            # Update agent position, ensuring it stays within grid bounds
            self._agent_relative_location = np.clip(
                self._agent_relative_location + direction, 0, self.map_size - 1
            )
        else:
            # Movement forbidden: stay in place.
            # todo: maybe add a small penalty for when the agent tries an illegal action
            pass

        # Global position
        self._agent_global_location = self._get_agent_global_position()

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_relative_location, self._target_location)

        # todo: add truncation mechanism if required (simulation ends before rover reaches the goal)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # todo: create a better reward model
        reward = 1 if terminated else 0

        # todo: create the logic according to which the rover cannot overcome certain obstacles (ad esempio non può scavalcare gradini più alti di tot cm)
        # todo: create an energy consumption mechanism depending on the slope

        observation = self._get_obs()
        info = self._get_info()
        self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment for human viewing.
        """
        # todo: farlo più carino con pygame, con una griglia effettiva che faccia vedere cosa succede (possibilmente con visuale
        #       corrispondente al FOV dell'agente, ma con possibilità di fare zoomm-out su intera mappa.
        #       guarda -> https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

        # todo: non includere solamente Agent e Target, ma prendere in considerazione anche le elevazioni della mappa
        #       contenute in "self._local_map" per disegnare ogni cella con un colore diverso

        if self.render_mode == "human":
            # Print a simple ASCII representation
            for y in range(self.map_size - 1, -1, -1):  # Top to bottom
                row = ""
                for x in range(self.map_size):
                    if np.array_equal([x, y], self._agent_relative_location):
                        row += "A "  # Agent
                    elif np.array_equal([x, y], self._target_location):
                        row += "T "  # Target
                    else:
                        row += ". "  # Empty
                print(row)
            print()

    def _get_agent_global_position(self):
        return self._local_map_position + self._agent_relative_location

    def _get_obs(self):
        """
        Convert internal state to observation format.

        :return: Observation with agent and target positions
        """
        return {
            "agent": self._agent_relative_location,
            "target": self._target_location,
            "local_map": self._local_map,
            "mask": self._mask
        }

    def _get_info(self):
        """
        Compute auxiliary information for debugging.

        :return: Debugging info that will be returned from the reset() and step() methods.
        """
        # todo: return additional information if needed
        return {
            "distance": np.linalg.norm(self._agent_relative_location - self._target_location, ord=1),
            "agent_relative_position": self._agent_relative_location,
            "agent_global_position": self._agent_global_location,
            "target_position": self._target_location,
        }
