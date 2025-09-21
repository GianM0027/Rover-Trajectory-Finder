import numpy as np
import gymnasium as gym
from typing import Optional
from hirise_dtm import HiriseDTM
import pygame

class GridMarsEnv(gym.Env):
    """
    GridMarsEnv: A Gymnasium environment simulating a Mars rover navigating a gridworld.

    The rover knows its landing position and the target location but can only perceive its
    local surroundings through a limited field of view (FOV). The environment provides a
    matrix of altitude values within the FOV and a corresponding mask indicating which
    cells are observable.

    Observation Space (Dict):
        - "agent": 2D coordinates [y, x] of the rover.
        - "target": 2D coordinates [y, x] of the destination.
        - "local_map": (fov_distance x fov_distance) matrix of altitude values in the rover's FOV.
        - "mask": (fov_distance x fov_distance) binary matrix indicating which cells are observable in the local_map.

    Action Space (Discrete):
        - 8 possible movements corresponding to the cardinal and diagonal directions.

    :param dtm: a HiriseDTM object containing the terrain data.
    :param map_size: size of the gridworld (map_size x map_size).
    :param fov_distance: ray of the rover's field of view. So that the full FOV size is a square matrix of size (fov_distance*2)+1.
    :param render_mode: possible values are "human", "ascii" and "rgb_array".
                        If set to "human", graphic rendering is performed by using pygame (does not work on Jupyter notebooks).
                        If set to "ascii", graphic rendering is performed by using ascii.
                        Otherwise, no rendering is performed.
    :param rover_max_step: maximum obstacle height the rover can overcome when moving on the map.
    :param rover_max_drop: maximum drop the rover can overcome when moving on the map.
    :param render_window_size: window size for the rendering of the environment when render_mode="human".
    """

    def __init__(self, dtm: HiriseDTM,
                 map_size: int = 200,
                 fov_distance: int = 20,
                 render_mode: str = 'rgb_array',
                 rover_max_step=0.3,
                 rover_max_drop=0.5,
                 render_window_size=512):
        # Retrieving min and max altitude from map
        self._dtm = dtm
        min_altitude, max_altitude = self._dtm.get_lowest_highest_altitude()

        # map size and fov distance
        self.map_size = map_size
        self._fov_distance = fov_distance
        self._fov_matrix_size = (fov_distance*2)+1

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_global_location = np.array([-1, -1], dtype=np.int32)
        self._agent_relative_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Initialize local map - it will be the area extracted from the DTM where the agent will navigate
        # setting local map position to -1 as "uninitialized" state
        self._local_map_position = np.array([-1, -1], dtype=np.int32)
        self._local_map = np.zeros([map_size, map_size], dtype=np.float32)

        # Initialize agent field of view - will be properly set in reset()
        # setting mask to 0 as "uninitialized" state
        self._fov_coordinates = np.zeros([self._fov_matrix_size, self._fov_matrix_size], dtype=np.float32)
        self._fov_mask = np.zeros([self._fov_matrix_size, self._fov_matrix_size], dtype=np.bool_)

        # Define what the agent can observe
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, self.map_size - 1, shape=(2,), dtype=np.int32),    # [y, x] agent coordinates
                "target": gym.spaces.Box(0, self.map_size - 1, shape=(2,), dtype=np.int32),   # [y, x] goal coordinates

                # matrix with field of view, along with the mask that states whether the agent can see that pixel
                "local_fov_map": gym.spaces.Box(min_altitude, max_altitude, shape=(self._fov_matrix_size, self._fov_matrix_size), dtype=np.float32),
                "fov_mask": gym.spaces.MultiBinary((self._fov_matrix_size, self._fov_matrix_size))
            }
        )

        # Define what actions are available (8 directions)
        self.action_space = gym.spaces.Discrete(8)

        # Map action numbers to actual movements on the grid
        self._action_to_direction = {
            0: np.array([0, 1]),        # Move right (y, x+1)
            1: np.array([1, 0]),        # Move down (y+1, x)
            2: np.array([0, -1]),       # Move left (y, x-1)
            3: np.array([-1, 0]),       # Move up (y-1, x)
            4: np.array([1, 1]),        # Move right-down (y+1, x+1)
            5: np.array([1, -1]),       # Move left-down (y+1, x-1)
            6: np.array([-1, 1]),       # Move right-up (y-1, x+1)
            7: np.array([-1, -1]),      # Move left-up (y-1, x-1)
        }

        self._action_to_direction_string = {
            0: "right",
            1: "down",
            2: "left",
            3: "up",
            4: "right-down",
            5: "left-down",
            6: "right-up",
            7: "left-up",
        }

        # render mode for visualisation
        self.render_mode = render_mode

        # rover max jump and max drop
        self.rover_max_step = rover_max_step
        self.rover_max_drop = rover_max_drop
        self.current_move_allowed_flag = True

        # rendering parameters
        self.render_window_size = render_window_size
        self.window = None
        self.clock = None

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
        self._update_fov_coordinates()

        # Compute global position of agent
        self._agent_global_location = self._compute_agent_global_position()

        # Retrieve the mask corresponding to the visible areas given the global agent position
        self._fov_mask = self._dtm.get_fov_mask(self._agent_global_location, self._fov_distance)

        # Randomly place target, ensuring it is different from agent position
        self._target_location = self._agent_relative_location
        while np.array_equal(self._target_location, self._agent_relative_location):
            self._target_location = self.np_random.integers(
                0, self.map_size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()
        self.render()

        return observation, info

    def step(self, action: int, verbose=False):
        """
        Execute one timestep within the environment.

        :param action: The action to take (0-8 for directions)

        :return: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-8) to a movement direction
        direction = self._action_to_direction[action]

        # Boolean matrix 3x3 indicating which positions the agent can move to and which are forbidden (wall, big step, etc...)
        movements_allowed = self._dtm.get_possible_moves(position=self._agent_global_location,
                                                         moves=self._action_to_direction,
                                                         max_step=self.rover_max_step,
                                                         max_drop=self.rover_max_drop)

        # Convert the direction (dx, dy) into an index in the 3x3 movements_allowed matrix, to retrieve whether that direction is allowed
        move_index = (1 + direction[0], 1 + direction[1])

        # Check if the move is allowed
        if movements_allowed[move_index]:
            # Update agent position, ensuring it stays within grid bounds
            self._agent_relative_location = np.clip(
                self._agent_relative_location + direction, 0, self.map_size - 1
            )
            self._update_fov_coordinates()
            self.current_move_allowed_flag = True
        else:
            # Movement forbidden: stay in place.
            self.current_move_allowed_flag = False
            # todo: maybe add a small penalty for when the agent tries an illegal action like bumping on an obstacle/wall
            pass

        # Global position and mask update
        self._agent_global_location = self._compute_agent_global_position()
        self._fov_mask = self._dtm.get_fov_mask(self._agent_global_location, self._fov_distance)

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_relative_location, self._target_location)

        # todo: add truncation mechanism if required (simulation ends before rover reaches the goal because max_steps reached)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # todo: create a better reward model
        reward = 1 if terminated else 0

        # todo: create an energy consumption mechanism depending on the slope

        observation = self._get_obs()
        info = self._get_info()
        self.render()

        if verbose:
            # todo: add other stuff for debugging purposes
            print(f"Action Selected: {self._action_to_direction_string[action]}")
            print(f"Movement allowed: {self.current_move_allowed_flag}")

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.render_pygame()
        elif self.render_mode == "ascii":
            self.render_ascii()
        else:
            pass

    def render_pygame(self):
        # todo: aggiungere una versione velocizzata per la simulazione
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.render_window_size, self.render_window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.render_window_size, self.render_window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.render_window_size // self.map_size

        # Map drown according to altitudes (grayscale)
        min_alt, max_alt = np.nanmin(self._local_map), np.nanmax(self._local_map)
        norm = (self._local_map - min_alt) / (max_alt - min_alt + 1e-9)

        for y in range(self.map_size):
            for x in range(self.map_size):
                val = norm[y, x]
                color = (int(255 * (1 - val)), int(255 * (1 - val)), int(255 * (1 - val)))
                rect = pygame.Rect(
                    x * pix_square_size,
                    y * pix_square_size,
                    pix_square_size,
                    pix_square_size,
                )
                pygame.draw.rect(canvas, color, rect)

        # Target
        ty, tx = self._target_location
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                tx * pix_square_size,
                ty * pix_square_size,
                pix_square_size,
                pix_square_size,
            ),
        )

        # Agent
        ay, ax = self._agent_relative_location
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (int((ax + 0.5) * pix_square_size), int((ay + 0.5) * pix_square_size)),
            pix_square_size / 3,
        )

        # Highlighted FOV
        for (fy, fx) in self._fov_coordinates:
            rect = pygame.Rect(
                fx * pix_square_size,
                fy * pix_square_size,
                pix_square_size,
                pix_square_size,
            )
            pygame.draw.rect(canvas, (0, 255, 0), rect, width=1)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(1)  # FPS

    def render_ascii(self):
        """
        Render the environment for human viewing.
        """

        if self.render_mode == "human":
            # Print a simple ASCII representation
            for y in range(self.map_size):
                row = ""
                for x in range(self.map_size):
                    if np.array_equal([y, x], self._agent_relative_location):
                        row += "A "  # Agent
                    elif np.array_equal([y, x], self._target_location):
                        row += "T "  # Target
                    elif (y, x) in self._fov_coordinates:
                        row += "* "  # Agent FOV
                    else:
                        row += ". "  # Empty
                print(row)
            print()

    def _update_fov_coordinates(self):
        agent_y, agent_x = self._agent_relative_location

        fov_x_low, fov_x_high = (max(agent_x - self._fov_distance, 0),
                                 min(agent_x + self._fov_distance, self.map_size - 1))
        fov_y_low, fov_y_high = (max(agent_y - self._fov_distance, 0),
                                 min(agent_y + self._fov_distance, self.map_size - 1))

        self._fov_coordinates = [
            (y, x)
            for y in range(fov_y_low, fov_y_high + 1)
            for x in range(fov_x_low, fov_x_high + 1)
        ]

    def _get_fov_altitudes(self):
        rows, cols = zip(*self._fov_coordinates)
        row_min, row_max = min(rows), max(rows) + 1
        col_min, col_max = min(cols), max(cols) + 1

        fov_values = self._local_map[row_min:row_max, col_min:col_max]

        # Add rows with np.inf if rover close to the map edge
        target_size = (self._fov_distance * 2) + 1
        while fov_values.shape[0] < target_size:
            fov_values = np.vstack([fov_values, np.full((1, fov_values.shape[1]), np.inf)])

        # Add columns with np.inf if rover close to the map edge
        while fov_values.shape[1] < target_size:
            fov_values = np.hstack([fov_values, np.full((fov_values.shape[0], 1), np.inf)])

        return fov_values

    def _compute_agent_global_position(self):
        # local map position in (width,height) format + agent location in (y,x) format
        return self._local_map_position + self._agent_relative_location

    def _get_obs(self):
        """
        Convert internal state to observation format.

        :return: Observation with agent and target positions
        """
        return {
            "agent": self._agent_relative_location,
            "target": self._target_location,
            "local_fov_map": self._get_fov_altitudes(),
            "mask": self._fov_mask
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
