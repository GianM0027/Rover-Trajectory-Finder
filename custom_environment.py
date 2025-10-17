import heapq
import numpy as np
import gymnasium as gym
from typing import Optional
from hirise_dtm import HiriseDTM
import pygame
from collections import deque


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
    :param draw_visited_locations: when render_mode is set to human. This flag sets whether to draw visited locations on the map.
    :param rover_max_step: maximum obstacle height the rover can overcome when moving on the map.
    :param rover_max_drop: maximum drop the rover can overcome when moving on the map.
    :param render_window_size: window size for the rendering of the environment when render_mode="human".
    """

    def __init__(self,
                 dtm: HiriseDTM = None,
                 map_size: int = 200,
                 fov_distance: int = 20,
                 render_mode: str = 'rgb_array',
                 draw_visited_locations: bool = False,
                 rover_max_step=0.3,
                 rover_max_drop=0.5,
                 previous_positions_in_obs=5,
                 rover_max_number_of_steps=1000,
                 render_window_size=512):

        # Retrieving min and max altitude from map
        self._dtm = dtm
        self.min_altitude, self.max_altitude = self._dtm.get_lowest_highest_altitude()

        # map size and fov distance
        self.map_size = map_size
        self._fov_distance = fov_distance
        self._fov_matrix_size = (fov_distance * 2) + 1

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_global_location = np.array([-1, -1], dtype=np.int32)
        self._agent_relative_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Previous positions of the agent
        self._n_previous_positions_in_obs = previous_positions_in_obs
        self._previous_positions = deque(maxlen=previous_positions_in_obs)

        # Initialize local map - it will be the area extracted from the DTM where the agent will navigate
        # setting local map position to -1 as "uninitialized" state
        self._local_map_position = np.array([-1, -1], dtype=np.int32)
        self._local_map = np.zeros([map_size, map_size], dtype=np.float32)

        self.visited_locations = np.zeros([map_size, map_size],
                                          dtype=np.int32)  # how many times the agent visited that location
        self._detected_altitudes = np.zeros([self.map_size, self.map_size],
                                            dtype=np.bool_)  # did the agent registered the altitude of that location?

        # Initialize agent field of view - will be properly set in reset()
        # setting mask to 0 as "uninitialized" state
        self._fov_coordinates = np.zeros([self._fov_matrix_size, self._fov_matrix_size], dtype=np.float32)
        self._fov_mask = np.zeros([self._fov_matrix_size, self._fov_matrix_size], dtype=np.bool_)

        # Define what the agent can observe
        # Define the bounds for the observation space
        self._relative_altitudes_clipping = 3.0
        obs_shape = (4, self.map_size, self.map_size)

        # Replace the gym.spaces.Dict with this:
        self.observation_space = gym.spaces.Box(
            low=-self._relative_altitudes_clipping,
            high=self._relative_altitudes_clipping,
            shape=obs_shape,
            dtype=np.float32
        )

        # Define what actions are available (8 directions)
        self.action_space = gym.spaces.Discrete(8)

        # Map action numbers to actual movements on the grid
        self._action_to_direction = {
            0: np.array([0, 1]),  # Move right (y, x+1)
            1: np.array([1, 0]),  # Move down (y+1, x)
            2: np.array([0, -1]),  # Move left (y, x-1)
            3: np.array([-1, 0]),  # Move up (y-1, x)
            4: np.array([1, 1]),  # Move right-down (y+1, x+1)
            5: np.array([1, -1]),  # Move left-down (y+1, x-1)
            6: np.array([-1, 1]),  # Move right-up (y-1, x+1)
            7: np.array([-1, -1]),  # Move left-up (y-1, x-1)
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

        # rover movements parameters
        self.rover_max_step = rover_max_step
        self.rover_max_drop = rover_max_drop
        self.current_move_allowed_flag = True
        self.rover_max_number_of_steps = rover_max_number_of_steps
        self.rover_steps_counter = 0
        self.best_distance_so_far = None

        # rendering parameters
        self.render_mode = render_mode
        self.render_window_size = render_window_size
        self.draw_visited_locations = draw_visited_locations
        self.window = None
        self.clock = None

        # Flag for seed setting
        self.is_first_execution = True

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        :param seed: Random seed for reproducible episodes
        :param options: Additional configuration (unused in this example)

        :return: (observation, info) for the initial state
        """
        # Seed the random number generator
        if self.is_first_execution:
            super().reset(seed=seed)
            self.is_first_execution = False

        # At the beginning of each episode, randomly rotate the global dtm for data augmentation
        # fixme: check if this method works and it is not too computationally expensive
        k = np.random.randint(0, 4)
        self._dtm.numpy_image = np.rot90(self._dtm.numpy_image, k)

        # Select a random portion of the DTM map to use as an environment map
        self._local_map, self._local_map_position = self._dtm.get_portion_of_map(self.map_size, random_rotation=True)
        self.rover_steps_counter = 0
        self.visited_locations = np.zeros([self.map_size, self.map_size], dtype=np.int32)

        # Randomly place the agent anywhere on the grid
        self._agent_relative_location = self.np_random.integers(0, self.map_size, size=2, dtype=int)
        self._update_fov_coordinates()
        self._update_visited_locations()

        # Compute global position of agent
        self._agent_global_location = self._compute_agent_global_position()

        # Retrieve the mask corresponding to the visible areas given the global agent position
        self._fov_mask = self._dtm.get_fov_mask(self._agent_global_location, self._fov_distance,
                                                self._action_to_direction)

        # Randomly place target, ensuring it is different from agent position
        self._target_location = self._agent_relative_location
        while np.array_equal(self._target_location, self._agent_relative_location):
            self._target_location = self.np_random.integers(
                0, self.map_size, size=2, dtype=int
            )

        self._previous_positions.append(self._agent_relative_location)
        self.best_distance_so_far = self._get_manhattan_distance(self._agent_relative_location, self._target_location)
        self._update_detected_altitudes()
        observation = self._get_obs()
        info = self._get_info()
        self.render()

        return observation, info

    def step(self, action: int, verbose=False):
        """
        Execute one timestep within the environment.

        :param action: The action to take (0-8 for directions)
        :param verbose: (default: False) Print real-time information about the agent behavior.

        :return: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-8) to a movement direction
        self.rover_steps_counter += 1
        direction = self._action_to_direction[action]

        # Boolean matrix 3x3 indicating which positions the agent can move to and which are forbidden (wall, big step, etc...)
        movements_allowed = self._dtm.get_possible_moves(position=self._agent_global_location,
                                                         moves=self._action_to_direction,
                                                         max_step=self.rover_max_step,
                                                         max_drop=self.rover_max_drop,
                                                         local_map_size=self.map_size,
                                                         local_map_position=self._local_map_position)

        # Convert the direction (dx, dy) into an index in the 3x3 movements_allowed matrix, to retrieve whether that direction is allowed
        move_index = (1 + direction[0], 1 + direction[1])

        # Check if the move is allowed
        if movements_allowed[move_index[0], move_index[1]]:
            # Update agent position, ensuring it stays within grid bounds
            self._agent_relative_location = self._agent_relative_location + direction
            self._update_fov_coordinates()
            self.current_move_allowed_flag = True

            self._update_detected_altitudes()
        else:
            # Movement forbidden: stay in place.
            self.current_move_allowed_flag = False

        self._update_visited_locations()
        self._previous_positions.append(self._agent_relative_location)

        # Global position and mask update
        self._agent_global_location = self._compute_agent_global_position()
        self._fov_mask = self._dtm.get_fov_mask(self._agent_global_location, self._fov_distance,
                                                self._action_to_direction)

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_relative_location, self._target_location)
        truncated = self.rover_max_number_of_steps == self.rover_steps_counter

        # Current reward
        reward = self._compute_reward(terminated, truncated)

        # todo: create an energy consumption mechanism depending on the slope.
        #       E.g if last pixel was higher than current -> self.rover_steps_counter += 0.5 instead of 1

        observation = self._get_obs()
        info = self._get_info()
        self.render()

        if verbose:
            agent_y, agent_x = self._agent_relative_location
            print(f"Agent Position (y, x): {self._agent_relative_location}")
            print(f"This location was visited {self.visited_locations[agent_y, agent_x]} times")
            print(f"Actions remained: {self.rover_max_number_of_steps - self.rover_steps_counter}")
            print(f"Action Selected: {self._action_to_direction_string[action]}")
            print(f"Movement allowed: {self.current_move_allowed_flag}")
            print(f"Reward: {reward}")

            if terminated:
                print("The Rover reached its goal. Simulation concluded")
            if truncated:
                print(
                    f"The Rover have not reached its goal withing {self.rover_max_number_of_steps} steps. Simulation concluded")
            print()

        return observation, reward, terminated, truncated, info

    def _compute_reward(self, terminated, truncated):
        reward = 0.0
        success_reward = 10.0
        time_penalty = success_reward / self.rover_max_number_of_steps
        current_distance = self._get_manhattan_distance(self._agent_relative_location, self._target_location)

        # 1. Reward agent for making real progress (setting a new best distance from the goal)
        if current_distance < self.best_distance_so_far:
            reward += 5*time_penalty # todo: tweak the multiplier as needed (5-10)
            self.best_distance_so_far = current_distance

        # 2. Big reward for reaching the goal
        if terminated and not truncated:
            reward += success_reward

        return reward - time_penalty

    def render(self):
        if self.render_mode == "human":
            self.render_pygame()
        elif self.render_mode == "ascii":
            self.render_ascii()
        else:
            pass

    def render_pygame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.render_window_size, self.render_window_size))
            pygame.display.set_caption("GridMarsEnv")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # --- Handle events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Quit
                    pygame.quit()
                    exit()
                elif event.key == pygame.K_p:  # Pause
                    paused = True
                    while paused:
                        for e in pygame.event.get():
                            if e.type == pygame.QUIT:
                                pygame.quit()
                                exit()
                            if e.type == pygame.KEYDOWN:
                                if e.key == pygame.K_p:
                                    paused = False
                                elif e.key == pygame.K_q:
                                    pygame.quit()
                                    exit()
                        self.clock.tick(10)
                elif event.key == pygame.K_y:  # Speed up toggle
                    if hasattr(self, "_fast_mode"):
                        self._fast_mode = not self._fast_mode
                    else:
                        self._fast_mode = True

        # --- Draw terrain into a raw map surface ---
        finite_map = np.where(np.isfinite(self._local_map), self._local_map, np.nan)
        min_alt, max_alt = np.nanmin(finite_map), np.nanmax(finite_map)

        # Normalize safely
        norm = (finite_map - min_alt) / (max_alt - min_alt + 1e-9)
        norm = np.nan_to_num(norm, nan=0.5)
        norm = np.clip(norm, 0.0, 1.0)

        raw_surface = pygame.Surface((self.map_size, self.map_size))
        for y in range(self.map_size):
            for x in range(self.map_size):
                if np.isinf(self._local_map[y, x]):
                    color = (0, 0, 0)  # very high wall
                else:
                    gray = int(255 * (1 - norm[y, x]))
                    color = (gray, gray, gray)
                raw_surface.set_at((x, y), color)

                # If visited, draw a light blue dot on location
                if self.visited_locations[y, x] > 0 and self.draw_visited_locations:
                    raw_surface.set_at((x, y), (173, 216, 230))

        # Scale terrain to window size (no gaps, no leftover border)
        canvas = pygame.transform.scale(raw_surface,
                                        (self.render_window_size, self.render_window_size))

        pix_square_size = self.render_window_size / self.map_size

        # --- Draw target ---
        ty, tx = self._target_location
        pygame.draw.rect(canvas, (255, 0, 0),
                         pygame.Rect(tx * pix_square_size, ty * pix_square_size,
                                     pix_square_size, pix_square_size))

        # --- Draw agent ---
        ay, ax = self._agent_relative_location
        pygame.draw.circle(canvas, (0, 0, 255),
                           (int((ax + 0.5) * pix_square_size),
                            int((ay + 0.5) * pix_square_size)),
                           int(pix_square_size // 3))

        # --- Draw FOV highlights ---
        for fy, fx in self._fov_coordinates:
            rel_idx = np.array([fy, fx]) - np.array(self._agent_relative_location)
            fov_idx = rel_idx + np.array([self._fov_distance, self._fov_distance])

            if (0 <= fov_idx[0] < self._fov_mask.shape[0] and
                    0 <= fov_idx[1] < self._fov_mask.shape[1]):
                rect = pygame.Rect(fx * pix_square_size, fy * pix_square_size,
                                   pix_square_size, pix_square_size)
                if self._fov_mask[tuple(fov_idx)]:
                    pygame.draw.rect(canvas, (0, 255, 0), rect, width=1)
                else:
                    pygame.draw.rect(canvas, (255, 0, 0), rect, width=1)

        # --- Draw legend with semi-transparent background ---
        font = pygame.font.SysFont("Arial", 16)
        legend_texts = ["P = Pause - Q = Quit - Y = Speed Up"]

        legend_width = max(font.size(text)[0] for text in legend_texts) + 10
        legend_height = len(legend_texts) * 20 + 10
        legend_surface = pygame.Surface((legend_width, legend_height), pygame.SRCALPHA)
        legend_surface.fill((255, 255, 255, 180))  # white with alpha = 180

        for i, text in enumerate(legend_texts):
            label = font.render(text, True, (0, 0, 0))
            legend_surface.blit(label, (5, 5 + i * 20))

        canvas.blit(legend_surface, (5, 5))

        # --- Push canvas to window ---
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()

        # --- Adjust FPS ---
        fps = 60 if getattr(self, "_fast_mode", False) else 5
        self.clock.tick(fps)

    def render_ascii(self, path=None):
        """
        Render the environment for human viewing.
        """
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
                    if path is not None and any((y, x) == tuple(p) for p in path):
                        row += "+ "
                    else:
                        row += ". "  # Empty
            print(row)
        print()

    def find_best_path(self, use_slope_cost=False, use_only_detected_altitudes=False):
        start = tuple(self._agent_relative_location)
        target = tuple(self._target_location)

        if use_only_detected_altitudes:
            if not self._detected_altitudes[start] or not self._detected_altitudes[target]:
                return None

        adjacency_list = self._dtm.get_adjacency_list(moves=self._action_to_direction,
                                                      max_step=self.rover_max_step,
                                                      max_drop=self.rover_max_drop,
                                                      local_map_size=self.map_size,
                                                      local_map_position=self._local_map_position)

        # Dijkstra: priority queue [(cost, node)]
        heap = [(0, start)]
        costs = {start: 0}
        parent = {start: None}

        while heap:
            current_cost, node = heapq.heappop(heap)

            if node == target:
                path = []
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return np.array(path[::-1], dtype=object)

            if current_cost > costs[node]:
                continue

            for neighbor in adjacency_list.get(node, []):
                # Ignoring nodes whose altitude is unknown if the flag is set
                if use_only_detected_altitudes and not self._detected_altitudes[neighbor[0], neighbor[1]]:
                    continue

                edge_cost = 1.0

                if use_slope_cost:
                    y1, x1 = node
                    y2, x2 = neighbor

                    altitude_diff = self._local_map[y2, x2] - self._local_map[y1, x1]

                    edge_cost += abs(altitude_diff) * 0.5

                new_cost = current_cost + edge_cost

                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    parent[neighbor] = node
                    heapq.heappush(heap, (new_cost, neighbor))

        return None

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

    def _get_fov_map_w_altitudes(self):
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

        # Create a matrix of positions
        fov_positions = np.empty(fov_values.shape, dtype=object)
        for i in range(fov_values.shape[0]):
            for j in range(fov_values.shape[1]):
                fov_positions[i, j] = (row_min + i, col_min + j)

        return fov_values, fov_positions

    def _update_detected_altitudes(self):
        local_fov_values, local_fov_positions = self._get_fov_map_w_altitudes()

        seen_altitudes_values = local_fov_values[self._fov_mask.astype(np.bool_)]
        seen_altitudes_positions = local_fov_positions[self._fov_mask.astype(np.bool_)]

        for altitude, position in zip(seen_altitudes_values, seen_altitudes_positions):
            y, x = position
            y = min(max(y, 0), self.map_size - 1)
            x = min(max(x, 0), self.map_size - 1)
            self._detected_altitudes[y, x] = 1

    def _compute_agent_global_position(self):
        # local map position in (width,height) format + agent location in (y,x) format
        return self._local_map_position + self._agent_relative_location

    def _process_observation(self):
        padding_number = 0.0
        map_shape = (self.map_size, self.map_size)

        # --- Channel 0: relative normalized altitudes ---
        channel_zero = np.where(self._detected_altitudes.astype(np.bool_), self._local_map, np.nan).astype(np.float32)
        padding_mask = np.isnan(channel_zero)
        center_altitude = self._local_map[self._agent_relative_location[0], self._agent_relative_location[1]]
        delta = self._local_map - center_altitude
        channel_zero = np.where(
            delta >= 0,
            delta / self.rover_max_step,
            delta / abs(self.rover_max_drop)
        )
        channel_zero = np.clip(channel_zero, -self._relative_altitudes_clipping, self._relative_altitudes_clipping)
        channel_zero[padding_mask] = padding_number

        # --- Channel 1: mask (0 = padding / invalid) ---
        channel_one = np.ones_like(channel_zero, dtype=np.float32)
        channel_one[padding_mask] = padding_number

        # --- Channel 2: agent (1), previous position (-1) ---
        channel_two = np.full(map_shape, padding_number, dtype=np.float32)
        previous_position_values = np.linspace(start=0, stop=1, num=len(self._previous_positions) + 1)[1:-1]
        for previous_location, location_value in zip(self._previous_positions, previous_position_values):
            channel_two[previous_location[0], previous_location[1]] = location_value
        channel_two[self._agent_relative_location[0], self._agent_relative_location[1]] = 1.0

        # --- Channel 3: target ---
        channel_three = np.full(map_shape, padding_number, dtype=np.float32)
        channel_three[self._target_location[0], self._target_location[1]] = 1.0

        return np.stack([channel_zero, channel_one, channel_two, channel_three], axis=0)

    def _get_obs(self):
        """
        Convert internal state to observation format.

        :return: Observation with agent and target positions
        """
        return self._process_observation()

    def _update_visited_locations(self):
        y, x = self._agent_relative_location
        self.visited_locations[y, x] += 1

    def _get_info(self):
        """
        Compute auxiliary information for debugging.

        :return: Debugging info that will be returned from the reset() and step() methods.
        """
        local_fov_values, local_fov_positions = self._get_fov_map_w_altitudes()

        return {
            "distance": np.linalg.norm(self._agent_relative_location - self._target_location, ord=1),
            "agent_relative_position": self._agent_relative_location,
            "agent_global_position": self._agent_global_location,
            "target_position": self._target_location,
            "n_previous_positions_in_obs": self._n_previous_positions_in_obs,
            "previous_positions": self._previous_positions,

            # fov information
            "local_fov_values": local_fov_values,  # all altitudes in FOV
            "local_fov_positions": local_fov_positions,

            # coordinates of FOV with local map as reference system (not global)
            "fov_mask": self._fov_mask.astype(np.bool_),
            
            # Integer matrix indicating the number of times the agent stepped on that location
            "visited_locations": self.visited_locations,

            # Local map the agent is navigating on
            "local_map": self._local_map,
        }

    @classmethod
    def _get_manhattan_distance(cls, pos1, pos2):
        return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])
