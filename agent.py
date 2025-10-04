import json
import os
import random
import torch
from torch import nn, optim
from tqdm import tqdm
from custom_environment import GridMarsEnv
from policy_network import PolicyNetwork
import numpy as np
from experience_manager import ExperienceManager


class Agent:
    def __init__(self,
                 policy_network,
                 fov_distance,
                 dtm_file,
                 map_size,
                 seed=None,
                 max_number_of_steps=None,
                 max_step_height=1,
                 max_drop_height=1):

        self.seed = seed
        self.policy_network = policy_network
        self.max_number_of_steps = max_number_of_steps
        self.max_step_height = max_step_height
        self.max_drop_height = max_drop_height
        self.fov_distance = fov_distance
        self.dtm_file = dtm_file

        self.mars_environment = GridMarsEnv(
            dtm=dtm_file,
            map_size=map_size,
            fov_distance=fov_distance,
            rover_max_step=max_step_height,
            rover_max_drop=max_drop_height,
            rover_max_number_of_steps=max_number_of_steps)

    def run_simulation(self, max_episodes=None, use_policy_network=False, verbose=False, device="cuda", sample_action=False, render_mode="human"):
        self.mars_environment.render_mode = render_mode
        if use_policy_network:
            self.policy_network.eval()
            self.policy_network.to(device)

        n_episode = 0
        while True:
            print(f"Episode #{n_episode + 1}")
            observation, _ = self.mars_environment.reset(seed=self.seed)
            terminated = False
            truncated = False

            while not terminated and not truncated:
                if use_policy_network:
                    processed_observation = self.preprocess_observation_single_matrix(observation, device)
                    action_probs, value = self.policy_network(processed_observation)

                    if sample_action:
                        action = torch.distributions.Categorical(logits=action_probs).sample().item()
                    else:
                        action = torch.argmax(action_probs).item()
                else:
                    action = np.random.randint(8)

                observation, reward, terminated, truncated, info = self.mars_environment.step(action, verbose=verbose)

            n_episode += 1
            if max_episodes is not None and n_episode >= max_episodes:
                break

    @classmethod
    def preprocess_observation_single_matrix(cls, observation, device="cuda", eps=1e-4):
        padding_number = 0
        agent_position = observation["agent_pos"]
        target_position = observation["target_pos"]

        local_map = observation["local_map"]
        local_map_mask = observation["local_map_mask"]
        visited_locations = observation["visited_locations"]
        map_shape = local_map.shape

        # Channel 0: altitude with mask
        channel_zero = np.array([
            [local_map[y, x] if local_map_mask[y, x] else np.nan
             for x in range(map_shape[1])]
            for y in range(map_shape[0])
        ], dtype=np.float32)

        # Channel 1: visited locations
        channel_one = visited_locations.astype(np.float32)

        # Channel 2: agent position
        channel_two = np.full(map_shape, fill_value=padding_number, dtype=np.float32)
        channel_two[agent_position[0], agent_position[1]] = 1.0

        # Channel 3: target position
        channel_three = np.full(map_shape, fill_value=padding_number, dtype=np.float32)
        channel_three[target_position[0], target_position[1]] = 1.0

        # Stack channels
        x = np.stack([channel_zero, channel_one, channel_two, channel_three], axis=0)  # (C, H, W)

        # Normalize channel 0 (altitude)
        altitude = x[0]
        mask = np.isnan(altitude)
        min_val = np.nanmin(altitude)
        max_val = np.nanmax(altitude)
        if min_val == max_val:
            x[0] = np.zeros_like(altitude)
        else:
            x[0] = ((altitude - min_val) / (max_val - min_val)) + eps
        x[0][mask] = padding_number  # sentinel for NaN

        # Normalize channel 1 (visited locations)
        x[1] = np.log1p(x[1])
        mask = channel_one == 0.
        x[1][mask] = padding_number
        x[1][~mask] += eps

        # Convert to torch tensor and add batch dimension
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    def train(self,
              training_episodes=1000,
              batch_size=512,
              minibatch_size=16,
              epochs=1,
              device='cuda',
              clip_ratio=0.2,
              c1=0.5,
              c2=0.01,
              learning_rate=1e-5,
              weights_path=None,
              training_info_path=None,
              training_losses_path=None,
              training_parameters_path=None,
              step_verbose=False,
              render_mode="rgb_array"):
        self.mars_environment.render_mode = render_mode

        mse_loss = nn.MSELoss()
        optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.policy_network.train()
        self.policy_network.to(device)

        experience_manager = ExperienceManager(batch_size=batch_size, minibatch_size=minibatch_size)
        all_episodes_info = []
        ppo_losses = []

        if training_parameters_path:
            parent_dir = os.path.dirname(training_parameters_path)
            os.makedirs(parent_dir, exist_ok=True)

            training_parameters = {
                "map_size": self.mars_environment.map_size,
                "fov_distance": self.fov_distance,
                "max_number_of_steps": self.max_number_of_steps,
                "max_step_height": self.max_step_height,
                "max_drop_height": self.max_drop_height,

                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "minibatch_size": minibatch_size,
                "epochs": epochs,
                "clip_ratio": clip_ratio,
                "c1": c1,
                "c2": c2,
            }

            print(f"Saving training parameters to {training_parameters_path}")
            with open(training_parameters_path, 'w') as f:
                json.dump(training_parameters, f, indent=4)

        # Loop for episodes
        for _ in tqdm(range(training_episodes), desc="Training Episodes"):
            observation, _ = self.mars_environment.reset(seed=self.seed)
            terminated = False
            truncated = False

            # Retrieve optimal paths and check whether there is one
            optimal_path_no_slope = self.mars_environment.find_best_path(use_slope_cost=False)
            optimal_path_w_slope = self.mars_environment.find_best_path(use_slope_cost=True)

            if optimal_path_no_slope is not None:
                optimal_path_no_slope = [[int(pos[0]), int(pos[1])] for pos in optimal_path_no_slope]
            else:
                optimal_path_no_slope = []

            if optimal_path_w_slope is not None:
                optimal_path_w_slope = [[int(pos[0]), int(pos[1])] for pos in optimal_path_w_slope]
            else:
                optimal_path_w_slope = []

            info_to_save = {
                "agent_positions": [],
                "rewards": [],
                "terminated": [],
                "truncated": [],
                "optimal_path_no_slope": optimal_path_no_slope,
                "optimal_path_w_slope": optimal_path_w_slope,
                "episode_length": 0
            }

            while not terminated and not truncated:
                info_to_save["episode_length"] += 1

                with torch.no_grad():
                    processed_observation = self.preprocess_observation_single_matrix(observation, device)
                    action_probs, value = self.policy_network(processed_observation)

                action = torch.distributions.Categorical(logits=action_probs).sample().item()

                observation, reward, terminated, truncated, info = self.mars_environment.step(action, verbose=step_verbose)
                processed_observation = self.preprocess_observation_single_matrix(observation, device)
                prob_of_taken_action = action_probs.squeeze()[action]

                experience_manager.appendTrajectory(
                    state=processed_observation.squeeze().cpu().numpy(),
                    action=action,
                    action_prob=prob_of_taken_action.cpu().numpy(),
                    reward=reward,
                    value=value.squeeze().cpu().numpy(),
                    terminated=terminated,
                    truncated=truncated
                )

                info_to_save["agent_positions"].append(observation["agent_pos"].tolist())
                info_to_save["rewards"].append(reward)
                info_to_save["terminated"].append(terminated)
                info_to_save["truncated"].append(truncated)

                if experience_manager.is_full():
                    with torch.no_grad():
                        processed_observation = self.preprocess_observation_single_matrix(observation, device)
                        _, next_value = self.policy_network(processed_observation)

                    dataloader = experience_manager.get_batches(device=device, next_value=next_value.cpu().numpy())

                    update_losses = []
                    for _ in range(epochs):
                        for states, actions, old_action_probs, advantages, returns in dataloader:
                            current_action_dist, values = self.policy_network(states)
                            current_action_probs = current_action_dist.gather(
                                dim=1,
                                index=actions.unsqueeze(-1).to(torch.int64)
                            ).squeeze(-1)

                            prob_ratio = current_action_probs / (old_action_probs + 1e-10)
                            clipped_ratio = torch.clamp(prob_ratio, 1 - clip_ratio, 1 + clip_ratio)

                            actor_loss = -torch.min(prob_ratio * advantages, clipped_ratio * advantages).mean()
                            critic_loss = mse_loss(values.squeeze(-1), returns)
                            entropy = -(current_action_dist * torch.log(current_action_dist + 1e-10)).mean()

                            total_loss = actor_loss + (c1 * critic_loss) - (c2 * entropy)
                            update_losses.append(total_loss.item())

                            optimizer.zero_grad()
                            total_loss.backward()
                            optimizer.step()

                    experience_manager.clear()
                    ppo_losses.append(sum(update_losses) / len(update_losses))

            all_episodes_info.append(info_to_save)

        if training_info_path:
            parent_dir = os.path.dirname(training_info_path)
            os.makedirs(parent_dir, exist_ok=True)

            print(f"Saving training info to {training_info_path}")
            with open(training_info_path, 'w') as f:
                json.dump(all_episodes_info, f, indent=4)

        if training_losses_path:
            parent_dir = os.path.dirname(training_losses_path)
            os.makedirs(parent_dir, exist_ok=True)

            print(f"Saving training loss to {training_losses_path}")
            with open(training_losses_path, 'w') as f:
                json.dump(ppo_losses, f, indent=4)

        if weights_path:
            parent_dir = os.path.dirname(weights_path)
            os.makedirs(parent_dir, exist_ok=True)

            print(f"Saving weights to {weights_path}")
            self.policy_network.save_weights(weights_path)
