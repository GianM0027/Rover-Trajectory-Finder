import random
import torch
from torch import nn, optim
from tqdm import tqdm
from custom_environment import GridMarsEnv
from policy_network import PolicyNetwork
import numpy as np
from experience_manager import ExperienceManager


class Agent:
    def __init__(self, mars_environment: GridMarsEnv, policy_network: PolicyNetwork = None, seed=None):
        self.mars_environment = mars_environment
        self.seed = seed
        self.policy_network = policy_network

    def run_simulation(self, max_episodes=None, use_policy_network=False, verbose=False, device="cuda"):
        if use_policy_network:
            self.policy_network.eval()
            self.policy_network.to(device)

        n_episode = 0
        while True:
            print(f"Episode #{n_episode + 1}")
            observation, _ = self.mars_environment.reset(seed=self.seed)
            terminated = False

            while not terminated:
                if use_policy_network:
                    processed_observation = self.preprocess_observation(observation, device)
                    action_probs, _ = self.policy_network(processed_observation)
                    action = torch.argmax(action_probs).item()
                else:
                    action = np.random.randint(8)

                observation, reward, terminated, truncated, info = self.mars_environment.step(action, verbose=verbose)

            n_episode += 1
            if max_episodes is not None and n_episode >= max_episodes:
                break

    @classmethod
    def preprocess_observation(cls, observation, device="cuda", return_type="torch"):
        agent_position = observation["agent"]
        target_position = observation["target"]

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

        # Channel 2: agent/target positions
        channel_two = np.zeros(map_shape, dtype=np.float32)
        if agent_position[0] == target_position[0] and agent_position[1] == target_position[1]:
            channel_two[agent_position[0], agent_position[1]] = 1.0
            channel_two[target_position[0], target_position[1]] = -1.0
        else:
            # todo: capire che fare se agente e target sono nella stessa posizione
            channel_two[agent_position[0], agent_position[1]] = 2.0

        # Stack channels
        x = np.stack([channel_zero, channel_one, channel_two], axis=0)  # (C, H, W)

        # Normalize channel 0 (altitude)
        altitude = x[0]
        mask = np.isnan(altitude)
        min_val = np.nanmin(altitude)
        max_val = np.nanmax(altitude)
        if min_val == max_val:
            x[0] = np.zeros_like(altitude)
        else:
            x[0] = (altitude - min_val) / (max_val - min_val)
        x[0][mask] = -1.0  # sentinel for NaN

        # Normalize channel 1 (visited locations)
        x[1] = np.log1p(x[1])

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
              learning_rate=1e-4,
              weights_path=None,
              step_verbose=False):

        mse_loss = nn.MSELoss()
        optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.policy_network.train()
        self.policy_network.to(device)

        experience_manager = ExperienceManager(batch_size=batch_size, minibatch_size=minibatch_size)

        # Wrap the outer loop with tqdm
        for _ in tqdm(range(training_episodes), desc="Training Episodes"):
            observation, _ = self.mars_environment.reset(seed=self.seed)
            terminated = False

            while not terminated:
                with torch.no_grad():
                    processed_observation = self.preprocess_observation(observation, device)
                    action_probs, value = self.policy_network(processed_observation)
                action = torch.distributions.Categorical(action_probs).sample().item()

                observation, reward, terminated, truncated, info = self.mars_environment.step(action, verbose=step_verbose)
                processed_observation = self.preprocess_observation(observation, device)
                prob_of_taken_action = action_probs.squeeze()[action]

                experience_manager.appendTrajectory(
                    state=processed_observation.squeeze().cpu().numpy(),
                    action=action,
                    action_prob=prob_of_taken_action.cpu().numpy(),
                    reward=reward,
                    value=value.squeeze().cpu().numpy(),
                    terminated=terminated
                )

                if experience_manager.is_full():
                    print("Updating weights")
                    with torch.no_grad():
                        processed_observation = self.preprocess_observation(observation, device)
                        _, next_value = self.policy_network(processed_observation)

                    dataloader = experience_manager.get_batches(device=device, next_value=next_value.cpu().numpy())

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
                            optimizer.zero_grad()
                            total_loss.backward()
                            optimizer.step()

                    experience_manager.clear()

        if weights_path:
            print(f"Saving weights to {weights_path}")
            self.policy_network.save_weights(weights_path)
