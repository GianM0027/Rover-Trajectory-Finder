import random
import torch
from torch import nn, optim
from tqdm import tqdm
from custom_environment import GridMarsEnv
from policy_network import PolicyNetwork
import numpy as np
from experience_manager import ExperienceManager


class Agent:
    def __init__(self, mars_environment: GridMarsEnv, policy_network: PolicyNetwork = None, max_number_of_steps=None, seed=None):
        self.seed = seed
        self.policy_network = policy_network
        self.mars_environment = mars_environment
        self.max_number_of_steps = max_number_of_steps

    def run_simulation(self, max_episodes=None, use_policy_network=False, verbose=False, device="cuda", sample_action=False):
        if use_policy_network:
            self.policy_network.eval()
            self.policy_network.to(device)

        # todo: aggiusta in base a nuovo modello

        n_episode = 0
        while True:
            print(f"Episode #{n_episode + 1}")
            observation, _ = self.mars_environment.reset(seed=self.seed)
            terminated = False
            truncated = False

            while not terminated and not truncated:
                if use_policy_network:
                    processed_observation = self.preprocess_model_input(observation, device)
                    action_probs, _ = self.policy_network(processed_observation)
                    if sample_action:
                        action = torch.distributions.Categorical(action_probs).sample().item()
                    else:
                        action = torch.argmax(action_probs).item()
                else:
                    action = np.random.randint(8)

                observation, reward, terminated, truncated, info = self.mars_environment.step(action, verbose=verbose)

            n_episode += 1
            if max_episodes is not None and n_episode >= max_episodes:
                break

    def preprocess_model_input(self, observation, device="cuda", eps=1e-4):
        padding_number = 0
        agent_previous_position = observation["agent_previous_pos"]
        agent_position = observation["agent_pos"]
        target_position = observation["target_pos"]

        local_map = observation["local_map"]
        local_map_mask = observation["local_map_mask"]
        visited_locations = observation["visited_locations"]
        map_shape = local_map.shape

        # Altitude Matrix with MinMax normalization
        altitudes = np.array([
            [local_map[y, x] if local_map_mask[y, x] else np.nan
             for x in range(map_shape[1])]
            for y in range(map_shape[0])
        ], dtype=np.float32)

        mask = np.isnan(altitudes)
        min_val = np.nanmin(altitudes)
        max_val = np.nanmax(altitudes)
        if min_val == max_val:
            altitudes = np.zeros_like(altitudes)
        else:
            altitudes = ((altitudes - min_val) / (max_val - min_val)) + eps
        altitudes[mask] = padding_number  # sentinel for NaN

        current_goal_vector = (target_position - agent_position) / map_shape[0]
        previous_goal_vector = target_position - agent_previous_position / map_shape[0]
        n_visits = visited_locations[agent_position[0], agent_position[1]] / self.max_number_of_steps

        # Convert to torch tensor and add batch dimension
        altitudes = torch.tensor(altitudes, dtype=torch.float32).unsqueeze(0).to(device)

        position_vector = [v[i] for i in range(2) for v in [current_goal_vector, previous_goal_vector]] + [n_visits]
        position_vector = torch.tensor(position_vector, dtype=torch.float32).unsqueeze(0).to(device)

        return altitudes, position_vector

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

        # todo: salvare su file tutte le statistiche del training:
        #       - loss
        #       - path trovati da RL vs path trovati da Dijkstra
        #       - Somma dei reward per ogni episodio
        #       - Lunghezza degli episodi in termmini di step

        # todo: creare la logica per cui le trajectories possono essere salvate in minibatch con osservazioni multiple,
        #       già pronte per LSTM.

        mse_loss = nn.MSELoss()
        optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.policy_network.train()
        self.policy_network.to(device)

        experience_manager = ExperienceManager(batch_size=batch_size, minibatch_size=minibatch_size)

        # Wrap the outer loop with tqdm
        for _ in tqdm(range(training_episodes), desc="Training Episodes"):
            observation, _ = self.mars_environment.reset(seed=self.seed)
            terminated = False
            truncated = False

            while not terminated and not truncated:
                with torch.no_grad():
                    processed_altitudes, processed_positions = self.preprocess_model_input(observation, device)
                    action_probs, value = self.policy_network(processed_altitudes, processed_positions)
                action = torch.distributions.Categorical(action_probs).sample().item()

                observation, reward, terminated, truncated, info = self.mars_environment.step(action, verbose=step_verbose)
                processed_altitudes, processed_positions = self.preprocess_model_input(observation, device)
                prob_of_taken_action = action_probs.squeeze()[action]

                experience_manager.appendTrajectory(
                    altitudes=processed_altitudes.cpu().numpy(),
                    position_vectors=processed_positions.squeeze().cpu().numpy(),
                    action=action,
                    action_prob=prob_of_taken_action.cpu().numpy(),
                    reward=reward,
                    value=value.squeeze().cpu().numpy(),
                    terminated=terminated
                )

                if experience_manager.is_full():
                    print("Updating weights")
                    with torch.no_grad():
                        processed_altitudes, processed_positions = self.preprocess_model_input(observation, device)
                        _, next_value = self.policy_network(processed_altitudes, processed_positions)

                    dataloader = experience_manager.get_batches(device=device, next_value=next_value.cpu().numpy())

                    for _ in range(epochs):
                        for altitude, position_vector, actions, old_action_probs, advantages, returns in dataloader:
                            current_action_dist, values = self.policy_network(altitude, position_vector)
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
