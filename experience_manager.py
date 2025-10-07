import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class ExperienceManager:
    """
    Class that stores and manages experiences from multiple environments.

    :param batch_size: Total number of trajectories to consider the batch full.
    :param minibatch_size: Number of trajectories in each minibatch for training.
    """

    def __init__(self, batch_size, minibatch_size):
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.current_batch_size = 0

        self.experienceDict = {
            key: [] for key in ["states", "next_states", "actions", "action_probs", "rewards", "values", "terminated", "truncated"]
        }

    def clear(self):
        """
        Clears the experience dictionary and resets the current batch size.
        """
        self.experienceDict = {
            key: [] for key in ["states", "next_states", "actions", "action_probs", "rewards", "values", "terminated", "truncated"]
        }
        self.current_batch_size = 0

    def get(self, key, return_type="np"):
        """
        Retrieve and concatenate data from the experience buffer based on specified key.

        :param key: Type of data to retrieve (e.g., "states", "actions").
        :param return_type: Format to return the data in ("np" for numpy array, "torch" for torch tensor).

        :return: The requested data as a numpy array or a torch tensor.
        """
        data = self.experienceDict[key]
        if return_type == "np":
            return np.array(data)

        elif return_type == "torch":
            # Convert list of arrays to a single numpy array before converting to a tensor
            np_array = np.array(data)
            return torch.tensor(np_array, dtype=torch.float32)

    def is_full(self):
        """
        Checks if the current batch size has reached or exceeded the set batch size.

        :return: Boolean indicating whether the batch size limit has been reached.
        """
        return self.current_batch_size >= self.batch_size

    def appendTrajectory(self, state, next_state, action, action_prob, reward, value, terminated, truncated):
        """
        Append a trajectory (experience at a timestep) for each environment.

        :param altitudes: altitudes matrix of the local map [1, map_size, map_size]
        :param position_vectors: position vectors processed and concatenated [5,]
        :param action: Array of actions for each environment.
        :param action_prob: Array of action probabilities for each environment.
        :param reward: Array of rewards for each environment.
        :param value: Array of value estimates for each environment.
        :param terminated: Array of done signals (indicating end of episode) for each environment.
        """
        self.current_batch_size += 1

        self.experienceDict["states"].append(state)
        self.experienceDict["next_states"].append(next_state)
        self.experienceDict["actions"].append(action)
        self.experienceDict["action_probs"].append(action_prob)
        self.experienceDict["rewards"].append(reward)
        self.experienceDict["values"].append(value)
        self.experienceDict["terminated"].append(terminated)
        self.experienceDict["truncated"].append(truncated)

    def compute_advantage_and_returns(self, next_values, gamma, lambda_):
        """
        Calculates GAE using the explicitly provided next_values.
        """
        rewards = self.get("rewards", return_type="np")
        values = self.get("values", return_type="np")
        terminated = self.get("terminated", return_type="np")

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            v_next = next_values[t] * (1 - terminated[t])
            delta = rewards[t] + gamma * v_next - values[t]
            advantages[t] = gae = delta + gamma * lambda_ * gae * (1 - terminated[t])

        returns = advantages + values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def get_batches(self, policy_network, device, td_gamma=0.999, gae_lambda=0.95, shuffle=True):
        """
        Creates batches of data for training using the stored experiences.
        """
        next_states = self.get("next_states", return_type="torch").to(device)

        with torch.no_grad():
            _, next_values_tensor = policy_network(next_states)
        next_values = next_values_tensor.squeeze().cpu().numpy()
        advantages, returns = self.compute_advantage_and_returns(next_values, td_gamma, gae_lambda)

        states = self.get("states", return_type="torch").to(device)
        actions = self.get("actions", return_type="torch").to(device)
        action_probs = self.get("action_probs", return_type="torch").to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)

        dataset = TensorDataset(states, actions, action_probs, advantages, returns)
        return DataLoader(dataset, batch_size=self.minibatch_size, shuffle=shuffle)
