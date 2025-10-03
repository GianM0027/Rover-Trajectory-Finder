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
            key: [] for key in ["altitudes", "position_vectors", "actions", "action_probs", "rewards", "values", "terminated", "truncated"]
        }

    def clear(self):
        """
        Clears the experience dictionary and resets the current batch size.
        """
        self.experienceDict = {
            key: [] for key in ["altitudes", "position_vectors", "actions", "action_probs", "rewards", "values", "terminated", "truncated"]
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

    def appendTrajectory(self, altitudes, position_vectors, action, action_prob, reward, value, terminated, truncated):
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

        self.experienceDict["altitudes"].append(altitudes)
        self.experienceDict["position_vectors"].append(position_vectors)
        self.experienceDict["actions"].append(action)
        self.experienceDict["action_probs"].append(action_prob)
        self.experienceDict["rewards"].append(reward)
        self.experienceDict["values"].append(value)
        self.experienceDict["terminated"].append(terminated)
        self.experienceDict["truncated"].append(truncated)

    def compute_advantage_and_returns(self, next_value, gamma, lambda_):
        """
        Calcola la Generalized Advantage Estimation (GAE) e i returns.
        Gestisce correttamente sia gli episodi terminati che quelli troncati.
        """
        rewards = self.get("rewards", return_type="np")
        values = self.get("values", return_type="np")
        terminated = self.get("terminated", return_type="np")
        truncated = self.get("truncated", return_type="np")
        dones = np.logical_or(terminated, truncated)

        env_advantages = np.zeros_like(rewards, dtype=np.float64)
        gae = 0

        for i in reversed(range(len(rewards))):
            current_next_value = values[i + 1] if i != len(rewards) - 1 else next_value
            bootstrap_value = current_next_value * (1 - terminated[i])
            gae = gae * (1 - dones[i])
            delta = rewards[i] + (gamma * bootstrap_value) - values[i]
            env_advantages[i] = gae = delta + (gamma * lambda_ * gae)

        env_advantages = (env_advantages - np.mean(env_advantages)) / (np.std(env_advantages) + 1e-10)
        returns = env_advantages + values

        return torch.tensor(env_advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def get_batches(self, device, next_value, td_gamma=0.999, gae_lambda=0.95, shuffle=True):
        """
        Create batches of data for training using the stored experiences.

        :param device: The device (CPU or GPU) to transfer the tensors to.
        :param next_value: The value estimates for the next state.
        :param td_gamma: discount factor for rewards.
        :param gae_lambda: Smoothing parameter for GAE.
        :param shuffle: Whether to shuffle the batches or not.

        :return: A DataLoader containing the batches of experiences for training.
        """
        altitudes = self.get("altitudes", return_type="torch").to(device)
        position_vectors = self.get("position_vectors", return_type="torch").to(device)
        actions = self.get("actions", return_type="torch").to(device)
        action_probs = self.get("action_probs", return_type="torch").to(device)
        advantages, returns = self.compute_advantage_and_returns(next_value, td_gamma, gae_lambda)
        advantages = advantages.to(device)
        returns = returns.to(device)

        dataset = TensorDataset(altitudes, position_vectors, actions, action_probs, advantages, returns)
        return DataLoader(dataset, batch_size=self.minibatch_size, shuffle=shuffle)
