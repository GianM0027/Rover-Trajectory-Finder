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
            key: [] for key in ["states", "actions", "action_probs", "rewards", "values", "terminated", "truncated"]
        }

    def clear(self):
        """
        Clears the experience dictionary and resets the current batch size.
        """
        self.experienceDict = {
            key: [] for key in ["states", "actions", "action_probs", "rewards", "values", "terminated", "truncated"]
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

    def appendTrajectory(self, state, action, action_prob, reward, value, terminated, truncated):
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
        self.experienceDict["actions"].append(action)
        self.experienceDict["action_probs"].append(action_prob)
        self.experienceDict["rewards"].append(reward)
        self.experienceDict["values"].append(value)
        self.experienceDict["terminated"].append(terminated)
        self.experienceDict["truncated"].append(truncated)


    def compute_advantage_and_returns(self, next_value, gamma, lambda_):
        """
        Calculate the generalized advantage estimation (GAE) and returns for each environment.

        @param next_values: The value estimates for the next state (beyond the current batch of data).
        @param gamma: Discount factor for rewards.
        @param lambda_: Smoothing parameter for GAE.
        @return: Two torch tensors containing the advantages and returns respectively.
        """
        
        rewards = self.get("rewards", return_type="np")
        values = self.get("values", return_type="np")
        terminated = self.get("terminated", return_type="np")
        truncated = self.get("truncated", return_type="np")
        dones = np.logical_or(terminated, truncated)
        gae = 0

        advantages = np.zeros_like(rewards, dtype=np.float64)
        returns = np.zeros_like(rewards, dtype=np.float64)

        # Calculate returns by iterating backwards through the rewards
        for i in reversed(range(len(rewards))):

            # if this is the last position of the batch retrieve the next value
            if i != len(rewards) - 1:
                next_value = values[i + 1]

            # update next_value and gae according to the fact that this may be a terminal state
            next_value = next_value * (1 - dones[i])
            gae = gae * (1 - dones[i])

            # compute advantages
            delta = rewards[i] + (gamma * next_value) - values[i]
            advantages[i] = gae = delta + gamma * lambda_ * gae

        # normalize the advantage
        advantages = advantages - np.mean(advantages) / (np.std(advantages) + 1e-10)

        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def get_batches(self, next_value, device, td_gamma=0.999, gae_lambda=0.95, shuffle=True):
        """
        Creates batches of data for training using the stored experiences.
        """
        advantages, returns = self.compute_advantage_and_returns(next_value, td_gamma, gae_lambda)

        states = self.get("states", return_type="torch").to(device)
        actions = self.get("actions", return_type="torch").to(device)
        action_probs = self.get("action_probs", return_type="torch").to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)

        dataset = TensorDataset(states, actions, action_probs, advantages, returns)
        return DataLoader(dataset, batch_size=self.minibatch_size, shuffle=shuffle)
