# todo classe agent, che si occupa di:
#   - logica di apprendimento -> decidere come aggiornare i pesi della rete, con quanti dati, quali parametri etc...
#   - Gestire il Replay buffer, le trajectories, etc.
#   - interazione con l'ambiente -> scegliere che azione effettuare
import random
from custom_environment import GridMarsEnv
from policy_network import PolicyNetwork
import numpy as np
from experience_manager import ExperienceManager


class Agent:
    def __init__(self, mars_environment: GridMarsEnv, policy_network: PolicyNetwork = None, seed=None):
        self.mars_environment = mars_environment
        self.seed = seed
        self.policy_network = policy_network

    def run_simulation(self, max_episodes=None, use_policy_network=False, verbose=False):
        n_episode = 0

        while True:
            print(f"Episode #{n_episode + 1}")
            observation, _ = self.mars_environment.reset(seed=self.seed)
            terminated = False

            while not terminated:
                if use_policy_network:
                    processed_observation = self.preprocess_observation(observation)
                    action, _ = self.policy_network(processed_observation)
                else:
                    action = np.random.randint(8)

                observation, reward, terminated, truncated, info = self.mars_environment.step(action, verbose=verbose)

            n_episode += 1
            if max_episodes is not None and n_episode >= max_episodes:
                break

    @classmethod
    def preprocess_observation(cls, observation):
        agent_position = observation["agent"]
        target_position = observation["target"]

        local_map = observation["local_map"]
        local_map_mask = observation["local_map_mask"]
        map_shape = local_map.shape

        channel_zero = np.array([
            [
                local_map[y, x] if local_map_mask[y, x] else np.nan
                for x in range(map_shape[1])
            ] for y in range(map_shape[0])
        ])

        channel_one = observation["visited_locations"]

        channel_two = np.zeros(local_map.shape)
        channel_two[agent_position[0], agent_position[1]] = 1
        channel_two[target_position[0], target_position[1]] = -1

        return np.array([channel_zero, channel_one, channel_two])

    def train(self, training_episodes=1000, batch_size=256, minibatch_size=8, loss=None, device='cuda'):
        experience_manager = ExperienceManager(batch_size=batch_size, minibatch_size=minibatch_size)
        episode_counter = 0

        terminated = True
        observation = None

        while episode_counter < training_episodes:
            while not experience_manager.is_full():
                if terminated:
                    observation, _ = self.mars_environment.reset(seed=self.seed)
                processed_observation = self.preprocess_observation(observation)
                action_probs, value = self.policy_network(processed_observation)
                action = np.argmax(action_probs)

                observation, reward, terminated, truncated, info = self.mars_environment.step(action)
                experience_manager.appendTrajectory(state=self.preprocess_observation(observation),
                                                    action=action,
                                                    action_prob=action_probs,
                                                    reward=reward,
                                                    value=value,
                                                    terminated=terminated)

            # Train
            dataloader = experience_manager.get_batches() # todo -> parameters
            for minibatch in dataloader:
                self.policy_network.fit(minibatch=minibatch, loss=loss)

            episode_counter += 1
