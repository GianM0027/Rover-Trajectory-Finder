# todo classe agent, che si occupa di:
#   - logica di apprendimento -> decidere come aggiornare i pesi della rete, con quanti dati, quali parametri etc...
#   - Gestire il Replay buffer, le trajectories, etc.
#   - interazione con l'ambiente -> scegliere che azione effettuare

import numpy as np

class Agent():
    def __init__(self, mars_environment, seed=None, policy_network=None, batch_size=None, minibatch_size=None):
        self.mars_environment = mars_environment
        self.seed = seed

        # Training Parameters
        self.policy_network = policy_network
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size

    def run_simulation(self, max_episodes=None, use_policy_network=False, verbose=False):
        n_episode = 0

        while True:
            print(f"Episode #{n_episode+1}")
            observation, _ = self.mars_environment.reset(seed=self.seed)
            terminated = False

            while not terminated:
                if use_policy_network:
                    processed_observation = self.preprocess_observation(observation)
                    action = self.policy_network(processed_observation)
                else:
                    action = np.random.randint(8)

                observation, reward, terminated, truncated, info = self.mars_environment.step(action, verbose=verbose)

            n_episode += 1
            if max_episodes is not None and n_episode >= max_episodes:
                break


    def preprocess_observation(self, observation):
        #todo: presa una singola observation, la trasforma nel formato richiesto dalla CNN
        return observation

    def train(self, parameters):
        # todo: organizzare tutti i dati nel formato richiesto e poi passarli alla funzione di training della policy network
        self.policy_network.fit()