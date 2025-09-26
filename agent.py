# todo classe agent, che si occupa di:
#   - logica di apprendimento -> decidere come aggiornare i pesi della rete, con quanti dati, quali parametri etc...
#   - Gestire il Replay buffer, le trajectories, etc.
#   - interazione con l'ambiente -> scegliere che azione effettuare


class Agent():
    def __init__(self, policy_network, batch_size, minibatch_size):
        # Training Parameters
        self.policy_network = policy_network
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size


    def train(self, parameters):
        # todo: organizzare tutti i dati nel formato richiesto e poi passarli alla funzione di training della policy network
        self.policy_network.fit()