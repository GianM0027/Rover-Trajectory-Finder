import torch


class policy_network(torch.nn.Module):
    def __init__(self, n_actions=8):
        super(policy_network, self).__init__()

        # todo: implementare il modello che prende in input una batch di observations (come organizzato in agent.py)
        #       e restituisce una di X possibili azioni (+ value se necessario)

    def forward(self, x):
        pass

    def train(self):
        pass