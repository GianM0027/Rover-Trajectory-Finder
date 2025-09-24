import torch

# todo: implementare la rete che prende in input una osservazione di qualsiasi grandezza e restituisce un di X possibili azioni

class policy_network(torch.nn.Module):
    def __init__(self):
        super(policy_network, self).__init__()
