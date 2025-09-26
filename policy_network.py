import torch


class policy_network(torch.nn.Module):
    def __init__(self, n_actions=8):
        super(policy_network, self).__init__()

        # todo: implementare la CNN con le seguenti restrizioni:
        #       .
        #       INPUT:
        #       - Deve prendere in input due matrici quadrate con shape (n, n). Anche fuse assieme in una matrice (n, n, 2)
        #       - Due array con shape (2,). Queste sono le posizioni del rover e del target.
        #       .
        #       OUTPUT:
        #       - Una delle 8 possibili azioni (0-7)
        #       - Un float restituito come logit. Questo rappresenta il "vantaggio" di usare l'azione restituita dall'altra testa.


    def forward(self, args):
        pass

    def fit(self, batch, minibatch_size, loss, epochs_per_minibatch=1):
        #todo: dati i parametri della funzione, esegui l'aggiornamento dei pesi della rete poi resituisci il valore delle loss per ogni minibatch
        loss_values = []
        return loss_values