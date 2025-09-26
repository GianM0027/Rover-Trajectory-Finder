import torch


class policy_network(torch.nn.Module):
    def __init__(self, n_actions=8):
        super(policy_network, self).__init__()

        # todo: implementare la CNN con le seguenti restrizioni:
        #       .
        #       INPUT:
        #       - Deve prendere in input una matrice con shape (3, n, n).
        #       .
        #       OUTPUT:
        #       - Una delle 8 possibili azioni (0-7)
        #       - Un float restituito come logit. Questo rappresenta il "vantaggio" di usare l'azione restituita dall'altra testa.

    def preprocess(self, x):
        # todo:
        #   - La normalizzazione delle altimetrie (canale 0) va fatta con minmax (ignorando i np.NaN)
        #   - La normalizzazione del counter (canale 1) va fatto con log(1 + x)
        #   - La posizione del rover e del target (canale 2) non va normalizzata
        pass

    def forward(self, args):
        pass

    def fit(self, minibatch, loss, epochs_per_minibatch=1):
        # todo: dati i parametri della funzione, esegui l'aggiornamento dei pesi della rete usando un unico minibatch,
        #      poi resituisci il valore delle loss per ogni minibatch

        loss_values = []
        return loss_values

    def save_weights(self, path):
        pass

    def load_weights(self, path):
        # todo: questo va fatto "inplace" -> questo metodo non restituisce nulla.
        pass