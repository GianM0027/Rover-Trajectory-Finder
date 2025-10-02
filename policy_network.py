import torch
import torch.nn as nn
from typing import Dict


# todo: capire se la network può ricevere uno stack di osservazioni e comprenderne la sequenzialità, oppure se
#       bisogna per forza aggiungere una LSTM. Guardare come viene fatto in https://arxiv.org/pdf/1802.01561

# todo: ingrandire la network. Aggiungere skip connection. Attenzione a pooling che potrebbe far perdere info importanti
#       soprattutto riguardo alle altitudini che circondano l'agente, la sua posizione e quella del target.
#       Capire se le posizioni di agente e target è meglio passarle come vettore (y,x) invece che come canale della matrice.

class PolicyNetwork(torch.nn.Module):
    def __init__(self, config: Dict):
        super(PolicyNetwork, self).__init__()
        """
            config Example:
               {
                   "input_channels": 3,
                   "backbone": [
                       {"type": "conv", "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},
                       {"type": "pool", "mode": "max", "kernel_size": 2},
                       {"type": "conv", "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},
                       {"type": "pool", "mode": "max", "kernel_size": 2}
                   ],
                   "head_action": [  
                       {"type": "fc", "out_features": 128, "activation": "relu"},
                       {"type": "fc", "out_features": 10}
                   ],
                   "head_value": [  
                       {"type": "fc", "out_features": 64, "activation": "relu"},
                       {"type": "fc", "out_features": 1}
                   ]
               }
        """
        super(PolicyNetwork, self).__init__()

        self.config = config

        self.backbone_layers = nn.ModuleList()
        in_channels = self.config["input_channels"]
        for layer_cfg in self.config["backbone"]:
            if layer_cfg["type"] == "conv":
                conv = nn.Conv2d(in_channels=in_channels,
                                 out_channels=layer_cfg["out_channels"],
                                 kernel_size=layer_cfg.get("kernel_size", 3),
                                 stride=layer_cfg.get("stride", 1),
                                 padding=layer_cfg.get("padding", 0)
                                 )
                self.backbone_layers.append(conv)
                if layer_cfg.get("activation") == "relu":
                    self.backbone_layers.append(nn.ReLU())
                if layer_cfg.get("activation") == "leakyRelu":
                    negative_slope = layer_cfg.get("negative_slope", 0.2)
                    self.backbone_layers.append(nn.LeakyReLU(negative_slope))
                in_channels = layer_cfg["out_channels"]

            elif layer_cfg["type"] == "pool":
                if layer_cfg["mode"] == "max":
                    self.backbone_layers.append(nn.MaxPool2d(kernel_size=layer_cfg.get("kernel_size", 3),
                                                             stride=layer_cfg.get("stride", 1),
                                                             padding=layer_cfg.get("padding", 0)))
                elif layer_cfg["mode"] == "avg":
                    self.backbone_layers.append(nn.AvgPool2d(kernel_size=layer_cfg.get("kernel_size", 3),
                                                             stride=layer_cfg.get("stride", 1),
                                                             padding=layer_cfg.get("padding", 0)))
                else:
                    raise NotImplementedError

        # placeholder for head
        self.head_action_config = self.config["head_action"]
        self.head_value_config = self.config["head_value"]
        self.head_action = None
        self.head_value = None
        self.fc_built = False

        self.softmax = nn.Softmax(dim=1)

    def _build_fc_head(self, head_config, in_features):
        layers = []
        for layer_cfg in head_config:
            fc = nn.Linear(in_features, layer_cfg["out_features"])
            layers.append(fc)
            if layer_cfg.get("activation") == "relu":
                layers.append(nn.ReLU())
            in_features = layer_cfg["out_features"]
        return nn.Sequential(*layers)

    def forward(self, matrice, vettore):

        # todo: matrice va nella CNN, ha dimensione [batch_size, map_size, map_size]
        # todo vettore va a una MLP, ha dimensione [5,]

        for layer in self.backbone_layers:
            x = layer(x)
        x = torch.flatten(x, 1)

        # Build heads
        if not self.fc_built:
            device = x.device
            self.head_action = self._build_fc_head(self.head_action_config, x.size(1)).to(device)
            self.head_value = self._build_fc_head(self.head_value_config, x.size(1)).to(device)
            self.fc_built = True

        out_action = self.head_action(x)
        out_action = self.softmax(out_action)

        out_value = self.head_value(x)

        return out_action, out_value

    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, device: str = "cuda"):
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
        self.eval()
