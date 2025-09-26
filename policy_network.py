from typing import Dict

import torch
import torch.nn as nn


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

    def forward(self, x):
        # estrazione feature

        x = self.normalize(x)

        for layer in self.backbone_layers:
            x = layer(x)
        x = torch.flatten(x, 1)

        # Build heads
        if not self.fc_built:
            self.head_action = self._build_fc_head(self.head_action_config, x.size(1))
            self.head_value = self._build_fc_head(self.head_value_config, x.size(1))
            self.fc_built = True

        out_action = self.head_action(x)
        out_action = self.softmax(out_action)

        out_value = self.head_value(x)

        return out_action, out_value

    def normalize(self, x):
        """
            Normalize a 3D tensor by applying two transformations to its channels.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (C, H, W), where:
                - C = number of channels (at least 2 are required).
                - H, W = spatial dimensions.

            Operations
            ----------
            1. Channel 0 (altitude):
               - Computes the minimum and maximum values while ignoring NaNs.
               - Applies min-max normalization to scale values into the [0, 1] range.
               - Preserves NaNs in their original positions, or replaces them with a
                 user-defined value (currently `float()` -> 0.0,
               - Result: altitude values scaled between 0 and 1.

            2. Channel 1 (position_counter):
               - Applies a logarithmic transformation `log(1 + x)`, which compresses
                 large values while preserving small ones.
               - Adding 1 prevents issues with log(0).

            Returns
            -------
            x : torch.Tensor
                The transformed tensor, with the same shape as the input (C, H, W).

            Notes
            -----
            - NaNs in channel 0 are either preserved or replaced with a fill value.
            - If min_val == max_val in channel 0, normalization will result in NaNs/Infs.
              This case should be handled explicitly if it can occur.
            - The log transformation in channel 1 assumes non-negative values.
              Negative values will result in NaNs.
            """

        altitude = x[0, :, :]
        mask = torch.isnan(altitude)
        min_val = torch.nanmin(altitude)
        max_val = torch.nanmax(altitude)
        normalized_altitude = (altitude - min_val) / (max_val - min_val)
        normalized_altitude[mask] = float()  # set your value #todo
        x[0, :, :] = normalized_altitude

        position_counter = x[1, :, :]
        x[1, :, :] = torch.log(1 + position_counter)

        return x

    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, device: str = "cuda"):
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
        self.eval()
