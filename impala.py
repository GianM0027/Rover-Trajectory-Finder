import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        if m.out_features == 15:
            # Initialize weights 100 times smaller than usual
            std = init.calculate_gain('linear') / 100
            init.normal_(m.weight, mean=0.0, std=std)
            init.constant_(m.bias, 0)
        else:
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)
        
        self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.resblock1 = ResidualBlock(out_channels)
        self.resblock2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        # x = self.pool(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        return x


class ImpalaModel(nn.Module):
    def __init__(self, action_space=8, input_channels=8):
        super(ImpalaModel, self).__init__()

        self.conv_block1 = ConvolutionalBlock(input_channels, 16)
        self.conv_block2 = ConvolutionalBlock(16, 32)
        self.conv_block3 = ConvolutionalBlock(32, 32)

        self.feed_forward = nn.LazyLinear(64)

        self.policy_head = nn.Linear(64, action_space)
        self.value_head = nn.Linear(64, 1)
        self.softmax_activation = nn.Softmax(dim=1)

        # self.apply(initialize_weights)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = F.relu(x)
        x = x.view(x.size(0), -1)

        x = self.feed_forward(x)
        x = F.relu(x)

        policy = self.softmax_activation(self.policy_head(x))
        value = self.value_head(x)

        return policy, value

    def save_weights(self, cnn_weights_path, full_weights_path):
        full_state_dict = self.state_dict()
        torch.save(full_state_dict, full_weights_path)
        feature_extractor_state_dict = {
            key: value for key, value in full_state_dict.items()
            if key.startswith(('conv_block1', 'conv_block2', 'conv_block3'))
        }
        
        if feature_extractor_state_dict:
            torch.save(feature_extractor_state_dict, cnn_weights_path)

    def load_feature_extractor_weights(self, path: str, device: str = "cuda"):
        """Loads weights into the feature extractor, ignoring the rest of the network."""
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict, strict=False)
        self.eval()
    
    def load_full_weights(self, path: str, device: str = "cuda"):
        """Loads weights into the full the network."""
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
        self.eval()
