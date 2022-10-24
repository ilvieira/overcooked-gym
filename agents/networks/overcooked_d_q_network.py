import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from torch.nn.functional import log_softmax

"""This is similar to the architecture of the atari networks from Mnih 2015, but it takes only one channel, as there is 
no need to account for velocity in the overcooked environment."""


class OvercookedDQNetwork(nn.Module):

    def __init__(self, number_of_actions, final_transformation=(lambda x: x)):
        super().__init__()

        self.final_transformation = final_transformation

        # ============================================================================================
        #                                    Convolutional Layers
        # ============================================================================================

        # Initialization:
        # stdv <- 1/(math.sqrt(kW*kH*nInputPlane))
        # weights <- uniform(-stdv, stdv)
        # bias <- uniform(-stdv, stdv)

        # size of the input: (84, 84, 1)
        kw = 8
        kh = kw
        n_input_plane = 1  # (hist_len*ncols)
        n_output = 32
        self.conv1 = nn.Conv2d(in_channels=n_input_plane, out_channels=n_output, kernel_size=kw, stride=4)
        stdv = 1/(math.sqrt(kw * kh * n_input_plane))
        torch.nn.init.uniform_(self.conv1.weight, a=-stdv, b=stdv)
        torch.nn.init.uniform_(self.conv1.bias, a=-stdv, b=stdv)

        # size of the input: (20, 20, 32)
        kw = 4
        kh = kw
        n_input_plane = n_output
        n_output = 64
        self.conv2 = nn.Conv2d(in_channels=n_input_plane, out_channels=n_output, kernel_size=kw, stride=2)
        stdv = 1 / (math.sqrt(kw * kh * n_input_plane))
        torch.nn.init.uniform_(self.conv2.weight, a=-stdv, b=stdv)
        torch.nn.init.uniform_(self.conv2.bias, a=-stdv, b=stdv)

        # size of the input: (9, 9, 64)
        kw = 3
        kh = kw
        n_input_plane = n_output
        n_output = 64
        self.conv3 = nn.Conv2d(in_channels=n_input_plane, out_channels=n_output, kernel_size=kw, stride=1)
        stdv = 1 / (math.sqrt(kw * kh * n_input_plane))
        torch.nn.init.uniform_(self.conv3.weight, a=-stdv, b=stdv)
        torch.nn.init.uniform_(self.conv3.bias, a=-stdv, b=stdv)

        # ============================================================================================
        #                                  Fully Connected Layers
        # ============================================================================================

        # Initialization:
        # input_size <- weights.size[1]
        # stdv <- 1/(math.sqrt(input_size))
        # weights <- uniform(-stdv, stdv)
        # bias <- uniform(-stdv, stdv)

        # size of the input: (7, 7, 64)
        input_size = 7 * 7 * 64
        n_output = 512
        self.fc1 = nn.Linear(input_size, n_output)
        stdv = 1 / (math.sqrt(input_size))
        torch.nn.init.uniform_(self.fc1.weight, a=-stdv, b=stdv)
        torch.nn.init.uniform_(self.fc1.bias, a=-stdv, b=stdv)

        input_size = n_output
        self.fc2 = nn.Linear(input_size, number_of_actions)
        stdv = 1 / (math.sqrt(input_size))
        torch.nn.init.uniform_(self.fc2.weight, a=-stdv, b=stdv)
        torch.nn.init.uniform_(self.fc2.bias, a=-stdv, b=stdv)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 7 * 7 * 64)))
        return self.final_transformation(self.fc2(x))