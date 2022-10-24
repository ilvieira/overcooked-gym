import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from torch.nn.functional import log_softmax


class OvercookedMultiDistillationNetwork(nn.Module):

    def __init__(self, number_of_actions, final_transformation=lambda vals: log_softmax(vals, dim=1)):
        """number_of_actions is a list or a tuple where component i corresponds to the number of possible actions for
        the agent for task i"""
        super().__init__()

        self.final_transformation = final_transformation

        # ============================================================================================
        #                                    Convolutional Layers
        # ============================================================================================

        # Initialization:
        # stdv <- 1/(math.sqrt(kW*kH*nInputPlane))
        # weights <- uniform(-stdv, stdv)
        # bias <- uniform(-stdv, stdv)

        # size of the input: (84, 84, 4)
        kw = 8
        kh = kw
        n_input_plane = 1
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
        self.output_layers = []
        for i in range(len(number_of_actions)):
            actions = number_of_actions[i]
            fc2_i = nn.Linear(input_size, actions)
            stdv = 1 / (math.sqrt(input_size))
            torch.nn.init.uniform_(fc2_i.weight, a=-stdv, b=stdv)
            torch.nn.init.uniform_(fc2_i.bias, a=-stdv, b=stdv)
            # add each of these layers to the modules of this net
            self.add_module("fc2_"+str(i), fc2_i)
            self.output_layers.append(fc2_i)
        self.current_task = 0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 7 * 7 * 64)))
        return self.final_transformation(self.output_layers[self.current_task](x))

    def choose_task(self, task_id):
        self.current_task = task_id

    def forward_for_all_tasks(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 7 * 7 * 64)))
        return [self.final_transformation(self.output_layers[i](x)) for i in range(len(self.output_layers))]