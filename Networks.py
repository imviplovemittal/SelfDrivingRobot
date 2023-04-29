import torch
import torch.nn as nn
import numpy as np


class Action_Conditioned_FF(nn.Module):
    def __init__(self, input_size=6, output_size=1):
        # STUDENTS: __init__() must initiatize nn.Module and define your network's
        # custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.layer_1 = nn.Linear(input_size, 12)
        self.layer_2 = nn.Linear(12, 96)
        self.layer_3 = nn.Linear(96, 192)
        self.layer_out = nn.Linear(192, output_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        # STUDENTS: forward() must complete a single forward pass through your network
        # and return the output which should be a tensor
        hidden = self.relu(self.layer_1(input))
        hidden = self.relu(self.layer_2(hidden))
        hidden = self.relu(self.layer_3(hidden))
        output = self.layer_out(hidden)
        return output

    def evaluate(self, model, test_loader, loss_function):
        # STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
        # mind that we do not need to keep track of any gradients while evaluating the
        # model. loss_function will be a PyTorch loss function which takes as argument the model's
        # output and the desired output.

        loss = 0
        for idx, sample in enumerate(test_loader):
            output = self.forward(torch.tensor(sample['input'], dtype=torch.float32))
            loss += loss_function(output, torch.tensor(sample['label'], dtype=torch.float32))
        return np.float(loss / len(test_loader))


def main():
    model = Action_Conditioned_FF()


if __name__ == '__main__':
    main()
