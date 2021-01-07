"""
    Convert PC model to smartphone model.
    Put the generated checkpoints/model.pt and data/ref_RED.csv under ContInf/app/src/main/assets.
"""

import torch
import numpy as np
import torch.nn as nn

## this reimplements the _get_reconstruction_error method
## the get_error method is a reimplementation of the '_get_reconstruction_error' method from
## the provided LSTM repo https://github.com/zechenghe/LSTMAD/blob/inference/detector.py#L67
## it only returns the RE, not the pred, but the pred isn't used for inference

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = provided_model

    def get_error(self, x):
        x = x.unsqueeze(1)
        truth = x[1:,:,:]
        init_state = (torch.zeros(1, 1, self.model.hidden_size), torch.zeros(1, 1, self.model.hidden_size))
        x, y = self.model.forward(x[:-1, :, :], init_state)
        pred = x.detach()
        guess = ((pred - truth) ** 2)
        guess = torch.sum(guess, -1).squeeze()
        return guess.detach()

    def forward(self, x):
        return self.get_error(x)

provided_checkpoint_name = 'checkpoints/AnomalyDetector.ckpt'
converted_model_name = 'checkpoints/model.pt'

input_data_name = 'data/test_normal.npy'
provided_model = torch.load(provided_checkpoint_name)
provided_model.eval()

np.savetxt("data/ref_RED.csv", provided_model.RED, delimiter=",")

data = np.load(input_data_name)
data = provided_model.normalize(data)
data = torch.tensor(data)

net = Net()
output_new_net = net.forward(data)
output_provided_net, _ = provided_model._get_reconstruction_error(data, False)

print(output_new_net.numpy())
print(output_provided_net)

traced_module = torch.jit.trace(net, data)
traced_module.save(converted_model_name)

# Check outputs of provided and converted are same
loaded_model = torch.jit.load(converted_model_name)
diff_data_name = 'data/test_normal.npy'
diff_data = np.load(diff_data_name)
diff_data = torch.tensor(diff_data)

loaded_model_out = loaded_model.forward(diff_data)
provided_model_out, _ = provided_model._get_reconstruction_error(diff_data)

print(loaded_model_out.numpy())
print(provided_model_out)
