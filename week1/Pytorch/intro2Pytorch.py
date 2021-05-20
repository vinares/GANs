import torch
import torch.nn as nn
import torch.optim as optim

example_tensor = torch.Tensor(
[
     [[1, 2], [3, 4]],
     [[5, 6], [7, 8]],
     [[9, 0], [1, 2]]
    ]
)

print(example_tensor)
print(example_tensor.device)
print(example_tensor.shape)
print(example_tensor.numel())

example_scaler = example_tensor[1,1,1].item()
print(example_scaler)

linear = nn.Linear(10,2)
example_input = torch.randn(3,10)
example_output = linear(example_input)
print(example_output)

relu = nn.ReLU()
relu_output = relu(example_output)
print(relu_output)

batchnorm = nn.BatchNorm1d(2)
batchnorm_output = batchnorm(relu_output)
print(batchnorm_output)

mlp_layer = nn.Sequential(
    nn.Linear(5,2),
    nn.BatchNorm1d(2),
    nn.ReLU()
)

test_example = torch.randn(5,5) + 1
print(test_example)
print(mlp_layer(test_example))

adam_opt = optim.Adam(mlp_layer.parameters(), lr= 10 ** (-1))

train_example = torch.randn(100, 5) + 1
adam_opt.zero_grad()
for i in range(10):
    cur_loss = torch.abs(1 - mlp_layer(train_example)).mean()
    cur_loss.backward()
    adam_opt.step()
    print(cur_loss)


class ExampleModule(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(ExampleModule, self).__init__()
        self.linear = nn.Linear(input_dims, output_dims)
        self.exponent = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        x = self.linear(x)

        # This is the notation for element-wise exponentiation,
        # which matches python in general
        x = x ** self.exponent

        return x

example_model = ExampleModule(10, 2)
print(list(example_model.parameters()))

print(list(example_model.named_parameters()))

input = torch.randn(2, 10)
print(example_model(input))