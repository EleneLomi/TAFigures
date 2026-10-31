import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class DeepLinearNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # 1 input -> 1 hidden unit, no bias
        self.input_to_hidden = nn.Linear(1, 1, bias=False)

        # 1 hidden unit -> 2 output units, no bias
        self.hidden_to_output = nn.Linear(1, 2, bias=False)

        # Initialise every weight to 1
        nn.init.constant_(self.input_to_hidden.weight, 1.0)
        nn.init.constant_(self.hidden_to_output.weight, 1.0)

    def forward(self, x):
        h = self.input_to_hidden(x)
        y_hat = self.hidden_to_output(h)
        return y_hat


# Data
x = torch.tensor([[1.0]])
target = torch.tensor([[-1.0, 1.0]])

# Model
model = DeepLinearNetwork()

# Standard squared-error loss
criterion = nn.MSELoss()

# Plain gradient descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Store trajectory of predictions
trajectory = []

n_steps = 2000

for step in range(n_steps):
    optimizer.zero_grad()

    y_hat = model(x)

    trajectory.append(y_hat.detach().squeeze().numpy().copy())

    loss = criterion(y_hat, target)

    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(
            f"step {step:4d} | "
            f"loss = {loss.item():.6f} | "
            f"prediction = {y_hat.detach().numpy().squeeze()}"
        )


# Final weights
print("\nFinal weights:")
print("input -> hidden:", model.input_to_hidden.weight.data)
print("hidden -> output:", model.hidden_to_output.weight.data)

print("\nFinal prediction:")
print(model(x).detach().numpy())

trajectory = torch.tensor(trajectory)

y1_hat = trajectory[:, 0]
y2_hat = trajectory[:, 1]

plt.figure(figsize=(5, 5))

plt.plot(y1_hat, y2_hat, label="Backpropagation")

# Initial prediction is [1, 1]
plt.scatter([1], [1], marker="o", label="initialisation")

# Target is [-1, 1]
plt.scatter([-1], [1], marker="x", s=100, label="target")

plt.xlabel(r"$\hat{y}_1$")
plt.ylabel(r"$\hat{y}_2$")
plt.axhline(1, linestyle="--", linewidth=1)
plt.legend()
plt.tight_layout()
plt.show()