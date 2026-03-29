import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)

class ChessNet(nn.Module):
    def __init__(self, blocks=10, filters=128):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(119, filters, 3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(filters) for _ in range(blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*8*8, 4672)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        return self.policy_head(x), self.value_head(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = ChessNet().to(device)
    dummy = torch.zeros(1, 119, 8, 8).to(device)
    policy, value = model(dummy)
    print("Policy shape:", policy.shape)
    print("Value shape:", value.shape)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))


'''python -m model.network'''