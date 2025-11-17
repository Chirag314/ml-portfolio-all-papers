from torch import nn


class LeNet5(nn.Module):
    """LeNet-5 (C1-S2-C3-S4-C5-F6-Output) with Tanh + AvgPool.

    C3 is simplified to full conv instead of the original partial connectivity.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.c1 = nn.Conv2d(
            in_channels, 6, kernel_size=5, stride=1, padding=2
        )  # 28x28 -> 28x28
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        self.c3 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # 14x14 -> 10x10
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)  # 10x10 -> 5x5
        self.c5 = nn.Conv2d(16, 120, kernel_size=5, stride=1)  # 5x5 -> 1x1
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.c1(x))
        x = self.s2(x)
        x = self.tanh(self.c3(x))
        x = self.s4(x)
        x = self.tanh(self.c5(x))  # N x 120 x 1 x 1
        x = x.view(x.size(0), -1)  # N x 120
        x = self.tanh(self.f6(x))  # N x 84
        x = self.out(x)  # N x num_classes
        return x
