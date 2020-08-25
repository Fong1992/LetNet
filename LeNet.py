import torch
import torch.nn as nn

LeNet = [6, 'M', 16, 'M', 120]


class Le_net(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(Le_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(LeNet)

        self.fcs = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)), nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Le_net(in_channels=1, num_classes=10).to(device)
x = torch.randn(64, 1, 32, 32).to(device)
print(model(x).shape)
