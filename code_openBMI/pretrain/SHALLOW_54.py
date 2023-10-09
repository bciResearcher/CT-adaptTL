import torch
import torch.nn as nn
from torch.nn import functional as F


class deep(nn.Module):
    def __init__(self):
        super(deep, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(25, 1), stride=(1, 1))
        self.conv1_1 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(1, 20), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool1 = nn.AvgPool2d(kernel_size=(75, 1), stride=(15, 1), padding=0)

        self.fc = nn.Sequential(
            nn.Linear(in_features=5080 , out_features=200),
            nn.ReLU()
        )

        self.conv_classifier = nn.Sequential(
            nn.Linear(in_features=200, out_features=2),

        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_1(out)
        out = F.elu(self.bn1(out))
        out = F.dropout(self.pool1(out), p=0.5, training=self.training)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.conv_classifier(out)
        out = nn.LogSoftmax(dim=1)(out)

        return out


if __name__ == '__main__':
    model = deep()
    print(model)
    for param in model.named_parameters():
        print(param[0])