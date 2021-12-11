import torch
import torch.nn as nn
import torch.nn.functional as F

from coordconv import CoordConv2d


class STNetBirds(nn.Module):
    def __init__(self):
        super(STNetBirds, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=8)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(20, 40, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16000, 1000)
        self.fc2 = nn.Linear(1000, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 46 * 46, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 46 * 46)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ConvNetBirds(nn.Module):
    def __init__(self):
        super(ConvNetBirds, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=8)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(20, 40, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16000, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CoordConvNetBirds(nn.Module):
    def __init__(self):
        super(CoordConvNetBirds, self).__init__()
        self.coordconv1 = CoordConv2d(3, 10, 8)
        self.coordconv2 = CoordConv2d(10, 20, 7)
        self.coordconv3 = CoordConv2d(20, 40, 5)
        self.coordconv2_drop = nn.Dropout2d()
        self.coordconv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16000, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.coordconv1(x), 2))
        x = F.relu(F.max_pool2d(self.coordconv2_drop(self.coordconv2(x)), 2))
        x = F.relu(F.max_pool2d(self.coordconv3_drop(self.coordconv3(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class STCoordNetBirds(nn.Module):
    def __init__(self):
        super(STCoordNetBirds, self).__init__()
        self.coordconv1 = CoordConv2d(3, 10, 8)
        self.coordconv2 = CoordConv2d(10, 20, 7)
        self.coordconv2_drop = nn.Dropout2d()
        self.coordconv3 = CoordConv2d(20, 40, 5)
        self.coordconv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16000, 1000)
        self.fc2 = nn.Linear(1000, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 46 * 46, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 46 * 46)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.coordconv1(x), 2))
        x = F.relu(F.max_pool2d(self.coordconv2_drop(self.coordconv2(x)), 2))
        x = F.relu(F.max_pool2d(self.coordconv3_drop(self.coordconv3(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

