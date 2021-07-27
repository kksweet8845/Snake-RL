from torch import nn
import torch.nn.functional as F
import torch
import random

# def conv2d_size_out(size, kernel_size, stride):
#     return (size - (kernel_size -1) -1) // stride + 1


        
    

class DQN_DNN(nn.Module):
    def __init__(self, in_features, outputs, device='cpu'):
        super(DQN_DNN, self).__init__()


        self.device = device

        self.l1 = nn.Linear(in_features, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, outputs)
        self.layers = [self.l1, self.l2, self.l3, self.l4, self.l5]

    def init_weights(self):
        for layer in self.layers:
            nn.init.normal_(layer.weight)
            nn.init.normal_(layer.bias)

    def gen_eps_mask(self, shape, ratio):
        w = torch.empty(shape)
        w = torch.nn.init.normal_(w)

        mask = torch.empty(shape)
        mask = torch.nn.init.uniform_(mask)

        w = w.masked_fill(mask < (1-ratio), 0)
        return w


    def randomly_alter(self, ratio):

        with torch.no_grad():
            for layer in self.layers:
                for param_tensor in layer.state_dict():
                    tensor = layer.state_dict()[param_tensor] 
                    w = self.gen_eps_mask(tensor.size(), ratio)   
                    w = w.cuda()   
                    tensor += w
                    tensor = torch.clamp(tensor, -1, 1)


    def forward(self, x):

        x = x.to(self.device)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)

        return x


class DQN(nn.Module):
    def __init__(self, h, w, outputs, device='cpu'):
        super(DQN, self).__init__()

        self.device = device
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        # self.bn4 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=3, stride=2, padding=1):
            return (size + 2*padding - (kernel_size - 1) - 1) // stride + 1

        convw, convh = w, h
        for i in range(3):
            convw = conv2d_size_out(convw, kernel_size=3, stride=2)
            convh = conv2d_size_out(convh, kernel_size=3, stride=2)

        # print(convw, convh)

        linear_input_size = convw * convh * 128

        self.fc1 = nn.Linear(linear_input_size, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, outputs)
        # self.fc2 = nn.Linear(256, outputs)

    

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.shape)
        x = self.fc1(x.view(x.size(0), -1))
        # x = F.relu(self.bn4(self.conv4(x)))
        x = self.fc2(x)

        return self.fc3(x)

