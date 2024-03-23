from .conv3d import Conv3D as Conv3d
from .resnet import ResNet, BasicBlock

import torch
import torch.nn as nn

class STCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv_block = nn.Conv3d(in_channels=1,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride, 
                                padding=padding,
                                bias=False
                                )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=(1,3,3),
                                     stride=(1,2,2),
                                     padding=(0, 1, 1))

    def forward(self, x):
        return self.max_pool(self.relu(self.batch_norm(self.conv_block(x))))


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=2)
    
    def forward(self, x):
        return self.gru(x)[0]


class ResNet_GRU(nn.Module):
    def __init__(self, in_channels, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.stcnn_1 = STCNN(in_channels=in_channels,
                             out_channels=64,
                             kernel_size=(5,7,7),
                             stride=(1,2,2),
                             padding=(2,3,3))

        self.gru_1 = BiGRU(input_size=512,
                           hidden_size=hidden_size)

        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2])
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2*hidden_size,
                      out_features=output_size)
        )

    def forward(self, x):
        b = x.size(0)
        x = self.stcnn_1(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet(x)
        x = x.view(b, -1, 512)
        x = self.gru_1(x)
        x = torch.cat((x[:, -1, :self.hidden_size], x[:, 0, self.hidden_size:]), dim=1)
        x = self.classifier(x)
        return x