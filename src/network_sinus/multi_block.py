""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F



class Upsample(nn.Module):
    def __init__(self,  scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor,mode='bilinear',align_corners=True)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels,stride=1,dilation=1,padding=1):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,dilation=dilation,stride=stride,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down_mr(nn.Module):
      
      def __init__(self,in_channels,out_channels):
          super(Down_mr,self).__init__()
          out_channels=int(out_channels/3)
          self.ds0=nn.Sequential(nn.MaxPool2d(2),DoubleConv(in_channels,out_channels))
          self.ds1=nn.Sequential(nn.MaxPool2d(2),DoubleConv(in_channels,out_channels,dilation=2,padding=2,stride=2))
          self.ds2=nn.Sequential(nn.MaxPool2d(2),DoubleConv(in_channels,out_channels,dilation=4,padding=4,stride=4))
      def forward(self,x):
          out1=self.ds0(x)
          out2=F.interpolate(self.ds1(x),(out1.shape[2],out1.shape[3]),mode='bilinear')
          out3=F.interpolate(self.ds2(x),(out1.shape[2],out1.shape[3]),mode='bilinear')
          return torch.cat((out1,out2,out3),1)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False,dilation=1):
        super(Up,self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1,(x2.shape[2],x2.shape[3]),mode='bilinear')
        x1 = torch.cat([x1,x2], dim=1)
        return self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
