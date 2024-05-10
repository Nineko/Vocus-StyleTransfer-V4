import torch
import torch.nn as nn

class CIN(nn.Module):
    def __init__(self, num_features):
        super(CIN, self).__init__()
        self.num_features = num_features
        
    def forward(self, x, gamma,beta):
        # 計算特徵的均值和標準差
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        # 應用條件實例正則化
        normalized = (x - mean) / (std + 1e-5)
        out = normalized * gamma + beta
        
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,style_vector_size):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.cin = CIN(out_channels)
        self.style_gamma = nn.Linear(style_vector_size, out_channels)
        self.style_beta = nn.Linear(style_vector_size, out_channels)

    def forward(self, x, style_vector):
        # 获取本层的gamma和beta
        gamma = self.style_gamma(style_vector).view(-1, self.conv2d.out_channels, 1, 1)
        beta = self.style_beta(style_vector).view(-1, self.conv2d.out_channels, 1, 1)
        
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.cin(out, gamma, beta)
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,style_vector_size, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.cin = CIN(out_channels)
        self.style_gamma = nn.Linear(style_vector_size, out_channels)
        self.style_beta = nn.Linear(style_vector_size, out_channels)

    def forward(self, x, style_vector):
        # 获取本层的gamma和beta
        gamma = self.style_gamma(style_vector).view(-1, self.conv2d.out_channels, 1, 1)
        beta = self.style_beta(style_vector).view(-1, self.conv2d.out_channels, 1, 1)
        
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.cin(out, gamma, beta)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels,style_vector_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1,style_vector_size=style_vector_size)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1,style_vector_size=style_vector_size)

    def forward(self, x, style_vector):
        residual = x
        out = self.relu(self.conv1(x, style_vector))
        out = self.conv2(out, style_vector)
        out = out + residual
        return out 

# Image Transform Network
class TransformNet(nn.Module):
    def __init__(self,style_vector_size = 2048):
        super(TransformNet, self).__init__()
        
        # nonlineraity
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # encoding layers
        self.conv1 = ConvLayer(in_channels=3 , out_channels=32 , kernel_size=9, stride=1,style_vector_size=style_vector_size)
        self.conv2 = ConvLayer(in_channels=32, out_channels=64 , kernel_size=3, stride=2,style_vector_size=style_vector_size)
        self.conv3 = ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=2,style_vector_size=style_vector_size)

        self.res1 = ResidualBlock(channels=128,style_vector_size=style_vector_size)
        self.res2 = ResidualBlock(channels=128,style_vector_size=style_vector_size)
        self.res3 = ResidualBlock(channels=128,style_vector_size=style_vector_size)
        self.res4 = ResidualBlock(channels=128,style_vector_size=style_vector_size)
        self.res5 = ResidualBlock(channels=128,style_vector_size=style_vector_size)

        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1,style_vector_size=style_vector_size, upsample=2 )
        self.deconv2 = UpsampleConvLayer(64 , 32, kernel_size=3, stride=1,style_vector_size=style_vector_size, upsample=2 )
        self.deconv1 = UpsampleConvLayer(32 , 3 , kernel_size=9, stride=1,style_vector_size=style_vector_size)

    def forward(self, x ,style_vector):

        y = self.relu(self.conv1(x,style_vector))
        y = self.relu(self.conv2(y,style_vector))
        y = self.relu(self.conv3(y,style_vector))

        y = self.res1(y,style_vector)
        y = self.res2(y,style_vector)
        y = self.res3(y,style_vector)
        y = self.res4(y,style_vector)
        y = self.res5(y,style_vector)

        y = self.relu(self.deconv3(y,style_vector))
        y = self.relu(self.deconv2(y,style_vector))
        y = self.deconv1(y,style_vector)
        y = self.sigmoid(y)

        return y