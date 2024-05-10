import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

class StylePredict(nn.Module):
    def __init__(self):
        super(StylePredict, self).__init__()
        self.inception_v3 = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        #for param in self.inception_v3.parameters():
        #    param.requires_grad = False
        self.features = nn.Sequential(*list(self.inception_v3.children())[:15])
        self.avgpool2d = nn.AvgPool2d(17)
        self.flat = nn.Flatten()
        self.dense1 = nn.Linear(768, 256)
        self.dense2 = nn.Linear(256, 2048)
    def forward(self, x):
        out = self.features(x)
        out = self.avgpool2d(out)
        out = self.flat(out)
        out = self.dense1(out)
        out = self.dense2(out)
        return out

#model = StylePredict().to('cuda')
#summary(model, input_size=(3, 299, 299))  