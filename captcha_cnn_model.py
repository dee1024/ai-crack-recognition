# -*- coding: UTF-8 -*-
import torch.nn as nn
import captcha_setting
import torchvision.models as models
import torchvision

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear((captcha_setting.IMAGE_WIDTH//8)*(captcha_setting.IMAGE_HEIGHT//8)*64, 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.rfc = nn.Sequential(
            nn.Linear(1024, captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out



class RES18(nn.Module):
    def __init__(self):
        super(RES18, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.resnet18(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class RES34(nn.Module):
    def __init__(self):
        super(RES34, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.resnet34(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class RES50(nn.Module):
    def __init__(self):
        super(RES50, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.resnet50(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class RES101(nn.Module):
    def __init__(self):
        super(RES101, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.resnet101(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class RES152(nn.Module):
    def __init__(self):
        super(RES152, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.resnet152(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class ALEXNET(nn.Module):
    def __init__(self):
        super(ALEXNET, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.alexnet(pretrained=False)
        self.base.classifier[-1] = nn.Linear(4096, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.vgg11(pretrained=False)
        self.base.classifier[-1] = nn.Linear(4096, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.vgg13(pretrained=False)
        self.base.classifier[-1] = nn.Linear(4096, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out
        
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.vgg16(pretrained=False)
        self.base.classifier[-1] = nn.Linear(4096, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.vgg19(pretrained=False)
        self.base.classifier[-1] = nn.Linear(4096, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class SQUEEZENET(nn.Module):
    def __init__(self):
        super(SQUEEZENET, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.squeezenet1_0(pretrained=False)
        self.base.classifier[-3] = nn.Linear(512, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class DENSE161(nn.Module):
    def __init__(self):
        super(DENSE161, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.densenet161(pretrained=False)
        self.base.classifier = nn.Linear(self.base.classifier.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class MOBILENET(nn.Module):
    def __init__(self):
        super(MOBILENET, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.mobilenet_v2(pretrained=False)
        self.base.classifier = nn.Linear(self.base.last_channel, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out
