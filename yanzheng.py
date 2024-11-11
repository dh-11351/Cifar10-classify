import torchvision
import torch
from  PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():

    return ResNet(ResidualBlock)

folder_path = 'D:\shenduxuexi\keshe\yanzhengtupian'
# 遍历文件夹内的图片文件
for filename in os.listdir(folder_path):

        if filename.endswith('.jpg') :
          image = Image.open(os.path.join(folder_path, filename))
          #image.show()
          # image = Image.open("D:\shenduxuexi\keshe\cat01.jpg")
          #print(image)  #<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=719x719 at 0x1BB943224C0>
          #  这里可以看到输出是ARGB类型，四通道，而我们的训练模式都是三通道的。
          #  所以这里使转换成RGB三通道的格式

          image = image.convert('RGB')

          # 使用Compose组合改变数据类型,先变成32*32的 然后在变成tensor类型
          transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

          image = transform(image)
          model = ResNet18()
          model.load_state_dict(torch.load('net_100.pth'))
          image = torch.reshape(image,(1,3,32,32))
          #print(image.shape)

          model.eval()
          with torch.no_grad():
             #image = image.cuda()
             output = model(image)
             #print(output)
             if output.argmax(1) == 0:
                print('plane')
             elif output.argmax(1) == 1:
                 print('automobile')
             elif output.argmax(1) == 2:
                print('bird')
             elif output.argmax(1) == 3:
                print('cat')
             elif output.argmax(1) == 4:
                print('deer')
             elif output.argmax(1) == 5:
                print('dog')
             elif output.argmax(1) == 6:
                print('frog')
             elif output.argmax(1) == 7:
                print('horse')
             elif output.argmax(1) == 8:
                print('ship')
             elif output.argmax(1) == 9:
                print('truck')
