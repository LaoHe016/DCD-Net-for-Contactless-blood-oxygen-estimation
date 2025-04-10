import torch
import torch.nn as nn
import numpy as np
from bert import BERT
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)  
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重

# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # 通道注意力实例
        self.sa = SpatialAttention(kernel_size)  # 空间注意力实例

    def forward(self, x):
        out = x * self.ca(x)  # 使用通道注意力加权输入特征图
        result = out * self.sa(out)  # 使用空间注意力进一步加权特征图
        return result  # 返回最终的特征图
    
def normalize(array):  
    if isinstance(array, torch.Tensor):  
        # 对于 PyTorch 张量，使用 PyTorch 方法  
        m = array.mean()  
        s = array.std()  
    else:  
        # 对于 NumPy 数组，使用 NumPy 函数  
        m = np.mean(array)  
        s = np.std(array)  
      
    if s == 0:   
        return array - m  
    else:  
        return (array - m) / s  

import torchvision.models as models

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, output_dim=1):
        super(ResNet50FeatureExtractor, self).__init__()
        # 加载预训练的ResNet50模型
        self.resnet50 = models.resnet50(pretrained=True)
        
        # 修改第一个卷积层以接受1通道输入（rPPG信号通常是单通道的）
        self.resnet50.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 移除全连接层，因为我们只需要特征提取
        self.features = nn.Sequential(*list(self.resnet50.children())[:-2])
        
        # 添加一个全局平均池化层，以减少特征图的空间维度
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 根据您的需要，可以添加额外的层来处理特征
        # 例如，这里我们添加了一个线性层来将特征图转换为一维特征向量
        self.fc = nn.Linear(self.resnet50.fc.in_features, output_dim)

    def forward(self, x):
        # 前向传播
        # print(x.shape)

        x = self.features(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.fc(x)
        # print(x.shape)

        return x

# # 假设您的rPPG信号数据的形状是（批量大小，通道数，序列长度，1）
# # 例如，批量大小为10，通道数为1，序列长度为600
# input_data = torch.randn(10, 1, 600, 1)

# # 实例化特征提取器
# feature_extractor = ResNet50FeatureExtractor(input_channels=1, output_dim=128)  # 假设我们想要128维的特征向量

# # 将输入数据传递给特征提取器
# features = feature_extractor(input_data)

# # 打印特征的形状
# print(features.shape)

class Fusion_Block(nn.Module):
    def __init__(self):
        super(Fusion_Block, self).__init__()

    def forward(self, x0,y0):
        """Definition of Fusion_Stem.
        Args:
          x [B,C,L] 原始的像素数据
        Returns:
          fusion_x [N*D,C,H/4,W/4]
        """
        B, C, L = x0.shape
        # 初始化一个张量来存储结果
        result1 = torch.zeros(B, C, L).to(device)

        # 对x的第三个维度（600）进行平移和点乘操作
        for i in range(L):
            # 将x的第三个维度向右平移i位
            x_rolled = torch.roll(x0, shifts=i, dims=2)  # dims=2表示第三个维度
            # 与y按位相乘
            product = x_rolled * y0
            # 将结果累加到result中
            result1 += product

        # 将result中的600个结果按位求平均
        result1 = result1 / L

        # 初始化一个张量来存储结果
        result2 = torch.zeros(B, C, L).to(device)

        # 对x的第三个维度（600）进行平移和点乘操作
        for i in range(L):
            # 将x的第三个维度向右平移i位
            y_rolled = torch.roll(y0, shifts=i, dims=2)  # dims=2表示第三个维度
            # 与y按位相乘
            product = y_rolled * y0
            # 将结果累加到result中
            result2 += product

        # 将result中的600个结果按位求平均
        result2 = result2 / L

        result = result1 + result2
        
        return result

class Fusion_Stem(nn.Module):
    def __init__(self):
        super(Fusion_Stem, self).__init__()

        self.stem11 = nn.Sequential(nn.Conv1d(12, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )
        
        self.stem12 = nn.Sequential(nn.Conv1d(12, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )

        self.stem21 =nn.Sequential(
            nn.Conv1d(128, 36, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(36),
            nn.ReLU(inplace=True),
        )

        self.stem22 =nn.Sequential(
            nn.Conv1d(128, 36, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(36),
            nn.ReLU(inplace=True),
        )

        self.apha = nn.Parameter(torch.tensor(0.5))
        self.belta = nn.Parameter(torch.tensor(0.5))
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [B,C,L] 原始的像素数据
        Returns:
          fusion_x [N*D,C,H/4,W/4]
        """
        B, C, L = x.shape
        x0 = x.view(B,4,3,L).permute(0,2,1,3)
        xr = x0[:,0,:,:]
        xg = x0[:,1,:,:]
        xb = x0[:,2,:,:]
        x1 = xr - xg
        x2 = xr - xb
        x3 = xg - xb

        x_diff = torch.cat([x1,x2,x3],1).view(B,4*3,L)
        # x_diff = normalize(x_diff)
        x_diff = self.stem12(x_diff)

        x = self.stem11(x)

        #fusion layer1
        x_path1 = self.apha*x + self.belta*x_diff
        x_path1 = self.stem21(x_path1)
        #fusion layer2
        x_path2 = self.stem22(x_diff)
        x = self.a*x_path1 + self.b*x_path2
        
        return x
    
class Fusion_Stem1(nn.Module):
    def __init__(self):
        super(Fusion_Stem1, self).__init__()

        self.stem11 = nn.Sequential(nn.Conv1d(12, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )
        
        self.stem12 = nn.Sequential(nn.Conv1d(12, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )

        self.stem21 =nn.Sequential(
            nn.Conv1d(128, 36, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(36),
            nn.ReLU(inplace=True),
        )

        self.stem22 =nn.Sequential(
            nn.Conv1d(128, 36, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(36),
            nn.ReLU(inplace=True),
        )

        self.apha = nn.Parameter(torch.tensor(0.5))
        self.belta = nn.Parameter(torch.tensor(0.5))
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.CBAM_block_32 = CBAM(128)#

        self.fc1 = nn.Linear(1200, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [B,C,L] 原始的像素数据
        Returns:
          fusion_x [N*D,C,H/4,W/4]
        """
        B, C, L = x.shape
        x0 = x
        x012 = torch.cat([x0[:,:,:1],x0[:,:,:L-1]],2)
        x123 = x0
        x234 = torch.cat([x0[:,:,1:],x0[:,:,L-1:]],2)
        x1 = x123 - x012
        x2 = x234 - x123
        x3 = x234 - x012
        x_diff = torch.cat([x1,x2,x3],1).view(B,C*3,L)
        # x_diff = normalize(x_diff)
        x_diff = self.stem12(x)

        x = self.stem11(x)

        #fusion layer1
        x_path1 = self.apha*x + self.belta*x_diff
        # x_path1 = self.CBAM_block_32(x_path1)
        x_path1 = self.stem21(x_path1)
        #fusion layer2
        x_path2 = self.stem22(x_diff)
        x = self.a*x_path1 + self.b*x_path2
        # f1 = x.view(-1,1200)
        # f2 = self.fc1(f1)
        # f3 = self.relu(f2)
        # f4 = self.fc2(f3)
        # f5 = self.relu(f4)
        # net1out = self.fc3(f5)
        return x

class RESCNN(nn.Module):
    def __init__(self):
        super(RESCNN, self).__init__()
        self.ConvBlock1 = nn.Sequential(
            nn.Conv1d(36,64,kernel_size = 7,stride = 1,padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )

        self.ConvResBK16_1 = nn.Sequential(
            nn.Conv1d(64,32,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32,32,kernel_size = 3,stride = 1,padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32,64,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.ConvResBK16_2 = nn.Sequential(
            nn.Conv1d(64,32,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32,32,kernel_size = 3,stride = 1,padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32,64,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.ConvResBK16_3 = nn.Sequential(
            nn.Conv1d(64,32,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32,32,kernel_size = 3,stride = 1,padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32,64,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv1d(64,128,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        )
        self.ConvResBK32_1 = nn.Sequential(
            # nn.AvgPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(64,64,kernel_size = 3,stride = 1,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            nn.Conv1d(64,128,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.ConvResBK32_2 = nn.Sequential(
            nn.Conv1d(128,64,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64,kernel_size = 3,stride = 1,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,128,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.ConvResBK32_3 = nn.Sequential(
            nn.Conv1d(128,64,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64,kernel_size = 3,stride = 1,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,128,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.ConvResBK32_4 = nn.Sequential(
            nn.Conv1d(128,64,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64,kernel_size = 3,stride = 1,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,128,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        # self.ConvBlock3 = nn.Sequential(
        #     nn.Conv1d(128,64,kernel_size = 1,stride = 1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(64,128,kernel_size = 1,stride = 1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU()
        # )

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.CBAM_block_128 = CBAM(128)
        self.CBAM_block_64 = CBAM(64)
        
        self.fc1 = nn.Linear(32*9, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        # 进来的形状是bs,36,600
        # x = x.unsqueeze(1)
        # print(x.shape)
        
        bk1 = self.ConvBlock1(x)#bs,64,300
        #print(f"bk1的形状是{bk1.shape}")#bs,64,150
        # avg1 = self.max_pool(bk1) #bs,64,300
        out1 = self.relu(self.ConvResBK16_1(bk1) + bk1)#bs,64,300
        # avg2 = self.max_pool(out1) #bs,64,150
        out2 = self.relu(self.ConvResBK16_2(out1) + out1)#bs,64,300
        # avg3 = self.max_pool(out2) #bs,64,75
        out3 = self.relu(self.ConvResBK16_2(out2) + out2)#bs,64,300

        cbam1 = self.CBAM_block_64(out3)#bs,64,300

        avg_2 = self.ConvBlock2(cbam1)#bs,64,75
        out4 = self.relu(self.ConvResBK32_1(cbam1) + avg_2)#bs,128,150
        # avg4 = self.max_pool(out4) #bs,64,37
        out5 = self.relu(self.ConvResBK32_2(out4) + out4)#bs,128,150
        # avg5 = self.max_pool(out5) #bs,64,18
        out6 = self.relu(self.ConvResBK32_3(out5) + out5)#bs,128,150
        # avg6 = self.max_pool(out6) #bs,64,9
        out7 = self.relu(self.ConvResBK32_4(out6) + out6)#bs,128,150

        # 通道、空间注意力机制
        cbam = self.CBAM_block_128(out7)#bs,128,150
        
        # out8 = self.ConvBlock3(cbam)#bs,32,600
        
        # f1 = out8.view(-1,288)
        # f2 = self.fc1(f1)
        # f3 = self.relu(f2)
        # f4 = self.fc2(f3)
        # f5 = self.relu(f4)
        # net1out = self.fc3(f5)
        # out9 = self.sigmoid(f6)*100*0.4 + 60

        return cbam
    



class DoubleConv(nn.Module):
    ""'双卷积块，通常用于U-Net网络的编码器和解码器部分。""'
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    ""'下采样块，用于U-Net网络的编码器部分。""'
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    ""'上采样块，用于U-Net网络的解码器部分。""'
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 输入x1和x2的形状需要匹配，可能需要crop或pad操作
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        if diffY != 0:
            x1 = x1[:, :, :x2.size()[2], :]
        if diffX != 0:
            x1 = x1[:, :, :, :x2.size()[3]]
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # print(x.shape)
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.down4(x4)
        # print(x5.shape)
        x = self.up1(x5, x4)
        # print(x.shape)
        x = self.up2(x, x3)
        # print(x.shape)
        x = self.up3(x, x2)
        # print(x.shape)
        x = self.up4(x, x1)
        # print(x.shape)
        logits = self.outc(x)
        # print(logits.shape)
        return logits


class Last_layer(nn.Module):
    def __init__(self):
        super(Last_layer, self).__init__()

        self.Last1 = nn.Sequential(
            nn.Conv1d(128,64,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,32,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.Last21 = nn.Sequential(
            nn.Conv1d(32,16,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),

            nn.Conv1d(16,8,kernel_size = 1,stride = 1),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )

        self.CBAM_block_32 = CBAM(32)#

        self.fc1 = nn.Linear(72, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

        # self.apha = nn.Parameter(torch.tensor(0.5))
        # self.belta = nn.Parameter(torch.tensor(0.5))
        # self.a = nn.Parameter(torch.tensor(0.5))
        # self.b = nn.Parameter(torch.tensor(0.5))
        # self.c = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [B,C,L] 原始的像素数据
        Returns:
          fusion_x [N*D,C,H/4,W/4]
        """

        #last layer1
        # x_1 = self.apha*x + self.belta*y
        x_1 = self.Last1(x)
        #last layer2
        # print(x_1.shape)
        x_1 = self.CBAM_block_32(x_1)
        x = self.Last21(x_1)
        # print(x.shape)
        f1 = x.view(-1,72)
        f2 = self.fc1(f1)
        f3 = self.relu(f2)
        f4 = self.fc2(f3)
        f5 = self.relu(f4)
        net1out = self.fc3(f5)
        return net1out

class rPPGEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, time_steps, stride):
        super(rPPGEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_steps = time_steps
        self.stride = stride
        self.embedding = nn.ModuleList(
            [nn.Linear(input_dim * time_steps, output_dim) for _ in range(9)]
        )

    def forward(self, x):
        # x: (batch_size, 36, 150)
        batch_size = x.size(0)
        # print(x.shape)  # [32, 128, 150]
        # Reshape x to (batch_size, 36, time_steps, (150 - time_steps) // stride + 1)
        # x = x.view(batch_size, self.input_dim, self.time_steps, -1)
        
        # Extract windows of time_steps with stride
        windows = []
        for i in range((x.size(2)-self.time_steps)//self.stride+1):
            embedded = self.embedding[i](x[:, :, i:i+self.time_steps].reshape(batch_size, -1)).unsqueeze(1)
            windows.append(embedded)
        
        # # Embed each window
        # embedded_windows = [self.embedding(window).unsqueeze(1) for window in windows]
        
        # Concatenate all embedded windows
        embedded_sequence = torch.cat(windows, dim=1)
        # print(embedded_sequence.shape)
        # import time
        # time.sleep(10)
        
        return embedded_sequence

class SpO2Net(nn.Module):
    def __init__(self):
        super(SpO2Net, self).__init__()

        # self.bk11 = nn.Sequential(nn.Conv1d(36, 128, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(inplace=True),
        #     )
        
        # self.stem12 = nn.Sequential(nn.Conv1d(36, 512, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        #     )

        # self.stem21 =nn.Sequential(
        #     nn.Conv1d(512, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(inplace=True),
        # )

        # self.stem22 =nn.Sequential(
        #     nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )

        # self.apha = nn.Parameter(torch.tensor(0.5))
        # self.belta = nn.Parameter(torch.tensor(0.5))

        self.fusion_stem0 = Fusion_Stem()
        # self.fusion_stem11 = Fusion_Stem1()
        self.fusion_stem12 = Fusion_Stem1()
        self.bert_128 = BERT(hidden= 128,
                            n_layers=[10,10,10,10],
                            attn_heads=  [
                                [16,16,16,16,16,16,16,16,16,16],
                                [8,8,8,8,8,8,8,8,8,8],
                                [4,4,4,4,4,4,4,4,4,4],
                                [2,2,2,2,2,2,2,2,2,2]
                            ],
                            dropout=0.1)
        # self.bert_128 = BERT(hidden= 128,
        #                     n_layers=[8,4,4,4],
        #                     attn_heads=  [
        #                         [1,2,4,8,16,32,64,128],
        #                         [1,2,4,8],
        #                         [4,8,16,32],
        #                         [16,32,64,128]
        #                     ],
        #                     dropout=0.1)
        
        # self.rescnn_o = RESCNN()
        # self.rescnn_z = RESCNN()
        self.rescnn_fb = RESCNN()
        self.fb = Fusion_Block()
        self.last = Last_layer()
        self.rppg_embedding = rPPGEmbedding(128,128,30,15)
        # self.unet = UNet(1, 1)
        # self.r50 = ResNet50FeatureExtractor(input_channels=3,output_dim=1)

    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [B,C,L] 原始的像素数据
        Returns:
          fusion_x [N*D,C,H/4,W/4]
        """
        # print(torch.mean(x))
        # x_o = x[:,:36]
        # x_o = torch.cat([x[:,:9,:],x[:,33:36,:]],1)
        x_o = x[:,0:12,:]

        x_z = x[:,12:24,:]

        x_o = self.fusion_stem0(x_o)# bs,36,150
        x_z = self.fusion_stem12(x_z)# bs,36,150

        # x_o = self.rescnn_o(x_o) # bs,128,150
        # x_z = self.rescnn_z(x_z) # bs,128,150

        fb = self.fb(x_o,x_z) # bs,128,150

        # x = self.unet(fb.unsqueeze(1)).squeeze(1)
        # print(fb.shape)
        fb = self.rescnn_fb(fb) # bs,128,150

        rppg_embedding = self.rppg_embedding(fb) # bs,9,128
        # print(rppg_embedding.shape)
        rppg_embedding = rppg_embedding.permute(0,2,1) # bs,128,150

        x = self.bert_128(rppg_embedding)# bs,128,9
        # print(x.shape)

        x = self.last(x)
        # x_z = self.fusion_stem12(x_z)# bs,36,150
        # x = x + x_z

        # x_o = x_o.view(-1,12,3,150).permute(0,2,1,3)
        # x = self.r50(x_o)

        # print(f"x的形状是{x.shape}")#bs,36,600
        # print(x_z.shape)
        # x_z = self.bert_36(x_z)# bs,36,150
        # x_z = x_z.permute(0,2,1)
        # x_z = self.bk11(x_z)
        # x_z = x_z.permute(0,2,1)
        # print(x_z.shape)
        # x_z = self.bert_128(x_z)# bs,128,150
        # print(x_o.shape)
        # x_o = self.rescnn(x_o) # bs,128,150
        # print(x_o.shape)
        # print(x_o.shape)
        # print(x_z.shape)
        # x = self.last(x_o)
        
        return x
