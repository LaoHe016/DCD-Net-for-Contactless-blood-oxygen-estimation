from tqdm import tqdm 

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from torch.utils.data import ConcatDataset
import os

from torch.autograd import Variable
from Scheduler import GradualWarmupScheduler

from torch import optim

import matplotlib.pyplot as plt
from pylab import mpl
import torch
import test_SPO2CNN
from 模型 import SpO2Net
from 损失 import Heyuanxia_Loss
from 数据加载器 import SPO2Dataset
import 数据加载器
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(figsize=(2,3),dpi=100)
# 设置正确显示符号
mpl.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

boundaries = None
group_counts = None
参数 = None

def getnum():
    # 先在输出文件夹中看看下一个要输出的时候什么文件名
    output_path = "./checkpoints/exp"
    count = 0
    path = "./checkpoints/exp"+str(count)
    while os.path.exists(path):
        count += 1
        path = "./checkpoints/exp"+str(count)
    # 如果目录不存在，则创建它  
    if not os.path.exists(path):  
        os.makedirs(path)
    return count

def 记录log(文字,输出文件夹名称):
    if not os.path.exists(f'./checkpoints/exp{输出文件夹名称}/log'):
        os.makedirs(f'./checkpoints/exp{输出文件夹名称}/log')
    log_path = os.path.join(
        f'./checkpoints/exp{输出文件夹名称}/log/','record' + '.log')
    # 打开文件用于写入
    output_file = open(log_path, 'a')

    # 重定向标准输出和错误输出到文件
    sys.stdout = output_file

    # 使用print函数输出内容
    print(文字, file=sys.stdout)

    # 恢复标准输出和错误输出
    sys.stdout = sys.__stdout__

    # 关闭文件
    output_file.close()

def quantize_tensor(tensor, num_groups):
    # 确保张量是一维的
    tensor = tensor.view(-1)
    # 获取张量的最小值和最大值
    min_val = 100.5-num_groups
    max_val = 100.5
    # 计算每个组的间隔
    interval = (max_val - min_val) / num_groups
    # 生成分界点
    boundaries = [min_val + i * interval for i in range(num_groups + 1)]
    
    # 分组并计数
    group_counts = [0] * num_groups
    
    print("\n")
    # 将张量的值分配到各个组
    for 第几个,value in enumerate(tensor):
        print(f"分组第{第几个}个数据",end="\r")
        # 确定值属于哪个组
        for i in range(num_groups):
            if value > (boundaries[i]) and value <= boundaries[i + 1]:
                group_counts[i] += 1
                break
    
    return boundaries, group_counts

def 计算加权误差损失(inputs, targets,误差):
    global boundaries, group_counts
    # 确保张量是一维的
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # 分组,每个batch进行
    # boundaries, group_counts = quantize_tensor(targets, num_groups)
    # print("分界点:", boundaries)
    # print("每个组的个数:", group_counts)
    
    # 初始化权重张量
    weights = torch.ones_like(targets)
    
    # 计算每个分组的权重
    batch总长度 = len(targets)

    for i, count in enumerate(group_counts):
        # 计算每个分组的权重，个数少的组权重高
        weight = (batch总长度 / (2*count)) if count > 0 else 0
        # 应用权重
        group_mask = (targets > boundaries[i]+ 0.0001) & (targets <= boundaries[i+1])
        weights[group_mask] = weight
    # print(f"权重：{weights}")
    
    # 计算加权均方误差
    if 误差 == '均方误差':
        squared_errors = (inputs - targets) ** 2
        weighted_squared_errors = squared_errors * weights
        weighted_mse = torch.mean(weighted_squared_errors)
        return weighted_mse
    elif 误差 == '平均绝对误差':
        # 计算绝对误差
        absolute_errors = torch.abs(inputs - targets)
        # 计算加权绝对误差
        weighted_absolute_errors = absolute_errors * weights
        # 计算加权平均绝对误差
        weighted_mae = torch.mean(weighted_absolute_errors)
        return weighted_mae

# 定义一个函数来绘制分布
def plot_distribution(dataloader, title,count):
    global boundaries, group_counts

    gt_SPO2_values = []
    for i, (et, gt_SPO2) in enumerate(dataloader):
        # gt_SPO2_flat = torch.mean(gt_SPO2, dim=2).view(-1).tolist()
        gt_SPO2_flat = gt_SPO2
        # print(f"第{i+1}个数据  形状={torch.mean(gt_SPO2, dim=2).shape}",end='\r')
        # print(f"第{i+1}个数据  形状={gt_SPO2.shape} 数值是={gt_SPO2}",end='\r')
        gt_SPO2_values.extend(gt_SPO2_flat)  # 假设gt_SPO2是Tensor，如果不是，需要相应转换
    # 将列表转换为张量
    gt_SPO2_tensor = torch.tensor(gt_SPO2_values, dtype=torch.float32)
    boundaries, group_counts = quantize_tensor(gt_SPO2_tensor, 15)# 15是分组
    from matplotlib.ticker import MultipleLocator
    # 关闭所有图形窗口
    plt.close('all')
    # 设置图像的尺寸
    plt.figure(figsize=(10, 6))  # 宽度为10英寸，高度为8英寸
    # 绘制直方图，并获取返回值
    n, bins, patches = plt.hist(gt_SPO2_values, bins=50, alpha=0.7, color='blue')

    # plt.title("ViInHealth Dataset SpO2 Distribution")
    plt.title(title)
    plt.xlabel('gt_SPO2 Values(%)')
    plt.ylabel('Number of samples')
    plt.grid(True)

    # 设置横坐标的刻度间隔为1（整数刻度）
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    # 在每个柱状图上显示数据值
    for i in range(len(patches)):
        if n[i] == 0:
            continue 
        plt.text(bins[i] + (bins[i+1] - bins[i]) / 2, n[i], str(int(n[i])), ha='center', va='bottom')

    # 保存直方图到文件
    plt.savefig(f'./checkpoints/exp{count}/{title}.png')  # 指定文件名和路径
    # plt.show()
    # 关闭图形，释放内存
    plt.close()


def train(参数字典):
    global 参数
    参数 = 参数字典
    global count
    count = None
    if 参数字典['输出文件夹名'] == None:
        count = getnum()
    else:
        count = 参数字典['输出文件夹名']
        output_path = "./checkpoints/exp"
        path = output_path+str(count)
        # 如果目录不存在，则创建它  
        if not os.path.exists(path):  
            os.makedirs(path)
    记录log(f"{参数字典}",count)

    output_path = "./checkpoints/exp"
    path = output_path+str(count)+'/best_model'
    # 如果目录不存在，则创建它  
    if not os.path.exists(path):  
        os.makedirs(path)

    model = SpO2Net()
    model = model.to(device)
    不加载数据 = []
    不加载数据.extend(参数字典["测试数据名称"])
    不加载数据.extend(参数字典["验证数据名称"])
    # 加载数据=参数字典["训练数据名称"]
    # print(f'加载数据集:\n{加载数据}')
    print(f'不加载数据集:\n{不加载数据}')
    train_set = SPO2Dataset(参数字典['训练数据文件夹'],D=参数字典['D'],step=参数字典['步长'],参数 = 参数字典,不加载数据=不加载数据,重采样=参数字典['重采样'])
    
    加载数据 = []
    加载数据.extend(参数字典["验证数据名称"])
    加载数据.extend(参数字典["测试数据名称"])
    print("加载数据")
    print(加载数据)
    不加载数据 = []
    # 不加载数据.extend(参数字典["测试数据名称"])
    # 不加载数据.extend(参数字典["验证数据名称"])
    # 不加载数据.extend(参数字典["不参加训练验证的文件夹"])
    # 训练集增强 = 参数字典["数据增强方法"]
    # 参数字典["数据增强方法"] = ['none']
    data_set = SPO2Dataset(参数字典['训练数据文件夹'],D=参数字典['D'],step=参数字典['步长'],参数 = 参数字典,不加载数据=不加载数据,加载数据=加载数据)

    #验证集占比
    val_precent = 参数字典['验证集占比']
    set2, val_set, test_set = torch.utils.data.random_split(data_set, [int(0.1*data_set.__len__()),int(val_precent*data_set.__len__()), data_set.__len__()-int(val_precent*data_set.__len__())-int(0.1*data_set.__len__())])

    valloader = DataLoader(val_set, batch_size=参数字典['batch_size'], shuffle=True)
    
    # 使用ConcatDataset将两个数据集合并为一个
    combined_dataset = ConcatDataset([train_set, set2])
    if 参数字典['重采样']:
    # 创建采样器，根据权重进行采样
        train_sampler = WeightedRandomSampler(weights=数据加载器.sample_weights, num_samples=len(数据加载器.sample_weights), replacement=True)
        trainloader = DataLoader(combined_dataset, batch_size=参数字典['batch_size'], shuffle=False,sampler=train_sampler)
    else:
        trainloader = DataLoader(combined_dataset, batch_size=参数字典['batch_size'], shuffle=True)

    testloader = DataLoader(test_set, batch_size=参数字典['batch_size'], shuffle=False)


    dataloaders = {
        'train': trainloader,
        'val': valloader,
        'test': testloader
    }
    et,gt_SPO2 = trainloader.dataset.__getitem__(0)
    len2 = trainloader.dataset.__len__()
    print(f"一个样本的输入形状是：{et.shape}")
    print(f"一个样本的标签形状是：{gt_SPO2.shape}")
    print(f"600帧为一个样本，共计有：{len2}个")
    
    print("统计数据分布")
    # 遍历测试集
    print("遍历测试集")
    plot_distribution(dataloaders['test'], 'Test Set gt_SPO2 Distribution',count)

    # 遍历测试集
    print("遍历验证集")
    plot_distribution(dataloaders['val'], 'Val Set gt_SPO2 Distribution',count)

    # 遍历训练集
    print("\n遍历训练集")
    plot_distribution(dataloaders['train'], 'Training Set gt_SPO2 Distribution',count)
    print("\n")

    # 显示图像
    # fig,ax = plt.subplots(2,3,figsize=(15,8))
    # for i in range(3):
    #     ax[i//3,i%3].plot(et[i,:].cpu().detach().numpy() ,label='rPPG')
    # ax[1,1].plot(gt_SPO2[0,:],label='gt_SPO2')

    # fig.suptitle(f'dataloader display 血氧：{gt_SPO2.mean()}')
    # plt.show()

    #从已经训练的参数开始训练
    if 参数字典['检查点'] != None:
        print("加载检查点")
        state_dict = torch.load(参数字典['检查点'])
        model.load_state_dict(state_dict) 
    # discriminator_state_dict = torch.load('./checkpoints/discriminator_244.pt')

    # criterion_G = Gloss() 
    # criterion_D = Dloss()
    # adversarial_loss = torch.nn.BCELoss()
    # 初始化L1损失函数  平均绝对误差
    print("初始化损失函数")
    mae_loss = nn.L1Loss()
    HYX_loss = Heyuanxia_Loss(参数字典)

    print("初始化优化器")
    optimizer = optim.Adam(model.parameters(), lr=参数字典['学习率'])

    print("初始化学习率调度器")
    num_epochs = 参数字典['epochs']
    multiplier = 参数字典['渐热学习放大倍率'] #渐热学习学习率变大几倍

    # 创建 ReduceLROnPlateau 调度器
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # 使用余弦退火学习率调度器，其学习率在每个epoch中从初始学习率线性增加到最大学习率，然后减少到最小学习率
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1)
    # 使用渐热学习率调度器，在初始几个epoch中逐渐增加学习率，然后切换到余弦退火调度器
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=multiplier, warm_epoch=num_epochs // 10 +1, after_scheduler=cosineScheduler)

    train_loss_batch_history = []
    val_loss_batch_history = []
    train_loss_history = []
    val_loss_history = []
    train_loss_mae_history = []
    val_loss_mae_history = []

    min_train = 100000 #记录训练时损失最小是多少，用来判断是否是最小损失
    min_Val = 100000   #记录验证时损失最小是多少，用来判断是否是最小损失

    min_train_mae = 100 #记录训练时mae损失最小是多少，用来判断是否是最小损失
    min_val_mae = 100   #记录验证时mae损失最小是多少，用来判断是否是最小损失

    # 测试集train_best参数损失最低 = [1000 for _ in range(len(参数字典['测试数据名称']))]
    # 测试集train_best参数损失最低索引 = [0 for _ in range(len(参数字典['测试数据名称']))]
    # 测试集train_maeloss_best参数损失最低 = [1000 for _ in range(len(参数字典['测试数据名称']))]
    # 测试集train_maeloss_best参数损失最低索引 = [0 for _ in range(len(参数字典['测试数据名称']))]
    # 测试集test_best参数损失最低 = [1000 for _ in range(len(参数字典['测试数据名称']))]
    # 测试集test_best参数损失最低索引 = [0 for _ in range(len(参数字典['测试数据名称']))]
    # 测试集test_maeloss_best参数损失最低 = [1000 for _ in range(len(参数字典['测试数据名称']))]
    # 测试集test_maeloss_best参数损失最低索引 = [0 for _ in range(len(参数字典['测试数据名称']))]

    验证集综合最优损失epoch = 0
    验证集MAE最优损失epoch = 0
    训练集综合最优损失epoch = 0
    训练集MAE最优损失epoch = 0

    全局最佳模型loss = 1000
    全局最佳模型名称 = '' 


    for epoch in tqdm(range(num_epochs)):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        记录log('\nEpoch {}/{}'.format(epoch, num_epochs - 1),count)
        print('-' * 10)
        记录log('-' * 10,count)

        # Each epoch has a training and validation phase
        phases = ['train', 'val','test']
        for phase in phases:
            running_loss = 0.0
            running_loss_mae = 0.0
            if phase == 'train':
                model.train()  # 设置模型进入训练模式，激活bn层和dropout层
            else:
                model.eval()  # 设置模型到验证层
            # Iterate over data.
            with tqdm(dataloaders[phase], dynamic_ncols=True) as tqdmDataLoader:

                # for inputs, targets in tqdmDataLoader:
                #     inputs_2dims = inputs.view(-1,600).to(device)
                #     inputs = GenerPPG.get_gPPG(inputs_2dims).view(-1,3,600).to(device)
                for inputs, targets in tqdmDataLoader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # 前向传播 forward
                    # 记得记录相关数据在每个epoch，batch
                    with torch.set_grad_enabled(phase == 'train'):
                        # -----------------------
                        # 训练SPO2CNN
                        # -----------------------
                        # 梯度归零
                        optimizer.zero_grad()
                        # print(model(inputs).shape)
                        # print(targets.shape)
                        output = model(inputs).squeeze(1)

                        # plt.figure(figsize=(10, 8))  # 宽度为10英寸，高度为8英寸
                        # plt.plot(output[0,:,:].cpu().detach().numpy(),linestyle = '-.',label="预测")
                        # plt.plot(targets[0,:,:].cpu().detach().numpy(),label = "标签")
                        # plt.legend()
                        # plt.grid(True)
                        # plt.show()
                        # print(output.shape)
                        # print(targets.shape)
                        
                        loss,loss_mae = HYX_loss(output,targets)
                        # print(loss)
                        # loss_mae = mae_loss(output,targets)
                        # loss = loss_mae
                        # print(f"损失：{loss_mae}")

                        
                        # backward + optimize only if in training phase
                        # 反向传播+只有优化器工作
                        if phase == 'train':
                            # loss = 计算加权误差损失(output,targets,参数字典['加权误差方法'])
                            train_loss_batch_history.append(loss.item())
                            loss.backward()
                            optimizer.step()
        #                     # 在这里显示梯度  
        #                     for name, param in generator.named_parameters():  
        #                         if param.grad is not None:  
        #                             print("显示梯度：")
        #                             print(name, param.grad) 
        #                     #print(torch.sum(generator.conv1.weight.grad))
                        else:
                            val_loss_batch_history.append(loss.item())
                        # 使用 tqdm 更新进度条，显示当前epoch、损失、图像形状和当前学习率
                        tqdmDataLoader.set_postfix(ordered_dict={
                            "epoch": epoch,
                            "loss: ": loss.item(),
                            "loss_mae: ": loss_mae.item(),
                            "batch shape: ": inputs.shape,
                            "LR": optimizer.state_dict()['param_groups'][0]["lr"],
                            "val_minloss(epoch)": f"{min_Val:.4f}({验证集综合最优损失epoch})",
                            "val_minmaeloss(epoch)": f"{min_val_mae:.4f}({验证集MAE最优损失epoch})",
                        })    
                    # statistics
                    running_loss += loss.item()
                    running_loss_mae += loss_mae.item()
    #         print(running_loss_G)
    #         print(len(dataloaders[phase]))
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_loss_mae = running_loss_mae / len(dataloaders[phase])
            print(f'\n{phase}: 平均绝对误差(非加权)：{epoch_loss}; 实际训练损失：{epoch_loss_mae}')
            记录log(f'{phase}: 平均绝对误差(非加权)：{epoch_loss}\n实际训练损失：{epoch_loss_mae}',count)

            if phase == 'val':
                if epoch_loss < min_Val or epoch == 0:
                    min_Val = epoch_loss
                    # torch.save(model.state_dict(), f'./checkpoints/exp{count}/best_model/SPO2_test_best.pt')
                    # torch.save(model.state_dict(), f'./checkpoints/exp{count}/best_model/SPO2_test_best_{epoch}.pt')
                    print(f'验证集综合loss最优更新：epoch = {epoch}')
                    记录log(f'验证集综合loss最优更新：epoch = {epoch}',count)
                    验证集综合最优损失epoch = epoch
                val_loss_history.append(epoch_loss)
                if epoch_loss_mae < min_val_mae or epoch == 0:
                    min_val_mae = epoch_loss_mae
                    torch.save(model.state_dict(), f'./checkpoints/exp{count}/best_model/SPO2_test_maeloss_best.pt')
                    # torch.save(model.state_dict(), f'./checkpoints/exp{count}/best_model/SPO2_test_maeloss_best_{epoch}.pt')
                    print(f'验证集MAEloss最优更新：epoch = {epoch}')
                    记录log(f'验证集MAEloss最优更新：epoch = {epoch}',count)
                    验证集MAE最优损失epoch = epoch
                    global 测试集MAE
                    测试集MAE = 验证集MAE最优损失epoch
                val_loss_mae_history.append(epoch_loss_mae)
                # scheduler.step(epoch_loss)

            elif phase == 'train':
                if epoch_loss < min_train or epoch == 0:
                    min_train = epoch_loss
                    # torch.save(model.state_dict(), f'./checkpoints/exp{count}/best_model/SPO2_train_best.pt')
                    # torch.save(model.state_dict(), f'./checkpoints/exp{count}/best_model/SPO2_train_best_{epoch}.pt')
                    print(f'训练集综合loss最优更新：epoch = {epoch}')
                    记录log(f'训练集综合loss最优更新：epoch = {epoch}',count)
                    训练集综合最优损失epoch = epoch
                train_loss_history.append(epoch_loss)
                if epoch_loss_mae < min_train_mae or epoch == 0:
                    min_train_mae = epoch_loss_mae
                    # torch.save(model.state_dict(), f'./checkpoints/exp{count}/best_model/SPO2_train_maeloss_best.pt')
                    # torch.save(model.state_dict(), f'./checkpoints/exp{count}/best_model/SPO2_train_maeloss_best_{epoch}.pt')
                    print(f'训练集MAEloss最优更新：epoch = {epoch}')
                    记录log(f'训练集MAEloss最优更新：epoch = {epoch}',count)
                    训练集MAE最优损失epoch = epoch
                train_loss_mae_history.append(epoch_loss_mae)
                warmUpScheduler.step()
            else:
                print(f"测试集损失={epoch_loss_mae}")
            total_params = sum(p.numel() for p in model.parameters())
            print(f"总参数量: {total_params}")  # 输出：109,258

            # 仅统计可训练参数
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"总可训练参数量: {trainable_params}")
        
        if (epoch+1) % 5 == 0:
            # torch.save(model.state_dict(), f'./checkpoints/exp{count}/SPO2_{epoch}.pt')        
            torch.save(model.state_dict(), f'./checkpoints/exp{count}/SPO2_last.pt')
            # 生成画布
            # 关闭所有图形窗口
            plt.close('all')
            # 创建一个画布，并设置大小和分辨率
            fig, axs = plt.subplots(2, 1, figsize=(20, 32), dpi=100)  # 4行1列的子图

            # 训练epoch损失
            axs[0].plot(train_loss_history[3:], linestyle='-', color='r', label="训练epoch损失")
            最小值索引 = 0
            最小值 = 100000
            for i,loss in enumerate(train_loss_history[3:]):
                if i <= 0:
                    continue
                if loss < 最小值:
                    最小值索引 = i
                    最小值 = loss
            min_idx = 最小值索引 + 3  # 加1是因为我们从第二个元素开始画图
            min_val = train_loss_history[min_idx]
            axs[0].annotate(f'Epoch_{min_idx}_Loss: {min_val:.4f}', xy=(最小值索引, min_val), xytext=(min_idx, min_val*1.02),
                            arrowprops=dict(facecolor='red', shrink=0.05), ha='center',fontsize=14)
            
            axs[0].plot(val_loss_history[3:], linestyle='-.', color='r', label="验证epoch损失")
            最小值索引 = 0
            最小值 = 100000
            for i,loss in enumerate(val_loss_history[3:]):
                if i <= 0:
                    continue
                if loss < 最小值:
                    最小值索引 = i
                    最小值 = loss
            min_idx = 最小值索引 + 3
            min_val = val_loss_history[min_idx]
            axs[0].annotate(f'Epoch_{min_idx}_Loss: {min_val:.4f}', xy=(最小值索引, min_val), xytext=(min_idx, min_val*1.02),
                            arrowprops=dict(facecolor='red', shrink=0.05), ha='center',fontsize=14)
            axs[0].set_title("综合损失 分别在训练集验证集上 随着epoch变化的曲线")
            axs[0].legend()
            axs[0].grid(True)

            # 训练epoch_MAE损失
            axs[1].plot(train_loss_mae_history[3:], linestyle='-', color='b', label="训练epoch_MAE损失")
            最小值索引 = 0
            最小值 = 100000
            for i,loss in enumerate(train_loss_mae_history[3:]):
                if i <= 0:
                    continue
                if loss < 最小值:
                    最小值索引 = i
                    最小值 = loss
            min_idx = 最小值索引 + 3
            min_val = train_loss_mae_history[min_idx]
            axs[1].annotate(f'训练集第{min_idx}MAELoss最低: {min_val:.4f}', xy=(最小值索引, min_val), xytext=(min_idx, min_val*1.02),
                            arrowprops=dict(facecolor='blue', shrink=0.05), ha='center',fontsize=14)
            
            axs[1].plot(val_loss_mae_history[3:], linestyle='-.', color='b', label="验证epoch_MAE损失")
            最小值索引 = 0
            最小值 = 100000
            for i,loss in enumerate(val_loss_mae_history[3:]):
                if i <= 0:
                    continue
                if loss < 最小值:
                    最小值索引 = i
                    最小值 = loss
            min_idx = 最小值索引 + 3
            min_val = val_loss_mae_history[min_idx]
            axs[1].annotate(f'验证集第{min_idx}MAELoss最低: {min_val:.4f}', xy=(最小值索引, min_val), xytext=(min_idx, min_val*1.02),
                            arrowprops=dict(facecolor='blue', shrink=0.05), ha='center',fontsize=14)
            axs[1].set_title("MAE损失 分别在训练集验证集上 随着epoch变化的曲线")
            axs[1].legend()
            axs[1].grid(True)

            # 调整子图间距
            plt.tight_layout()

            # 保存图片到指定路径
            plt.savefig(f'./checkpoints/exp{count}/{epoch}_训练损失曲线.png')  # 替换为你的文件名和路径
            # plt.show()
        
        if 参数字典['test周期'] != None:
            if (epoch+1) % 参数字典['test周期'] == 0 and ((epoch - 验证集MAE最优损失epoch) < 参数字典['test周期']):
                验证集上最佳模型预测_list = []
                验证集上最佳模型标签_list = []
                验证测试集 = []
                验证测试集.extend(参数字典["测试数据名称"])
                验证测试集.extend(参数字典["验证数据名称"])
                
                #***跨数据集使用***
                # 验证测试文件夹 = f"{参数字典['训练数据文件夹']}/不进行归一化_{验证测试集[0]}_一阶/"
                # 测试数据集文件名 = [
                #     name for name in os.listdir(验证测试文件夹)
                #     if os.path.isdir(os.path.join(验证测试文件夹, name))
                #     # 可选：排除隐藏文件夹（以 . 开头）
                #     and not name.startswith('.')
                # ]
                # # 存储所有目标文件的绝对路径
                # target_files = []

                # # 遍历主文件夹下的所有子文件夹
                # for foldername, subfolders, filenames in os.walk(验证测试文件夹):
                #     # 检查当前文件夹中是否存在目标文件
                #     if "预处理过的rPPG.csv" in filenames:
                #         # 拼接绝对路径
                #         # file_path = os.path.join(foldername, "预处理过的rPPG.csv")
                #         # 将绝对路径添加到列表
                #         target_files.append(os.path.abspath(foldername))
                # 验证测试集 = target_files
                # for path,一个文件夹名 in zip(target_files,测试数据集文件名):
                #     print(path,一个文件夹名)
                #*****************

                for i,一个测试数据名称 in enumerate(验证测试集):
                    # pre,tar,loss_mae = test_SPO2CNN.test(f'./checkpoints/exp{count}/best_model/SPO2_test_maeloss_best.pt',f"{参数字典['训练数据文件夹']}/{一个测试数据名称}/none_0",f'./checkpoints/exp{count}/{epoch}_valbest_in_testset{一个测试数据名称}.png',参数字典)
                    pre,tar,loss_mae = test_SPO2CNN.test(f'./checkpoints/exp{count}/best_model/SPO2_test_maeloss_best.pt',f"{参数字典['训练数据文件夹']}/{一个测试数据名称}/none_0",f'./checkpoints/exp{count}/{epoch}_testbest_in_testset{一个测试数据名称}.png',参数字典)
                    # pre,tar,loss_mae = test_SPO2CNN.test(f'./checkpoints/exp{count}/best_model/SPO2_test_maeloss_best.pt',一个测试数据名称,f'./checkpoints/exp{count}/{epoch}_testbest_in_testset{测试数据集文件名[i]}.png',参数字典)
                    记录log(f'{一个测试数据名称} 在 SPO2_test_maeloss_best.pt上的MAE ： {loss_mae}',count)
                    print(f'{一个测试数据名称} 在 SPO2_test_maeloss_best.pt上的MAE ： {loss_mae}')
            
                    验证集上最佳模型预测_list.extend(pre)
                    验证集上最佳模型标签_list.extend(tar)
                验证集上最佳模型名称 = f'SPO2_test_maeloss_best_{验证集MAE最优损失epoch}.pt'
                    
    return 验证集上最佳模型预测_list,验证集上最佳模型标签_list,验证集上最佳模型名称,min_Val
    # return 测试集MAE