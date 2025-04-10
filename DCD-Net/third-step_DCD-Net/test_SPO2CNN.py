import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from pylab import mpl
from 模型 import SpO2Net
from 数据加载器 import SPO2Dataset
import os
import pandas as pd

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# plt.figure(figsize=(2,3),dpi=100)
# 设置正确显示符号
mpl.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

class SPO2Dataset_1(Dataset):
    """Dataset of rPPGs"""

    def __init__(self, csv_path, D = 600,step = 60,val_num = '01-04',数据增强方法=['none','画面扭曲','运动模糊','缩放画面','旋转画面','水平翻转']):
        """
        Args:
            csv_path (string): 文件夹地址：指向存放着预处理完的不同人的data和gt
            D = 600：表示一段信号的长度
            step = 60：表示一段信号与下一段信号之间的间隔时间
            val_num = -1:表示K折验证中作为验证集的那个人的编号，-1表示不进行K折交叉验证
        """
        
        # 初始化一个空的DataFrame  
        self.all_data = []
        # 初始化一个空的list  存放视频帧用作提示词
        self.all_image = []
        # 定义一个K折跳过的目录
        skip_dir = val_num
        # 遍历当前目录及其所有子目录  
        for root, dirs, files in os.walk(csv_path):  
            # 检查当前目录名，如果是要跳过的目录，则从dirs中移除  
            if skip_dir in dirs:  
                dirs.remove(skip_dir) 
            for file in files:  
                if file == '预处理过的rPPG.csv':  
                    # 构造完整的文件路径  
                    file_path = os.path.join(root, file) 
                    # print(file_path)
                    if any(method in file_path for method in 数据增强方法):
                        # 读取CSV文件  
                        df = pd.read_csv(file_path,encoding='utf-8')
                        # 将读取的DataFrame添加到all_data中，使用pd.concat()合并  行合并
                        数据_array = np.array(df)
                        df = None
                        while 数据_array.shape[0] >= 3600:
                            print(f'{file_path}读取成功！')
                            self.all_data.append(数据_array[:3600,:])
                            数据_array = 数据_array[3600:,:]
                
        self.D = D
        self.step = step

        # 一个文件夹的样本个数
        self.all_data = np.array(self.all_data)
        self.all_image = np.array(self.all_image)
        self.num_samples = ((self.all_data.shape[1]-self.D)//self.step+1)

    # 整个数据集的样本个数
    def __len__(self):
        return self.all_data.shape[0]*self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # pp_index表示人员标号，从0开始
        pp_index = idx // self.num_samples
        # pn_index表示这个人的第几个样本，从0开始
        pn_index = idx % self.num_samples
        
        et_new = self.all_data[pp_index,pn_index*self.step:pn_index*self.step+self.D,0:36].T 
        
        gt_SPO2 = self.all_data[pp_index,pn_index*self.step:pn_index*self.step+self.D,36]
        gt_SPO2_reshape = gt_SPO2.reshape(1,-1)
        et_new = normalize(et_new)

        # cast to tensor
        et_tensor = torch.from_numpy(et_new).type(torch.FloatTensor)
        gt_SPO2_reshape = torch.from_numpy(gt_SPO2_reshape).type(torch.FloatTensor)

        # return et_tensor,gt_SPO2_reshape
        return et_tensor,gt_SPO2_reshape

class HYXSPO2CNNtest:  
    def __init__(self, state_dict_path = './checkpoints/exp0/SPO2_last.pt',device = device):  
        """    
        state_dict_path: 模型参数路径
        device: 设备
        """  
        self.model = SpO2Net()
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(state_dict_path))
        self.model.eval()
        self.device = device
        self.spo2 = None

         # 新增可视化参数
        self.visualization_dir = "./attention_visualizations"
        os.makedirs(self.visualization_dir, exist_ok=True)
        print(f"可视化文件将保存到：{os.path.abspath(self.visualization_dir)}")

    def get_spo2(self,et):  
        """  
        输入：et:输入是形状（36,600）的rPPG信号
        返回: 模型计算得到的血氧值 
        """  
        inputs = et.to(self.device)
        
        if inputs.shape[1] == 24:
            self.spo2 = self.model(inputs).cpu().detach().numpy()
        elif inputs.shape[0] == 24:
            self.spo2 = self.model(inputs.unsqueeze(0)).cpu().detach().numpy() 
        return self.spo2[0,0]
    def get_long_spo2(self,data_path,图片保存路径,参数字典,visualize_flag=True):
        ds = SPO2Dataset(data_path,D=参数字典['D'],step=参数字典['步长'],参数 = 参数字典)
        tloader = DataLoader(ds, batch_size=3, shuffle=False)

        # 新增：可视化标志
        # visualize_flag = True  # 控制是否可视化

        pre = []
        tar = []
        import time
        start_time = time.time()
        if tloader.dataset.__len__() >=1:
            for i in range(tloader.dataset.__len__()):
                et,gt_SPO2 = tloader.dataset.__getitem__(i)
                # 新增：对第一个样本进行可视化
                if i == 0 and visualize_flag:
                    print("\n开始可视化...")
                    print("输出模型：")
                    print(self.model)
                    print(f"输入形状：{et.shape}")
                    self.visualize_attention(et)
                    print("可视化完成\n")

                if gt_SPO2 != None:
                    targets = gt_SPO2.to(device)
                    print(np.mean(targets.cpu().detach().numpy()),end='  ')
                    # tar.append(np.mean(targets.cpu().detach().numpy()[0]))
                    tar.append(np.mean(targets.cpu().detach().numpy()))
                SPO2_value = self.get_spo2(et)
                # SPO2_value = 96.5-(SPO2_value-96.5)
                print(SPO2_value)
                # time.sleep(0.1)
                SPO2_value = min(100,SPO2_value)
                SPO2_value = max(85,SPO2_value)
                # print(model(inputs.unsqueeze(0)).cpu().detach().numpy().shape)
                pre.append(SPO2_value)
            end_time = time.time()
            elapsed = (end_time - start_time)/tloader.dataset.__len__()
            print(f"单样本耗时：{elapsed*1000:.2f}毫秒")
            pre = np.array(pre)
            tar = np.array(tar)
            # print(np.abs(tar-pre))
            print(f"平均绝对误差：{np.mean(np.abs(tar-pre))}")
            # 设置图像的尺寸
            # 关闭所有图形窗口
            plt.close('all')
            plt.figure(figsize=(10, 8))  # 宽度为10英寸，高度为8英寸
            plt.title(f"MAE：{np.mean(np.abs(tar-pre))}")
            plt.plot(pre,linestyle = '-.',label="Prediction")
            if gt_SPO2 != None:
                plt.plot(tar,label = "Label")
            plt.legend()
            plt.grid(True)
            plt.savefig(图片保存路径)  # 指定文件名和路径
            # plt.show()
            
            return pre,tar,np.mean(np.abs(tar-pre))
        else:
            return [100],[100],0
        
    def visualize_attention(self, inputs, sample_idx=0):
        """专用CBAM可视化"""
        # 输入预处理
        inputs = inputs.to(self.device)
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)  # 添加批次维度

        # ================== 1. 创建通道可视化目录 ==================
        channel_dir = os.path.join(self.visualization_dir, "input_channels")
        os.makedirs(channel_dir, exist_ok=True)
        
        # ================== 2. 独立通道折线图生成 ==================
        def plot_single_channel(data, channel_idx, save_dir):
            """绘制单个通道的折线图"""
            plt.figure(figsize=(24, 8),dpi=300)
            
            # 绘制曲线
            plt.plot(data, 
                    color='#1f77b4',  # matplotlib默认蓝色
                    linewidth=1.5,
                    alpha=0.8)
            
            # 格式设置
            plt.title(f"Channel {channel_idx} Signal\n(Mean: {np.mean(data):.2f}, Std: {np.std(data):.2f})")
            plt.xlabel("Time Steps (0-600)")
            plt.ylabel("Normalized Value")
            plt.grid(True, alpha=0.3)
            plt.xlim(0, len(data))
            
            # 保存文件
            save_path = os.path.join(save_dir, f"channel_{channel_idx:02d}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"通道 {channel_idx} 已保存至：{save_path}")

        # 处理数据并生成图像
        input_data = inputs[sample_idx].cpu().numpy()  # [channels, time]
        num_channels = min(12, input_data.shape[0])    # 最多处理12个通道
        
        for ch_idx in range(num_channels):
            plot_single_channel(input_data[ch_idx], ch_idx, channel_dir)
            
        # 注册钩子
        activations = {}
        def hook_fn(module, input, output, layer_name):
            activations[layer_name] = output.detach().cpu().numpy()
        
        # 获取所有CBAM模块
        cbam_layers = {
            "CBAM_block_128": self.model.rescnn_fb.CBAM_block_128,
            "CBAM_block_64": self.model.rescnn_fb.CBAM_block_64
        }
        
        # 注册钩子
        handles = []
        for name, layer in cbam_layers.items():
            handle = layer.register_forward_hook(
                lambda m, i, o, n=name: hook_fn(m, i, o, n)
            )
            handles.append(handle)
            print(f"已注册 {name} 的钩子")  # 调试输出
        
        # 执行前向传播
        with torch.no_grad():
            print("正在进行前向传播...")
            _ = self.model(inputs)
            print("前向传播完成")
        
        # 移除钩子
        for handle in handles:
            handle.remove()
        
        # 保存可视化结果
        for layer_name, att_map in activations.items():
            print(att_map.shape)
            bs,num_heads,seq_len = att_map.shape
            for i in range(num_heads):
                plt.figure(figsize=(20, 12),dpi=300)
                batch_data = np.zeros((15,150))
                for j in range(15):
                    batch_data[j,:] = att_map[0,i:i+1]
                
                plt.imshow(batch_data, cmap='viridis')  # 显示第一个样本
                plt.title(layer_name)
                plt.colorbar()
                save_path = os.path.join(self.visualization_dir, f"{layer_name}_{i}.png")
                plt.savefig(save_path)
                plt.close()
                print(f"已保存 {save_path}")

def test(检查点,test数据,图片保存路径,参数字典,注意力可视化=False):
    print("\n测试验证")
    print(f"测试数据 = {test数据}")
    # 实例化HYXtest
    HYX = HYXSPO2CNNtest(state_dict_path = 检查点,device = device)

    # 数据输入文件夹路径
    data_path = test数据
    pre,tar,loss_mae = HYX.get_long_spo2(data_path,图片保存路径,参数字典,visualize_flag=注意力可视化)

    # 将数据转换为DataFrame
    data = pd.DataFrame({
        'True Labels': tar,
        'Predicted Labels': pre
    })

    # 保存为CSV文件
    data.to_csv(f'./checkpoints/Scatter_Plot.csv', index=False)

    print("数据预测结果已成功保存到 validation_results.csv 文件中！")

    return pre,tar,loss_mae

if __name__ == "__main__":
    参数字典 = {
    "D" : 600, #数据长度
    '步长' : 60,
    "数据增强方法": ['真实','none'],
    "波峰对齐": False,
    "噪声均值": -1
    }
    测试模型参数 = r"F:\Face_rPPG_Get\集成2\3-2、训练_将BERT作为主干网_72通道\checkpoints\exp30、完全体\动作第2折_验证数据xx_03_测试数据xx_02\第1次\best_model\SPO2_test_maeloss_best.pt"
    测试数据文件夹 = "../0-1、预处理后的数据/30实例测试/self/10004/"
    注意力可视化=False
    test(测试模型参数,测试数据文件夹,'./checkpoints/test.png',参数字典,注意力可视化=注意力可视化)