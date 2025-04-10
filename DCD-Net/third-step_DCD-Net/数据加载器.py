import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt

label_counts = None
labels = None
sample_weights = None


# 定义一个函数来添加高斯噪声  
def 加噪声(data, mean=0, std_dev=0.01):  
    noise = np.random.normal(mean, std_dev, data.shape)  
    noisy_data = data + noise  
    return noisy_data  

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

def maxim_peaks_above_min_height(pn_locs, pn_x, n_size, n_min_height):
    """
    Find all peaks above a minimum height.
    """
    pn_locs.clear()
    i = 1
    n_npks = 0
    
    while i < n_size - 1:
        if pn_x[i] > n_min_height and pn_x[i] > pn_x[i - 1]:  # find left edge of potential peaks
            n_width = 1
            while i + n_width < n_size and pn_x[i] == pn_x[i + n_width]:  # find flat peaks
                n_width += 1
            if i+n_width >= n_size:
                n_width = n_size - 1 - i
            if pn_x[i] > pn_x[i + n_width] and n_npks < 100:  # find right edge of peaks
                pn_locs.append(i)
                n_npks += 1
                # for flat peaks, peak location is left edge
                i += n_width + 1
            else:
                i += n_width
        else:
            i += 1
    # print(pn_locs)
    return pn_locs,n_npks

def maxim_remove_close_peaks(pn_locs, pn_x, n_min_distance):
    """
    Remove peaks separated by less than a minimum distance.
    """
    n_npks = len(pn_locs)
    if n_npks == 0:
        return [],n_npks
    
    # Order peaks from large to small
    idx_sorted = sorted(range(n_npks), key=lambda i: pn_locs[i], reverse=False)
    
    new_pn_locs = [pn_locs[idx] for idx in idx_sorted]
    
    i = -1
    n_new_npks = 0
    for j, idx in enumerate(idx_sorted):
        n_dist = new_pn_locs[j] - (-1 if i == -1 else new_pn_locs[i])
        if n_dist > n_min_distance or n_dist < -n_min_distance:
            # print(f"{new_pn_locs[j]} - {new_pn_locs[i]} = {n_dist}")
            i = j
            pn_locs[n_new_npks] = new_pn_locs[j]
            n_new_npks += 1
    pn_locs = pn_locs[:n_new_npks]
    
    return pn_locs,n_new_npks

def maxim_find_peaks(pn_x, n_size, n_min_height, n_min_distance, n_max_num):
    """
    Find at most MAX_NUM peaks above MIN_HEIGHT separated by at least MIN_DISTANCE.
    """
    pn_locs = []
    pn_locs, n_npks = maxim_peaks_above_min_height(pn_locs, pn_x, n_size, n_min_height)
    
    pn_locs, n_npks = maxim_remove_close_peaks(pn_locs, pn_x, n_min_distance)
    
    if n_npks > n_max_num:
        pn_locs = pn_locs[:n_max_num]
    
    return pn_locs, n_npks

def 列表元素是否出现在字符串中吗(元素列表, 字符串):
    return any(element in 字符串 for element in 元素列表)
    
class SPO2Dataset(Dataset):
    """Dataset of rPPGs"""

    def __init__(self, csv_path, D,step ,参数 = None, 不加载数据 = None, 加载数据 = None, 重采样 = False):
        """
        Args:
            csv_path (string): 文件夹地址：指向存放着预处理完的不同人的data和gt
            D = 600：表示一段信号的长度
            step = 60：表示一段信号与下一段信号之间的间隔时间
            val_num = -1:表示K折验证中作为验证集的那个人的编号，-1表示不进行K折交叉验证
        """
        
        D = 参数['D']
        step = 参数['步长']
        数据增强方法 = 参数['数据增强方法']
        
        # 初始化一个空的DataFrame  
        self.all_data = []
        # 初始化一个空的list  存放视频帧用作提示词
        self.all_image = []
        # 定义一个K折跳过的目录
        skip_dirs = 不加载数据
        # 遍历当前目录及其所有子目录  
        print(f'所有数据存放在{csv_path}')
        print(f'不加载数据集：{skip_dirs}')
        for root, dirs, files in os.walk(csv_path):  
            # print(root)
            # 检查当前目录名，如果是要跳过的目录，则从dirs中移除  
            if 不加载数据 != None:
                for sk in skip_dirs:
                    # 构造完整的文件路径  
                    # dir_path = os.path.join(root, dirs) 
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        if sk in dir_path:
                            # print(dir_path)
                            dirs.remove(dir)  
            for file in files:  
                # file_path = os.path.join(root, file) 
                # print(file_path)
                if file == '预处理过的rPPG.csv':  
                    # 构造完整的文件路径  
                    file_path = os.path.join(root, file) 
                    # print(file_path)
                    
                    if any(method in file_path for method in 数据增强方法):
                        # 筛选我们想要的数据
                        if 加载数据 != None:
                            ctn = 0
                            for data in 加载数据:
                                # print(data,file_path)
                                if data in file_path:
                                    ctn = 1
                                    continue
                            if ctn != 1:
                                continue
                        # 读取CSV文件  
                        df = pd.read_csv(file_path,encoding='utf-8')
                        # 将读取的DataFrame添加到all_data中，使用pd.concat()合并  行合并
                        数据_array = np.array(df)
                        df = None

                        # 判断数据长度是否超过600
                        if 数据_array.shape[0] < D:
                            pass
                            # raise Exception("Error: 数据长度小于600")
                        if np.any((数据_array[:,24] < 75) | (数据_array[:,24] > 100)):
                            continue
                            raise Exception(f"Error: {file_path} 数据标签有问题，血氧值过低（小于75%）")
                        print(f"{file_path} {数据_array.shape[0]}")

                        while 数据_array.shape[0] >= D:
                            self.all_data.append(数据_array[:D,:])
                            数据_array = 数据_array[step:,:]
                
        self.D = D
        self.step = step
        self.参数 = 参数

        # 一个文件夹的样本个数
        self.all_data = np.array(self.all_data)
        print(f"(样本数,长度,通道+标签) = {self.all_data.shape}")
        if self.all_data.shape[0]>= 1:


            print("标签四舍五入")
            self.all_data[:,:,24] = np.around(self.all_data[:,:,24])

            # print("标签均值滤波")
            # 输出文件夹名 = 参数['输出文件夹名']
            # 软化标签长度 = 参数['整个数据集范围内的软化标签长度']
            # path = f"./checkpoints/exp{输出文件夹名}/均值滤波结果显示"
            # # 如果目录不存在，则创建它  
            # if not os.path.exists(path):  
            #     os.makedirs(path)
            # if 参数['整个数据集范围内的软化标签长度'] != None:
            #     for i in range(self.all_data.shape[0]):
            #         onedata = self.all_data[i,:,72].copy()
            #         print(f'第{i+1}个数据的均值滤波 输入形状={onedata.shape} 均值滤波窗口大小 = {软化标签长度}',end='\r')
            #         # 创建卷积核
            #         kernel = np.ones(参数['整个数据集范围内的软化标签长度']) / 参数['整个数据集范围内的软化标签长度']
            #         # 对每一列应用均值滤波
            #         数据_array = np.convolve(onedata, kernel, mode='same')
            #         self.all_data[i,:,72] = 数据_array

            #         # 关闭所有图形窗口
            #         plt.close('all')
            #         plt.figure(figsize=(10, 8))  # 宽度为10英寸，高度为8英寸
            #         plt.title(f"{i}均值滤波前后标签变化")
            #         plt.plot(onedata[软化标签长度//2:-软化标签长度//2],linestyle = '-.',label="均值滤波前")
            #         plt.plot(self.all_data[i,:,72][软化标签长度//2:-软化标签长度//2],label = "均值滤波后")
            #         plt.legend()
            #         plt.grid(True)
            #         plt.savefig(f'./checkpoints/exp{输出文件夹名}/均值滤波结果显示/第{i}个数据 均值滤波前后标签变化.png')  # 指定文件名和路径
            #         # plt.show()
            #         # print(onedata.shape[3])
            # self.all_data = self.all_data[:,软化标签长度//2:-软化标签长度//2,:]
            # print(f"\n均值滤波之后的形状 = {self.all_data.shape}")

            # self.all_image = np.array(self.all_image)
            self.num_samples = 1

            if 重采样:
                global label_counts
                global labels
                labels = []
                for i in range(self.all_data.shape[0]):# i表示第几个人
                    for j in range(self.num_samples):# j表示某人的第几个样本
                        labels.append(np.around(np.mean(self.all_data[i,j*self.step:j*self.step+self.D,24])))
                
                labels = torch.tensor(labels).to(torch.int64)
                label_counts = torch.bincount(labels)
                print(f"\n所有样本的标签个数：{labels.shape[0]}")
                # 计算每个类别的权重，这里使用类别的倒数作为权重
                class_weights = len(labels) / label_counts.float()
                # print(f"权重：{class_weights}")
                # print(class_weights.shape)
                class_weights = []
                if 参数["非平衡重采样"]:
                    for count in label_counts:
                        if count / len(labels) < 参数["过采样阈值"]:
                            class_weight = len(labels) * 参数["过采样阈值"] / (count)
                        elif count / len(labels) > 参数["欠采样阈值"]:
                            class_weight = len(labels) * 参数["欠采样阈值"] / (count)
                        else:
                            class_weight = 1
                        class_weights.append(class_weight)
                        
                class_weights = torch.tensor(class_weights)
                print(f"权重：{class_weights}")
                print(f"权重的形状 ： {class_weights.shape}")
                # print(torch.tensor(class_weights))
                # print(len(class_weights))

                # 根据标签为每个样本分配权重
                global sample_weights
                sample_weights = class_weights[labels]
        else:
            print("数据样本小于1")

        
    # 整个数据集的样本个数
    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        global 上一个好的样本
        global 上一个好的标签
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # pp_index表示人员标号，从0开始
        pp_index = idx // self.num_samples
        # pn_index表示这个人的第几个样本，从0开始
        pn_index = idx % self.num_samples
        
        et_new = self.all_data[pp_index,pn_index*self.step:pn_index*self.step+self.D,0:24].T 
        # et_new = normalize(et_new)
        # 输入信号长什么样取消注释查看
        # plt.plot(et_new[0], color='red', label='R')
        # plt.plot(et_new[1], color='green', label='G')
        # plt.plot(et_new[2], color='blue', label='B')
        # plt.plot(et_new[3], color='gray', label='I')
        # plt.plot(et_new[48],linestyle = '-.', color='red', label='R_DC')
        # plt.plot(et_new[49],linestyle = '-.', color='green', label='G_DC')
        # plt.plot(et_new[50],linestyle = '-.', color='blue', label='B_DC')
        # plt.plot(et_new[51],linestyle = '-.', color='gray', label='I_DC')
        
        
        gt_SPO2 = self.all_data[pp_index,pn_index*self.step:pn_index*self.step+self.D,24]
        gt_SPO2_reshape = gt_SPO2.reshape(1,-1)
        # plt.title(f'输入数据图像 血氧 = {np.mean(gt_SPO2_reshape)}')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.show()
        # gt_SPO2_reshape = gt_SPO2_reshape[:,300-self.参数['软标签平均范围']:300+self.参数['软标签平均范围']].mean(axis=1)

        if self.参数['波峰对齐'] == True:
            for i,rppg in enumerate(et_new):
                rppg = nk.ppg_clean(rppg,sampling_rate=60, method='elgendi')
                n_th1 = (sum(rppg) // 600)
                # n_th1 = (sum(rppg) // 600) if sum(rppg) // 600 > 30 else 30
                # n_th1 = min(n_th1, 60)
                # print(rppg)
                pn_locs, n_npks = maxim_find_peaks(rppg, 600, n_th1, 40, 15)
                # print(pn_locs)
                if len(pn_locs) >= self.参数['波峰个数']:
                    rppg = rppg[pn_locs[0]:pn_locs[2]]
                    原来的索引 = np.linspace(0, len(rppg), len(rppg))
                    插值的索引 = np.linspace(0, len(rppg), 600)
                    插值后的rppg = nk.signal_interpolate(原来的索引, rppg, 插值的索引)
                    插值后的rppg = normalize(插值后的rppg)
                    et_new[i] = 插值后的rppg
                    # print(插值后的rppg.shape)
                # elif len(pn_locs) >= 2:
                #     rppg = rppg[pn_locs[0]:pn_locs[-1]]
                #     原来的索引 = np.linspace(0, len(rppg), len(rppg))
                #     插值的索引 = np.linspace(0, len(rppg), 600)
                #     插值后的rppg = nk.signal_interpolate(原来的索引, rppg, 插值的索引)
                #     插值后的rppg = normalize(插值后的rppg)
                #     et_new[i] = 插值后的rppg
                else:
                    return 上一个好的样本,上一个好的标签

        if self.参数['噪声均值'] != -1:
            et_new = 加噪声(et_new, self.参数['噪声均值'], self.参数['噪声方差']) 
            gt_SPO2_reshape = 加噪声(gt_SPO2_reshape, self.参数['噪声均值'], self.参数['噪声方差'])
        
        # cast to tensor
        et_tensor = torch.from_numpy(et_new).type(torch.FloatTensor)
        gt_SPO2_reshape = torch.from_numpy(gt_SPO2_reshape).type(torch.FloatTensor)
        
        if self.参数['波峰对齐'] == True:
            if n_npks >= self.参数['波峰个数']:
                上一个好的样本 = et_tensor
                上一个好的标签 = gt_SPO2_reshape

        # return et_tensor,gt_SPO2_reshape
        return et_tensor,gt_SPO2_reshape[0,self.参数['D']//2]