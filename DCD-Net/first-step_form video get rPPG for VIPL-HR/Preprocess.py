import numpy as np
import os
import time
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
from datetime import datetime  
import json
import torch
from sklearn.decomposition import FastICA, PCA 
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签  
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from GetrPPGFromVideo import GetrPPGFromVideo

def 检查文件后缀名(file_path):
    # 获取文件的后缀名
    _, file_extension = os.path.splitext(file_path)
    # 将后缀名转换为小写，以确保比较时不区分大小写
    file_extension = file_extension.lower()
    
    # 检查后缀名是否为.json或.csv
    if file_extension == '.json':
        return 'JSON'
    elif file_extension == '.csv':
        return 'CSV'
    else:
        return 'Unknown'


class Preprocessing():
    # 输入地址读取数据
    def __init__(self, main_path, color_path_signal, ir_path_signal, path_SPO2, output_path):
        self.main_path = main_path
        self.color_path_signal = color_path_signal
        self.ir_path_signal = ir_path_signal
        self.path_SPO2 = path_SPO2
        self.output_path = output_path

    # 多张方法实现多个区域的数据滤波
    def 滤波(self, 信号, 低截断, 高截断, 方法='butterworth', 采样率=60, order=5):
        return nk.signal_filter(信号,sampling_rate=采样率, lowcut=低截断 ,highcut=高截断, method=方法, order=order)
            
    def 插值(self, 可见光rPPG, 红外rPPG,血氧真实值):
        # 血氧真实值采样率是可见光的两倍

        信号长度 = 可见光rPPG.shape[0]

        血氧真实值 = 血氧真实值[:2*信号长度]

        血氧真实值长度 = 血氧真实值.shape[0]
        # 心率真实值长度 = 心率真实值.shape[0]

        print(f"插值前真实血氧值长度：{血氧真实值长度}")
        # print(f"插值前心率真实值长度：{心率真实值长度}")

        原来的可见光索引 = np.linspace(0, 信号长度, 信号长度)

        插值的可见光索引 = np.linspace(0, 信号长度, 信号长度*2)

        if 红外rPPG is not None:
            print("红外rPPG插值")
            len_IR = 红外rPPG.shape[0]
            原来的红外索引 = np.linspace(0, len_IR, len_IR)
            插值后的红外rPPG= np.zeros((信号长度*2, 12))
            插值的红外索引 = np.linspace(0, len_IR, 信号长度*2)

            for i in range(12):
                通道 = 红外rPPG[:, i]
                通道 = nk.signal_interpolate(原来的红外索引, 通道, 插值的红外索引)
                插值后的红外rPPG[:, i] = 通道
        else:
            插值后的红外rPPG = None
        
        原来的血氧索引 = np.linspace(0, 血氧真实值长度, 血氧真实值长度)
        # 原来的心率索引 = np.linspace(0, 心率真实值长度, 心率真实值长度)
        插值的血氧索引 = np.linspace(0, 血氧真实值长度, 信号长度*2)
        插值后的真实血氧值 = nk.signal_interpolate(原来的血氧索引, 血氧真实值, 插值的血氧索引)
        # 插值后的真实心率值 = nk.signal_interpolate(原来的心率索引, 心率真实值, 插值的血氧索引)

        # 初始化插值后的数据数组
        插值后的可见光rPPG = np.zeros((信号长度*2, 12))
        
        # 处理可见光和红外光数据的插值
        for i in range(12):
            channel = 可见光rPPG[:, i]
            channel = nk.signal_interpolate(原来的可见光索引, channel, 插值的可见光索引)
            插值后的可见光rPPG[:, i] = channel

        
        return 插值后的可见光rPPG, 插值后的红外rPPG,插值后的真实血氧值
    
    # 读取存放在excel中的rgb数据，
    #     0: B
    #     1: G
    #     2: R
    
    # 读取彩色摄像头的excel数据
    def 读取数据(self, 可见光视频路径,红外光视频路径,真实血氧值路径,数据增强方法,参数,output_path):

        # 读取可见光摄像头的数据
        print("读取可见光摄像头的数据")
        c_GRPPG = GetrPPGFromVideo(可见光视频路径)
        df = c_GRPPG.getrppgfromvideo(数据增强方法 = 数据增强方法,参数 = 参数,output_mp4=output_path,cunshipin=False)
        c_data = np.array(df)
        # if c_data.shape[0] < 1800:
        #     return c_data,c_data,c_data
        

        if os.path.exists(红外光视频路径):
            # 读取红外光摄像头的数据
            print("读取红外光摄像头的数据")
            i_GRPPG = GetrPPGFromVideo(红外光视频路径)
            df = i_GRPPG.getrppgfromvideo(数据增强方法 = 数据增强方法,参数 = 参数,output_mp4=output_path,cunshipin=False)
            i_data = np.array(df)
        else:
            i_time = None
            i_data = None
            i_t = None
    
        print("读取血氧以及心率数值")
        if 检查文件后缀名(真实血氧值路径) == 'CSV':
            SpO2_HR_and_Time = pd.read_csv(真实血氧值路径)
            GT_SPO2 = np.array(SpO2_HR_and_Time['SpO2'])
            # HR = np.array(SpO2_HR_and_Time['pulseRate'])

            # 用前一个非零值替换0值
            for i in range(1, len(GT_SPO2)):
                if GT_SPO2[i] == 0:
                    GT_SPO2[i] = GT_SPO2[i - 1]
            # 如果前面一位是0，就用后一个代替
            for i in range(len(GT_SPO2)-2,-1,-1):
                if GT_SPO2[i] == 0:
                    GT_SPO2[i] = GT_SPO2[i + 1]
                    
        elif 检查文件后缀名(真实血氧值路径) == 'JSON':
            with open(真实血氧值路径, 'r') as file:
                SpO2 = json.load(file)
            GT_SPO2 = np.array(SpO2)
             # 将数据转换为numpy数组
            GT_SPO2 = np.asarray(GT_SPO2)
            
            # 用前一个非零值替换0值
            for i in range(1, len(GT_SPO2)):
                if GT_SPO2[i] == 0:
                    GT_SPO2[i] = GT_SPO2[i - 1]
            # 如果前面一位是0，就用后一个代替
            for i in range(len(GT_SPO2)-2,-1,-1):
                if GT_SPO2[i] == 0:
                    GT_SPO2[i] = GT_SPO2[i + 1]

            
    
        print("进行插值")
        c_data_inter,i_data_inter,GT_SPO2_inter = self.插值(c_data,i_data,GT_SPO2)
        # c_data_inter = c_data_inter[360:] # 前三秒不取
        # if i_data_inter is not None:
        #     i_data_inter = i_data_inter[360:] # 前三秒不取
        # GT_SPO2_inter = GT_SPO2_inter[360:] # 前三秒不取
        # n1 = (c_data_inter.shape[0]//(10800+360))*10800
        # n2 = (i_data_inter.shape[0]//(10800+360))*10800
        # n3 = (GT_R_inter.shape[0]//(10800+360))*10800
        print(f"可见光视频长度：{c_data_inter.shape[0]}")
        print(f"真实血氧值长度：{GT_SPO2_inter.shape[0]}")
        # print(f"真实心率值长度：{HR_inter.shape[0]}")
        # n1 = (c_data_inter.shape[0]//(3600+360))*3600
        # if i_data_inter is not None:
        #     n2 = (i_data_inter.shape[0]//(3600+360))*3600
        #     n = min(n1,n2)
        # else:
        #     n = n1
        # if i_data_inter is not None:
        #     i_data_inter = i_data_inter[0:180+n+180,:]
        # c_data_inter = c_data_inter[0:180+n+180,:]
        # GT_SPO2_inter = GT_SPO2_inter[0:180+n+180]
    
        return c_data_inter,i_data_inter,GT_SPO2_inter
    
    # 去趋势
    def detrend(self, data, n, order=1):
        print("进行去趋势")
        time.sleep(2)

        data_detrend = np.zeros((data.shape[0],n))

        for i in range(n):
            channel = data[:,i]
            channel = nk.signal_detrend(channel,order=order)
            data_detrend[:,i] = channel
        
        return data_detrend
        
    def normalize(self, array):  
        print("进行标准化")
        time.sleep(2)

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
        
    
    # 快速傅里叶变换
    def fft_data(self, data, n, low, high):
        data_fft = np.zeros((data.shape[0],n))
    
        print("进行滤波")
        time.sleep(2)

        # 处理插值人脸可见光与红外光在脸部划分的12个区域的时空图
        for i in range(n):
            print(i,end=' ')
            channel = data[:,i]
            filtered = self.滤波(channel, low, high)
            # 将信号加上均值
            filtered = filtered + np.mean(channel)
            data_fft[:,i] = filtered
    
        return data_fft
    
    def fastICA_data(self, c_data, n):
        # 使用PCA进行白化  
        pca = PCA(n_components=n, whiten=True)  
        X_pca = pca.fit_transform(c_data)  
        print(X_pca.shape)
        
        # 使用FastICA提取独立成分  
        ica = FastICA(n_components=n)  
        X_ica = ica.fit_transform(X_pca)
        return X_ica
    
    def process(self, c_data,i_data):
        c_data = self.normalize(c_data)
        c_data_detrend = self.detrend(c_data, 36, order=1)
        # 进行滤波处理
        c_data_fft = self.fft_data(c_data_detrend, 36, low = 0.5,high = 4)
        # 进行fastICA
        # c_data_ICA = self.fastICA_data(c_data_fft, 36)
        if i_data is not None:
            i_data = self.normalize(i_data)
            i_data_detrend = self.detrend(i_data, 12, order=1)
            # 进行滤波处理
            i_data_fft = self.fft_data(i_data_detrend, 12, low = 0.5,high = 4)
            # # 进行fastICA
            # i_data_ICA = self.fastICA_data(i_data_fft, 12)
        else:
            i_data_fft = None
        return c_data_fft,i_data_fft
        #return c_data_fft,i_data_fft

    def 处理数据并保存(self,color_path_signal,ir_path_signal,path_SPO2,数据增强方法,参数,output_path):
        # 这里开关是否添加数据增强的数据
        # if not 'none' in output_data:
        #     print("不存在none")
        #     return 0
        参数_字符串 = f"{参数}"
        if not isinstance(参数, list):
            if 参数 < 0:
                参数绝对值 = abs(参数)
                参数_字符串 = f'负{参数绝对值}'
        output_path = output_path + f"/{数据增强方法}_{参数_字符串}"
        output_data = output_path + "/获取的rPPG.csv"
        output_不存在 = output_path + "/获取的rPPG长度不足.csv"
        if os.path.exists(output_data) or os.path.exists(output_不存在):
            print(f'{output_data}已经存在')
            return 0
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("进行数据处理")
        c_data_inter,i_data_inter,GT_SPO2_inter = self.读取数据(color_path_signal,ir_path_signal,path_SPO2,数据增强方法,参数,output_path)
        
        # if c_data_inter.shape[0] < 3600:
        #     print("可见光长度低于3600退出")
        #     output_data = output_path + "/获取的rPPG长度不足3600.csv"
        #     df1 = pd.DataFrame([0,0,0])  
        #     df1.to_csv(output_data,index=False)
        #     return 0
        # c_data,i_data = self.process(c_data_inter,i_data_inter)
        c_data,i_data = c_data_inter,i_data_inter
        GT_SPO2_inter = GT_SPO2_inter.reshape(-1, 1)
        # HR_inter = HR_inter.reshape(-1, 1)
        print("可见光数据形状")
        print(c_data.shape)
        if i_data is not None:
            print("红外光数据形状")
            print(i_data.shape)
        print("真实SPO2的数据形状")
        print(GT_SPO2_inter.shape)
        # print("真实HR的数据形状")
        # print(HR_inter.shape)

        R_all = np.zeros((1,12))
        # IR_all = np.zeros((1,36))
        R_IR_all = np.zeros((1,24))
        # SPO2_all = np.zeros((1,2))
        SPO2_all = np.zeros((1,1))
        # SPO2_HR = np.concatenate((GT_SPO2_inter,HR_inter),axis=1)
        SPO2_all = np.concatenate((SPO2_all,GT_SPO2_inter),axis=0)
                        
        if i_data is not None:
            all_segment = np.empty((c_data.shape[0], 0))
            for i in range(c_data.shape[1] // 3):
                start_col = i * 3
                end_col = start_col + 3
                segment = np.concatenate((c_data[:, start_col:end_col], i_data[:, i].reshape(-1, 1)), axis=1)
                all_segment = np.concatenate((all_segment, segment), axis=1)
                
            R_IR_all = np.concatenate((R_IR_all, all_segment), axis=0)
            data_all = np.concatenate((R_IR_all,SPO2_all),axis=1)
        else:
            R_all = np.concatenate((R_all,c_data),axis=0)
            data_all = np.concatenate((R_all,SPO2_all),axis=1)    

        df1 = pd.DataFrame(data_all[1:])  
        df1.to_csv(output_data,index=False)
    
    def getdata(self):   
        # folders = next(os.walk(self.main_path))[1]
        # for folder in folders:
        #     print(folder)
        #     color_path_signal = self.main_path + str(folder)+ self.color_path_signal
        #     print(color_path_signal)
        #     ir_path_signal = self.main_path + str(folder) + self.ir_path_signal
        #     path_SPO2 = self.main_path + str(folder) + self.path_SPO2
        #     output_path = self.output_path + str(folder)

            # 数据增强方法 = ['none','画面扭曲','运动模糊','环境光变化','缩放画面','调整亮度','调整对比度','调整饱和度','添加噪声','旋转画面','水平翻转']
            # 数据增强方法 = ['none','画面扭曲','运动模糊','缩放画面','旋转画面','水平翻转']
        数据增强方法 = ['none']
        
        """
        遍历指定目录及其子目录，查找所有 .avi 文件，并检查路径是否包含特定子路径。
        """
        for dirpath, dirnames, filenames in os.walk(self.main_path):
            for filename in filenames:
                if filename.endswith('.avi'):
                    full_path = os.path.join(dirpath, filename)
                    if 'source1' in full_path:
                    # if 'v1\\source1\\' in full_path or 'v7\\source1\\' in full_path:
                        print(f"Found: {full_path}")
                        color_path_signal = dirpath + self.color_path_signal
                        # print(color_path_signal)
                        ir_path_signal = dirpath + self.ir_path_signal
                        path_SPO2 = dirpath + self.path_SPO2
                        # # 去掉路径末尾的斜杠
                        # dirpath1 = dirpath.rstrip(os.sep)
                        # # 现在获取最后一个文件夹的名称
                        # folder = os.path.basename(dirpath1)
                        # 按 'VIPL-HR' 字符串切分字符串，最多切分一次
                        folder = dirpath.split('VIPL-HR', maxsplit=1)[1]
                        output_path = self.output_path + str(folder)
                        print(f"输出文件夹: {output_path}")

                        if os.path.exists(color_path_signal):
                            # self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,'调整色调',-30,output_path)
                            index = 0
                            print(color_path_signal)
                            for i in range(len(数据增强方法)):
                                if 数据增强方法[i] == '调整亮度':
                                    for j in range(-50,51,10):
                                        index += 1
                                        print(f'{index} {folder}')
                                        self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],j,output_path)
                                elif 数据增强方法[i] == '调整饱和度':
                                    for j in range(-50,51,10):
                                        index += 1
                                        print(f'{index} {folder}')
                                        self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],j,output_path)
                                elif 数据增强方法[i] == '画面扭曲':
                                    for j in range(3):
                                        index += 1
                                        print(f'{index} {folder}')
                                        self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],[0.001,0.005,f'第{j+1}次'],output_path)
                                elif 数据增强方法[i] == '运动模糊':
                                    for k in [3,9]:
                                        for angle in [-25,45]:
                                            index += 1
                                            print(f'{index} {folder}')
                                            self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],[k,angle],output_path)
                                elif 数据增强方法[i] == '环境光变化':
                                    for 几帧变化一次 in [5,10,20,40,60]:
                                        index += 1
                                        print(f'{index} {folder}')
                                        self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],[几帧变化一次,1],output_path)
                                elif 数据增强方法[i] == '调整对比度':
                                    for j in [0.05,0.1,0.3,0.5,0.7]:
                                        index += 1
                                        print(f'{index} {folder}')
                                        self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],j,output_path)
                                elif 数据增强方法[i] == '添加噪声':
                                    for j in [0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04]:
                                        index += 1
                                        print(f'{index} {folder}')
                                        self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],j,output_path)
                                elif 数据增强方法[i] == '调整色调':
                                    for j in [-30,-15,-8,-4,-2,3,6,10,15,30]:
                                        index += 1
                                        print(f'{index} {folder}')
                                        self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],j,output_path)
                                elif 数据增强方法[i] == '旋转画面':
                                    for j in [-45,-30,-20,-10,10,20,30,45]:
                                        index += 1
                                        print(f'{index} {folder}')
                                        self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],j,output_path)
                                elif 数据增强方法[i] == '水平翻转':
                                    index += 1
                                    print(f'{index} {folder}')
                                    self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],0,output_path)
                                elif 数据增强方法[i] == '缩放画面':
                                    for j in [0.85,0.9,1.1,1.2]:
                                        index += 1
                                        print(f'{index} {folder}')
                                        self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],j,output_path)
                                else:
                                    self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,'none',0,output_path)
                    else:
                        print(f"Skipped: {full_path}")
 
            

