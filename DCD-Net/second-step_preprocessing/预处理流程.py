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
import sys


from GetrPPGFromVideo import GetrPPGFromVideo

def plot_wavelet_transform(signal, sampling_rate=60, wavelet='cmor', scales=np.arange(1, 128)):
    """
    计算信号的连续小波变换（CWT）并绘制时频图。

    参数:
    signal : ndarray
        输入信号。
    sampling_rate : int, optional
        信号的采样率，默认为1000。
    wavelet : str, optional
        使用的小波名称，默认为'cmor'（复数Morlet小波）。
    scales : ndarray, optional
        小波变换的尺度范围，默认为1到127。

    返回:
    None
    """
    # 计算小波变换
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1.0 / sampling_rate)
    
    # 绘制时频图
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(coefficients), extent=[0, 1, frequencies[-1], frequencies[0]],
               cmap='jet', aspect='auto', vmax=abs(coefficients).max(), vmin=0)
    plt.colorbar(label='Magnitude')
    plt.title('Time-Frequency Representation using Continuous Wavelet Transform')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()

def 记录log(文字,输出文件夹路径,文件名 = 'record'):
    if not os.path.exists(f'{输出文件夹路径}/log'):
        os.makedirs(f'{输出文件夹路径}/log')
    log_path = os.path.join(
        f'{输出文件夹路径}/log/',文件名 + '.log')
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


class 预处理():
    # 输入地址读取数据
    def __init__(self, main_path, 文件名, 输出路径):
        self.主文件夹 = main_path
        self.文件名 = 文件名
        self.输出路径 = 输出路径

    def 自己写的滤波(self,channel_c,low,high):
        # 对信号进行FFT变换  
        n = len(channel_c)
        # R_mean = np.mean(channel_c)
        # IR_mean = np.mean(channel_i)
        R_fft = np.fft.rfftn(channel_c)  # FFT变换 
        
        freqs = np.fft.rfftfreq(n,1/60)
        # print(freqs)
        # print(freqs.shape)

        filtered_fft_R = np.abs(R_fft)
        # print(filtered_fft_R)
        # print(filtered_fft_R.shape)
        
        # 过滤低频和高频信号  
        for f in freqs:  
            if ((low > f) or f > high):  # 过滤条件可以根据实际需要进行调整
                if int(np.argwhere(freqs == f)[0]) >= int(filtered_fft_R.shape[0]):
                    continue
                filtered_fft_R[int(np.argwhere(freqs == f)[0])] = 0
            
        # 将过滤后的FFT结果反变换回时域，得到过滤后的信号  
        R_filtered = np.fft.irfft(filtered_fft_R)
        return R_filtered


    def kimi写的滤波(self,i, channel_c, low, high):
        n = len(channel_c)
        
        # FFT变换
        R_fft = np.fft.rfft(channel_c)  # 使用实数输入的FFT
        
        # 生成频率数组
        freqs = np.fft.rfftfreq(n, 1/60)
        
        # 创建一个掩码，初始假设所有频率都保留
        mask = np.ones_like(freqs, dtype='bool')
        
        # 过滤低频和高频信号
        mask[(freqs < low) | (freqs > high)] = False
        
        # 应用掩码
        filtered_fft_R = R_fft.copy()
        filtered_fft_R[~mask] = 0
        
        # 将过滤后的FFT结果反变换回时域，得到过滤后的信号
        R_filtered = np.fft.irfft(filtered_fft_R)

        if i < 48:
            # 绘制频谱图
            plt.figure(figsize=(12, 6))
            # 原始信号的频谱
            plt.subplot(2, 1, 1)
            in_range = (freqs >= 0.4) & (freqs <= 5)
            plt.plot(freqs[in_range], np.abs(R_fft)[in_range], label='Original Signal Spectrum')
            plt.title('Original Signal Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.grid(True)
            plt.legend()

            # 滤波后的信号频谱
            plt.subplot(2, 1, 2)
            plt.plot(freqs[in_range], np.abs(filtered_fft_R)[in_range], label='Filtered Signal Spectrum', color='r')
            plt.title('Filtered Signal Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.savefig(f'{self.output_path}/频域分布图{i}.png')
            # plt.show()
            plt.close()

            # plt.show()
        
        return R_filtered

    def 自己写的滤波的AC除以DC(self, i, 信号, 低截断, 高截断, 方法='butterworth', 采样率=60, order=5):
        AC = self.kimi写的滤波(i,信号,低截断 ,高截断)
        DC = self.kimi写的滤波(i+48,信号,0,0.1)
        # print(np.array(DC).mean())
        # AAC = self.滤波(信号,低截断 ,高截断)
        # DDC = self.滤波(信号,0 ,0.1)
        # plt.plot(AC, color='red', label='AC')
        # plt.plot(AAC, color='blue', label='AAC')
        # plt.title('Scatter Plot of Labels vs Predictions')
        # plt.xlabel('Labels')
        # plt.ylabel('Predictions')
        # plt.show()
        # if i % 3 == 0:
        #     plt.plot(self.snormalize(DC), color = 'r', label=f'{i}')
        # elif i % 3 == 1:
        #     plt.plot(self.snormalize(DC), color = 'g', label=f'{i}')
        # elif i % 3 == 2:
        #     plt.plot(self.snormalize(DC), color = 'b', label=f'{i}')
        # plt.plot(DDC, color='blue', label='DDC')
        # plt.title('Scatter Plot of Labels vs Predictions')
        # plt.xlabel('Labels')
        # plt.ylabel('Predictions')
        # plt.show()

        # plt.plot(AC/DC, color='red', label='AC/DC')
        # plt.title('Scatter Plot of Labels vs Predictions')
        # plt.xlabel('Labels')
        # plt.ylabel('Predictions')
        # plt.show()
        return AC/DC,DC

    def find_max_amplitude_frequency(self,column, fs=60):
        # FFT变换
        fft_result = np.fft.rfft(column)
        # 生成频率数组
        freqs = np.fft.rfftfreq(len(column), 1/fs)
        # 计算对数因子
        log_factor = 0.1*freqs+0.8
        log_factor[freqs >= 1] = 1  # 对于1Hz及以上的频率，对数因子为1

        # 应用对数因子
        modified_fft_result = fft_result * log_factor
        # 找到1Hz到2Hz之间的频率索引
        in_range = (freqs >= 0.8) & (freqs <= 2)
        # 找到1Hz到2Hz之间振幅最大的频率位置
        max_amplitude_index = np.argmax(np.abs(modified_fft_result)[in_range])
        max_amplitude_frequency = freqs[in_range][max_amplitude_index]
        # print(f"{max_amplitude_frequency:.4f}",end=" ")
        return max_amplitude_frequency

    # 多张方法实现多个区域的数据滤波
    def 滤波(self, 信号, 低截断, 高截断, 方法='butterworth', 采样率=60, order=5):
        return nk.signal_filter(信号,sampling_rate=采样率, lowcut=低截断 ,highcut=高截断, method=方法, order=order)
            
    def AC除以DC(self, 信号, 低截断, 高截断, 方法='butterworth', 采样率=60, order=5):
        AC = nk.signal_filter(信号,sampling_rate=采样率, lowcut=低截断 ,highcut=高截断, method=方法, order=order)
        DC = nk.signal_filter(信号,sampling_rate=采样率, lowcut=0 ,highcut=0.3, method=方法, order=order)
        print(np.array(DC).mean())
        return AC/DC

    def 插值(self, 可见光rPPG, 红外rPPG,血氧真实值):
        血氧真实值长度 = 血氧真实值.shape[0]

        信号长度 = 可见光rPPG.shape[0]

        原来的可见光索引 = np.linspace(0, 信号长度, 信号长度)

        插值的可见光索引 = np.linspace(0, 信号长度, 信号长度*2)

        if 红外rPPG is not None:
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
        插值的血氧索引 = np.linspace(0, 血氧真实值长度, 信号长度*2)
        插值后的真实血氧值 = nk.signal_interpolate(原来的血氧索引, 血氧真实值, 插值的血氧索引)

        # 初始化插值后的数据数组
        插值后的可见光rPPG = np.zeros((信号长度*2, 36))
        
        # 处理可见光和红外光数据的插值
        for i in range(36):
            channel = 可见光rPPG[:, i]
            channel = nk.signal_interpolate(原来的可见光索引, channel, 插值的可见光索引)
            插值后的可见光rPPG[:, i] = channel

        
        return 插值后的可见光rPPG, 插值后的红外rPPG,插值后的真实血氧值
    
    # 读取存放在excel中的rgb数据，
    #     0: B
    #     1: G
    #     2: R
    
    # 读取彩色摄像头的excel数据
    def 读取数据(self, 粗制rPPG数据路径):

        print("读取粗制rPPG数据")
        if 检查文件后缀名(粗制rPPG数据路径) == 'CSV':
            粗制rPPG = pd.read_csv(粗制rPPG数据路径)
            粗制rPPG = np.array(粗制rPPG)
    
        print(f"数据长度：{粗制rPPG.shape[0]}")
        if 粗制rPPG.shape[1]>=48:
            ci_data_inter = 粗制rPPG[:,:24]
            GT_SPO2_inter = 粗制rPPG[:,24]
        else:
            ci_data_inter = 粗制rPPG[:,:12]
            GT_SPO2_inter = 粗制rPPG[:,12]
            if 粗制rPPG.shape[1] == 38:
                HR_inter = 粗制rPPG[:,37]
            else:
                HR_inter = None


        return ci_data_inter,GT_SPO2_inter,HR_inter
    
    # 去趋势
    def detrend(self, data, n, order=1):
        print("进行去趋势")
        # time.sleep(2)

        data_detrend = np.zeros((data.shape[0],n))

        for i in range(n):
            channel = data[:,i]
            channel = nk.signal_detrend(channel,order=order)
            data_detrend[:,i] = channel
        
        return data_detrend
        
    def snormalize(self, array):  
        # print("进行标准化")
        # time.sleep(2)

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
        
    def normalize(self, array):  
        print("进行归一化")
        # time.sleep(2)
        
        if isinstance(array, torch.Tensor):  
            # 对于 PyTorch 张量，使用 PyTorch 方法  
            max = array.max()  
            min = array.min()  
        else:  
            # 对于 NumPy 数组，使用 NumPy 函数  
            max = np.max(array)  
            min = np.min(array) 

        # 检查 max 和 min 是否相等
        if max == min:
            raise ValueError("max and min values are the same, cannot divide by zero")
        
        # 检查是否存在 NaN 或 Inf 值
        if np.isnan(min) or np.isnan(max) or np.isinf(min) or np.isinf(max):
            raise ValueError("Array contains NaN or Inf values")

        return (array - min) / (max - min)

        
    
    # 快速傅里叶变换
    def fft_data(self, data, n, HR):
        data_fft = np.zeros((data.shape[0],n))
        data_dc = np.zeros((data.shape[0],n))
    
        print("进行滤波")
        # time.sleep(2)

        # 处理插值人脸可见光与红外光在脸部划分的12个区域的时空图
        
        for i in range(n):
            print(i,end=' ')
            channel = data[:,i]
            # for j in range(channel.shape[0] // 600 + 1):
            if HR is not None:
                HR_mean = np.mean(HR[600:-600])
                lowFrequency = HR_mean/60 - 0.1
                highFrequency = HR_mean/60 + 0.1
            else:
                lowFrequency = 0.8
                highFrequency = 4
            #     channel[j*600:j*600+600] = self.自己写的滤波的AC除以DC(i+48,channel[j*600:j*600+600], lowFrequency, highFrequency)
                # print(channel.shape[0])
                # channel = self.data_dtcwt(channel)
            # plot_wavelet_transform(channel)
            filtered,dc = self.自己写的滤波的AC除以DC(i,channel, lowFrequency, highFrequency)
            # 调用函数绘制时频图
            
            # channel = self.data_dtcwt(channel)
                # 将信号加上均值
                # filtered = filtered + np.mean(channel)
                # print(filtered.shape[0])
            data_dc[:dc.shape[0],i] = dc
            data_fft[:filtered.shape[0],i] = filtered
    
        return data_fft,data_dc
    
    def fastICA_data(self, c_data, n):
        # 使用PCA进行白化  
        pca = PCA(n_components=n, whiten=True)  
        X_pca = pca.fit_transform(c_data)  
        print(X_pca.shape)
        
        # 使用FastICA提取独立成分  
        ica = FastICA(n_components=n)  
        X_ica = ica.fit_transform(X_pca)
        return X_ica
    
    def 振幅计算(self,data,win):
        for i in range(data.shape[1]):
            for j in range(0,(data.shape[0]//win+1)):
                window = data[j*win:min(j*win+win,data.shape[0]-1),i]
                if j*win+win > data.shape[0]-1:
                    window = data[data.shape[0]-win:data.shape[0]-1,i]
                if len(window) == 0:
                    raise ValueError("Window is empty. Check the data input and window size.")
                
                A = np.max(window) - np.min(window)
                data[j*win:min(j*win+win,data.shape[0]-1),i] = A

        return data
    
    # 步骤3：自定义均值滤波函数
    def custom_mean_filter(self,signal, window_size):
        # k 是每个窗口内要去掉的最大值和最小值的数量
        filtered_signal = np.zeros_like(signal)
        for i in range(len(signal)):
            # 确定窗口的范围
            start = max(0, i - window_size // 2)
            end = min(len(signal), i + window_size // 2 + 1)
            window = signal[start:end]
            
            # # 计算窗口的均值
            # window_mean = np.mean(window)
            
            # # 去掉超过2倍均值的数
            # window = window[(window >= -k * window_mean) & (window <= k * window_mean)]
            
            # 计算剩余值的平均
            if len(window) > 0:
                filtered_signal[i] = np.mean(window)
            else:
                filtered_signal[i] = 0  # 如果窗口内所有值都被去掉，则设为0或进行其他处理
        return filtered_signal

    def 基线计算2(self,data,win):
        for i in range(data.shape[1]):
            # 步骤1：将负数取反
            signal = np.abs(data[:,i])

            # 步骤2：复制前100个数和后100个数
            extended_signal = np.concatenate((signal[:win//2], signal, signal[-(win//2):]))

            # 步骤3：使用窗口是200的均值滤波
            filtered_signal = self.custom_mean_filter(extended_signal,win)
            # print(len(signal),end=' ')
            data[:,i] = filtered_signal[win//2:len(filtered_signal)-win//2]
            # print(len(data))

        return data
    
    def 一次多项式计算趋势(self,data,d=1):
        for i in range(data.shape[1]):
            # 拟合一个一次多项式（线性趋势）
            coeffs = np.polyfit(np.arange(len(data[:,i])), data[:,i], d)
            data[:,i] = np.poly1d(coeffs)(np.arange(len(data[:,i])))
        return data

    def calculate_pearson_correlation_coefficient(self,signal1, signal2):
        """
        计算两个信号的皮尔逊相关性系数。

        参数:
        signal1 (array_like): 第一个信号数组。
        signal2 (array_like): 第二个信号数组。

        返回:
        float: 两个信号的皮尔逊相关性系数。
        """
        signal1 = np.asarray(signal1).flatten()
        signal2 = np.asarray(signal2).flatten()
        # 计算皮尔逊相关系数矩阵
        correlation_matrix = np.corrcoef(signal1, signal2)
        
        # 相关系数矩阵中，[0, 1] 位置的值是 signal1 和 signal2 的相关系数
        correlation_coefficient = correlation_matrix[0, 1]
        
        return correlation_coefficient
    
    def 减鼻子像素(self,data):
        # 
        for i in range(data.shape[1]//3):
            if i == 6:
                data[:,i*3] = data[:,i*3] - data[:,33]
                data[:,i*3+1] = data[:,i*3+1] - data[:,34]
                data[:,i*3+2] = data[:,i*3+2] - data[:,35]
                continue
            print(i,end=' ')
            data[:,i*3] = data[:,i*3] - data[:,18]
            data[:,i*3+1] = data[:,i*3+1] - data[:,19]
            data[:,i*3+2] = data[:,i*3+2] - data[:,20]

        return data

    def 计算滤波范围(self,c_data):
        g_data = np.concatenate((c_data[:,1].reshape(-1,1), c_data[:,1].reshape(-1,1), c_data[:,1].reshape(-1,1), c_data[:,1].reshape(-1,1), c_data[:,1].reshape(-1,1), c_data[:,1].reshape(-1,1), 
                                 c_data[:,7].reshape(-1,1), c_data[:,7].reshape(-1,1), c_data[:,7].reshape(-1,1),c_data[:,7].reshape(-1,1), c_data[:,7].reshape(-1,1), c_data[:,7].reshape(-1,1),
                                 c_data[:,13].reshape(-1,1),
                                 c_data[:,16].reshape(-1,1),c_data[:,16].reshape(-1,1),
                                 c_data[:,17].reshape(-1,1),
                                 c_data[:,19].reshape(-1,1),c_data[:,19].reshape(-1,1),c_data[:,19].reshape(-1,1),c_data[:,19].reshape(-1,1),c_data[:,19].reshape(-1,1),c_data[:,19].reshape(-1,1),
                                 c_data[:,22].reshape(-1,1),c_data[:,22].reshape(-1,1),
                                 c_data[:,28].reshape(-1,1),
                                 c_data[:,34].reshape(-1,1),), axis=1)
        max_amplitude_frequencies = np.apply_along_axis(self.find_max_amplitude_frequency, 0, g_data)

        # 排序频率数组
        sorted_frequencies = np.sort(max_amplitude_frequencies)
        # print(sorted_frequencies)

        # 滑动窗口
        window_width = 0.2
        max_count = 0
        lowFrequency = 0
        highFrequency = 0
        mean = 0
        fff = [0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.0]
        for i in range(len(fff)):
            # 确定窗口的起始和结束位置
            start = fff[i]
            end = start + window_width
            
            # 计算窗口内包含的频率数量
            count = np.sum((sorted_frequencies >= start) & (sorted_frequencies <= end))
            
            # 更新最大数量和对应的频率范围
            if count > max_count:
                max_count = count
                mean = np.mean(sorted_frequencies[(sorted_frequencies >= start) & (sorted_frequencies <= end)])
                lowFrequency = mean - 0.15
                highFrequency = mean + 0.15

        # 输出结果
        # print(f"\nlowFrequency: {lowFrequency}, highFrequency: {highFrequency} mean: {mean}")

        # 统计分布
        plt.hist(max_amplitude_frequencies*60, bins=72, range=(48, 120))
        # 在取的频域范围的位置画一条竖直线
        plt.axvline(x=lowFrequency*60, color='r', linestyle='--')
        plt.axvline(x=highFrequency*60, color='r', linestyle='--')
        plt.axvline(x=mean*60, color='blue', linestyle='-')
        plt.title('Distribution of Maximum Amplitude Frequencies')
        plt.xlabel('心率')
        plt.ylabel('Count')
        plt.savefig(f'{self.output_path}/最大振幅范围_FFT.png')
        # plt.show()
        plt.close()

        return lowFrequency,highFrequency,max_count/len(sorted_frequencies)

    def remove_extremes(self,lst):
    # 1. 对列表进行排序
        sorted_lst = sorted(lst)
        
        # 2. 计算要去除的元素数量（即列表长度的25%）
        n = len(sorted_lst)
        quarter = n // 4
        
        # 3. 去除前25%和后25%的元素
        # 从排序后的列表中，去除前quarter个元素和后quarter个元素
        filtered_lst = sorted_lst[quarter:-quarter] if n > 1 else []
        
        return filtered_lst

    def 计算心率(self,data):
        hrs = []
        for i in range(data.shape[1]//3):
            if i == 1 or i == 3 or i==8 or i == 10:
                continue
            signals, info = nk.ppg_process(data[:,i*3+1], sampling_rate=60)
            ppg_rate_column = signals['PPG_Rate']
            # time.sleep(1000)
            hrs.extend(np.array(ppg_rate_column))
        # hrs = self.remove_extremes(hrs)

        hrs = np.sort(np.array(hrs))
         # 滑动窗口
        window_width = 18
        max_count = 0
        lowFrequency = 0
        highFrequency = 0
        mean = 0
        fff = np.arange(40, 150, 6)
        for i in range(len(fff)):
            # 确定窗口的起始和结束位置
            start = fff[i]
            end = start + window_width
            
            # 计算窗口内包含的频率数量
            count = np.sum((hrs >= start) & (hrs <= end))
            
            # 更新最大数量和对应的频率范围
            if count > max_count:
                max_count = count
                mean = np.mean(hrs[(hrs >= start) & (hrs <= end)])
                lowFrequency = (mean - 9)/60
                highFrequency = (mean + 9)/60

        # 输出结果
        # print(f"\nlowFrequency: {lowFrequency}, highFrequency: {highFrequency} mean: {mean/60}")
        # 统计分布
        plt.hist(hrs, bins=120, range=(30, 150))
        # 在取的频域范围的位置画一条竖直线
        plt.axvline(x=lowFrequency*60, color='r', linestyle='--')
        plt.axvline(x=highFrequency*60, color='r', linestyle='--')
        plt.axvline(x=mean, color='blue', linestyle='-')
        plt.title('Distribution of Maximum Amplitude Frequencies')
        plt.xlabel('心率')
        plt.ylabel('Count')
        plt.savefig(f'{self.output_path}/最大振幅范围_峰值计算.png')
        # plt.show()
        plt.close()
        return lowFrequency,highFrequency,max_count/len(hrs)

    def soft_thresholding(self,x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def data_dtcwt(self,data):
        # 初始化DTCWT变换对象，设置分解层数
        transform = dtcwt.Transform1d()
        coeffs = transform.forward(data, nlevels=3)

        # 这里可以应用阈值处理或其他去噪方法来处理coeffs
        # 例如，使用软阈值去噪
        # 软阈值去噪
        threshold = 0.9  # 设置阈值

        # 由于Pyramid对象不能直接迭代，我们需要分别处理每个子带
        denoised_highpasses = []
        for highpass in coeffs.highpasses:
            # 应用软阈值处理
            denoised_highpass = self.soft_thresholding(highpass, threshold)
            denoised_highpasses.append(denoised_highpass)

        # 重构Pyramid对象
        denoised_coeffs = dtcwt.Pyramid(coeffs.lowpass, tuple(denoised_highpasses))

        # 逆变换以重构去噪后的信号
        denoised_signal = transform.inverse(denoised_coeffs)

        plt.figure(figsize=(10, 6))
        plt.plot(data, label='Original Signal')
        plt.plot(denoised_signal, label='Denoised Signal', linestyle='--')
        plt.legend()
        plt.show()
        plt.close()

        return denoised_signal


    def process(self, ci_data,GT_SPO2,HR_inter):

        plt.plot(ci_data[:,0], color='red', label='r')
        plt.plot(ci_data[:,1], color='green', label='g')
        plt.plot(ci_data[:,2], color='blue', label='b')
        if ci_data.shape[1] >= 24:
            plt.plot(ci_data[:,3], color='gray', label='i')
        plt.title('原始数据')
        plt.xlabel('Labels')
        plt.ylabel('Predictions')
        plt.savefig(f'{self.output_path}/原始数据.png')
        plt.close()

        # plt.show()
        # 进行减鼻子上的像素
        # c_data = self.减鼻子像素(c_data)
        # plt.plot(c_data[:,0], color='red', label='r')
        # plt.plot(c_data[:,1], color='green', label='g')
        # plt.plot(c_data[:,2], color='blue', label='b')
        # plt.title('减鼻子')
        # plt.xlabel('Labels')
        # plt.ylabel('Predictions')
        # plt.show()
        两者不同 = 0
        # lowFrequency_fft,highFrequency_fft,score_fft = self.计算滤波范围(c_data)
        # lowFrequency_d,highFrequency_d,score_d = self.计算心率(c_data)
        # print(f'峰值心率 = {60*(lowFrequency_d+highFrequency_d)/2} lowFrequency = {lowFrequency_d:.4f} , highFrequency = {highFrequency_d:.4f} , 分数 = {score_d}')
        # print(f'FFT心率 = {60*(lowFrequency_fft+highFrequency_fft)/2} lowFrequency = {lowFrequency_fft:.4f} , highFrequency = {highFrequency_fft:.4f} , 分数 = {score_fft}')
        # if np.abs(60*(lowFrequency_d+highFrequency_d)/2 - 60*(lowFrequency_fft+highFrequency_fft)/2) > 9:
        #     lowFrequency = min(lowFrequency_d,lowFrequency_fft)
        #     highFrequency = max(highFrequency_d,highFrequency_fft)
        #     两者不同 = 1
        #     记录log(f'峰值心率 = {60*(lowFrequency_d+highFrequency_d)/2} lowFrequency = {lowFrequency_d:.4f} , highFrequency = {highFrequency_d:.4f} , 分数 = {score_d}',self.输出路径)
        #     记录log(f'FFT心率 = {60*(lowFrequency_fft+highFrequency_fft)/2} lowFrequency = {lowFrequency_fft:.4f} , highFrequency = {highFrequency_fft:.4f} , 分数 = {score_fft}',self.输出路径)
        # elif score_fft > 0.8:
        #     lowFrequency = lowFrequency_fft
        #     highFrequency = highFrequency_fft
        # elif score_d > 0.4:
        #     lowFrequency = lowFrequency_d
        #     highFrequency = highFrequency_d
        # else:
        #     lowFrequency = np.mean([lowFrequency_d,lowFrequency_fft])
        #     highFrequency = np.mean([highFrequency_d,highFrequency_fft])

        # time.sleep(1000)
        dataa = ci_data.copy()
        cidc_data = self.基线计算2(dataa,win=120)

        ci_data_d = np.concatenate([cidc_data[1:,:],cidc_data[cidc_data.shape[0]-1:,:]],axis=0)
        ci_diff = ci_data_d - cidc_data
        if ci_data.shape[1] >= 24:
            Q1 = np.percentile(ci_diff[:,22], 25)
            Q3 = np.percentile(ci_diff[:,22], 75)
        else:
            Q1 = np.percentile(ci_diff[:,10], 25)
            Q3 = np.percentile(ci_diff[:,10], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR
        print("Q1,Q3")
        print(Q1,Q3)
        print("低值边界")
        print(lower_bound)
        print("高值边界")
        print(upper_bound)

        # 标记异常值
        if ci_data.shape[1] >= 24:
            outliers = (ci_diff[:,22] < lower_bound) | (ci_diff[:,22] > upper_bound)
        else:
            outliers = (ci_diff[:,10] < lower_bound) | (ci_diff[:,10] > upper_bound)

        window_size = 30

        # 扩展异常值标记到邻居
        extended_outliers = np.zeros_like(outliers, dtype='bool')
        for i in range(ci_diff.shape[0]):
            if outliers[i]:
                # 将当前异常值及其前后window_size个值标记为异常
                start = max(0, i - window_size)
                if ci_data.shape[1] >= 24:
                    end = min(len(ci_diff[:,22]), i + window_size + 1)
                else:
                    end = min(len(ci_diff[:,10]), i + window_size + 1)
                extended_outliers[start:end] = True
        
        # 切除异常值
        print(extended_outliers)
        # 记录异常值的索引
        outlier_indices = np.where(extended_outliers)[0]
        # 绘制数据
        plt.figure(figsize=(10, 6))
        if ci_data.shape[1] >= 24:
            plt.scatter(range(len(ci_data[:,22])), ci_data[:,22], label='Data', color='blue')
            plt.scatter(outlier_indices, ci_data[extended_outliers,22], label='Outliers', color='red')

        else:
            plt.scatter(range(len(ci_data[:,10])), ci_data[:,10], label='Data', color='blue')
            plt.scatter(outlier_indices, ci_data[extended_outliers,10], label='Outliers', color='red')

        # # 绘制异常值
        # print(outlier_indices.shape)
        # print(ci_data[extended_outliers].shape)
        # plt.scatter(outlier_indices, ci_data[extended_outliers,45], label='Outliers', color='red')

        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Data with Outliers')
        plt.legend()
        plt.savefig(f'{self.output_path}/删除的异常值.png')
        # plt.show()
        plt.close()

        cidc_data = cidc_data[~extended_outliers,:]
        ci_diff = ci_diff[~extended_outliers,:]
        ci_data = ci_data[~extended_outliers,:]
        GT_SPO2 = GT_SPO2[~extended_outliers]
        
        plt.plot(ci_diff[:,0], color='red', label='r')
        plt.plot(ci_diff[:,1], color='green', label='g')
        plt.plot(ci_diff[:,2], color='blue', label='b')
        if ci_data.shape[1] >= 24:
            plt.plot(ci_diff[:,3], color='gray',label='i')
        # if i_data is not None:
        #     plt.plot(i_data[:,0], label='i')
        plt.title('diff')
        plt.xlabel('Labels')
        plt.ylabel('Predictions')
        plt.legend()
        plt.savefig(f'{self.output_path}/diff.png')
        # plt.show()
        plt.close()

        plt.plot(cidc_data[:,0], color='red', label='r')
        plt.plot(cidc_data[:,1], color='green', label='g')
        plt.plot(cidc_data[:,2], color='blue', label='b')
        if ci_data.shape[1] >= 24:
            plt.plot(cidc_data[:,3], color='gray',label='i')
        # if i_data is not None:
        #     plt.plot(i_data[:,0], label='i')
        plt.title('dc去除异常值数据')
        plt.xlabel('Labels')
        plt.ylabel('Predictions')
        plt.legend()
        plt.savefig(f'{self.output_path}/dc去除异常值数据.png')
        # plt.show()
        plt.close()

        plt.plot(ci_data[:,0], color='red', label='r')
        plt.plot(ci_data[:,1], color='green', label='g')
        plt.plot(ci_data[:,2], color='blue', label='b')
        if ci_data.shape[1] >= 24:
            plt.plot(ci_data[:,3], color='gray',label='i')
        # if i_data is not None:
        #     plt.plot(i_data[:,0], label='i')
        plt.title('原始数据去除异常值数据')
        plt.xlabel('Labels')
        plt.ylabel('Predictions')
        plt.legend()
        plt.savefig(f'{self.output_path}/原始数据去除异常值数据.png')
        # plt.show()
        plt.close()

        plt.plot(GT_SPO2[:], color='red', label='r')

        # if i_data is not None:
        #     plt.plot(i_data[:,0], label='i')
        plt.title('血氧数据去除异常值数据')
        plt.xlabel('Labels')
        plt.ylabel('Predictions')
        plt.legend()
        plt.savefig(f'{self.output_path}/血氧数据去除异常值数据.png')
        # plt.show()
        plt.close()

        
        # 心率
        # HR_mean = np.mean(HR_inter)
        # lowFrequency = HR_mean/60 - 0.1
        # highFrequency = HR_mean/60 + 0.1

        # 进行滤波处理
        if ci_data.shape[1] >= 24:
            ci_data,data_dc = self.fft_data(ci_data, 24, None)
            Q1 = np.percentile(ci_data[:,22], 25)
            Q3 = np.percentile(ci_data[:,22], 75)
        else:
            ci_data,data_dc = self.fft_data(ci_data, 12, None)
            Q1 = np.percentile(ci_data[:,10], 25)
            Q3 = np.percentile(ci_data[:,10], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        print(Q1,Q3)
        print(lower_bound)
        print(upper_bound)

        # 标记异常值
        if ci_data.shape[1] >= 24:
            outliers = (ci_data[:,22] < lower_bound) | (ci_data[:,22] > upper_bound)
        else:
            outliers = (ci_data[:,10] < lower_bound) | (ci_data[:,10] > upper_bound)

        window_size = 30

        # 扩展异常值标记到邻居
        extended_outliers = np.zeros_like(outliers, dtype='bool')
        for i in range(ci_data.shape[0]):
            if outliers[i]:
                # 将当前异常值及其前后window_size个值标记为异常
                start = max(0, i - window_size)
                if ci_data.shape[1] >= 24:
                    end = min(len(ci_data[:,22]), i + window_size + 1)
                else:
                    end = min(len(ci_data[:,10]), i + window_size + 1)
                    
                extended_outliers[start:end] = True
        
        # 切除异常值
        print(extended_outliers)
        # 记录异常值的索引
        outlier_indices = np.where(extended_outliers)[0]
        # 绘制数据
        plt.figure(figsize=(10, 6))

        if ci_data.shape[1] >= 24:
            plt.scatter(range(len(ci_data[:,22])), ci_data[:,22], label='Data', color='blue')

        # # 绘制异常值
        # print(outlier_indices.shape)
        # print(ci_data[extended_outliers].shape)
            plt.scatter(outlier_indices, ci_data[extended_outliers,22], label='Outliers', color='red')

        else:
            plt.scatter(range(len(ci_data[:,10])), ci_data[:,10], label='Data', color='blue')
            plt.scatter(outlier_indices, ci_data[extended_outliers,10], label='Outliers', color='red')

        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Data with Outliers')
        plt.legend()
        plt.savefig(f'{self.output_path}/FFT之后删除的异常值.png')
        # plt.show()
        plt.close()

        # cidc_data = data_dc[~extended_outliers,:]# 这里使用的是滤波之后得到的0-0.1HZ的波
        cidc_data = cidc_data[~extended_outliers,:] 
        # ci_diff = ci_diff[~extended_outliers,:]
        ci_data = ci_data[~extended_outliers,:]
        GT_SPO2 = GT_SPO2[~extended_outliers]

        for i in range(4):
            if ci_data.shape[1] >= 24:
                plt.plot(ci_data[:,i*4+0], color='red', label='r')
                plt.plot(ci_data[:,i*4+1], color='green', label='g')
                plt.plot(ci_data[:,i*4+2], color='blue', label='b')
                plt.plot(ci_data[:,i*4+3], color='gray', label='i')
            else:
                plt.plot(ci_data[:,i*3+0], color='red', label='r')
                plt.plot(ci_data[:,i*3+1], color='green', label='g')
                plt.plot(ci_data[:,i*3+2], color='blue', label='b')
            plt.title('FFT')
            plt.xlabel('Labels')
            plt.ylabel('Predictions')
            plt.legend()
            plt.savefig(f'{self.output_path}/FFT之后删除异常值之后的rPPG图像{i}.png')

            # plt.show()
            plt.close()

        # cidc_data = self.基线计算2(cidc_data,win=120)
        ci_data_copy = ci_data.copy()
        cidc_data = self.一次多项式计算趋势(ci_data_copy,1) # 使用一阶

        for i in range(cidc_data.shape[1]):
            cidc_data[:,i] = cidc_data[:,i]/np.mean(cidc_data[:,i]) - 1
        # cidc_data[:,9:12] = cidc_data[:,33:36]

        plt.plot(cidc_data[:,0], color='red', label='r')
        plt.plot(cidc_data[:,1], color='green', label='g')
        plt.plot(cidc_data[:,2], color='blue', label='b')
        if ci_data.shape[1] >= 24:
            plt.plot(cidc_data[:,3], color='gray',label='i')
        # if i_data is not None:
        #     plt.plot(i_data[:,0], label='i')
        plt.title('dc去除异常值数据-除以均值减1')
        plt.xlabel('Labels')
        plt.ylabel('Predictions')
        plt.legend()
        plt.savefig(f'{self.output_path}/dc去除异常值数据-除以均值减1.png')
        # plt.show()
        plt.close()



        # if ci_data.shape[1] <= 38:
        #     c_data = ci_data.reshape(-1,12,3)
        #     data = np.zeros((c_data.shape[0],12,3))
        #     r = np.concatenate([c_data[:,:3,0],c_data[:,11:12,0]],1)
        #     g = np.concatenate([c_data[:,:3,1],c_data[:,11:12,1]],1)
        #     b = np.concatenate([c_data[:,:3,2],c_data[:,11:12,2]],1)
        #     # 进行盲源分离
        #     S_estimated_r, whitening_matrix_r, V_r = sobi(r, num_lags=200, max_iter=10000, tol=1e-8)
        #     S_estimated_g, whitening_matrix_g, V_r = sobi(g, num_lags=200, max_iter=10000, tol=1e-8)
        #     S_estimated_b, whitening_matrix_b, V_r = sobi(b, num_lags=200, max_iter=10000, tol=1e-8)
        #     print(S_estimated_r.shape)
        #     data[:,:,0] = np.repeat(S_estimated_r, 3, axis=1)
        #     data[:,:,1] = np.repeat(S_estimated_g, 3, axis=1)
        #     data[:,:,2] = np.repeat(S_estimated_b, 3, axis=1)
        #     data = data.reshape(-1,36)

        #     # S_estimated_r = self.fft_data(S_estimated_r, 9, low = 0.8,high = 2.5)
        #     plt.plot(S_estimated_r)

        #     plt.title('r通道SOBI之后的信号')
        #     plt.xlabel('Labels')
        #     plt.ylabel('Predictions')
        #     # plt.legend()
        #     plt.savefig(f'{self.output_path}/r通道SOBI之后的信号.png')
        #     # plt.show()
        #     plt.close()

        #     # S_estimated_g = self.fft_data(S_estimated_g, 9, low = 0.8,high = 2.5)
        #     plt.plot(S_estimated_g)
        #     plt.title('g通道SOBI之后的信号')
        #     plt.xlabel('Labels')
        #     plt.ylabel('Predictions')
        #     # plt.legend()
        #     plt.savefig(f'{self.output_path}/g通道SOBI之后的信号.png')
        #     # plt.show()
        #     plt.close()
            
        #     # S_estimated_b = self.fft_data(S_estimated_b, 9, low = 0.8,high = 2.5)
        #     plt.plot(S_estimated_b)
        #     plt.title('b通道SOBI之后的信号')
        #     plt.xlabel('Labels')
        #     plt.ylabel('Predictions')
        #     # plt.legend()
        #     plt.savefig(f'{self.output_path}/b通道SOBI之后的信号.png')
        #     # plt.show()
        #     plt.close()


        # d = 400
        # if ci_data.shape[0] - 400 < 3700:
        #     d = min(1,ci_data.shape[0] - 3700)


        # ci_data = self.snormalize(ci_data)


        # i_data = self.snormalize(i_data[d//2:-1*(d//2),:])
        # ci_data = ci_data[d//2:-1*(d//2),:]
        # cidc_data = cidc_data[d//2:-1*(d//2),:]
        
        # plt.plot(c_data[:,0], color='red', label='r')
        # plt.plot(c_data[:,1], color='green', label='g')
        # plt.plot(c_data[:,2], color='blue', label='b')
        # plt.plot(c_data[:,3:], color='blue', label='b')
        # plt.title('normlize')
        # plt.xlabel('Labels')
        # plt.ylabel('Predictions')
        # plt.legend()
        # plt.show()
        # print(c_data.shape)
        # c_data = self.detrend(c_data, 36, order=1)
        # if i_data is not None:
        #     print(i_data.shape)
        #     i_data = self.detrend(i_data, 12, order=1)

        # plt.plot(c_data[:,0], color='red', label='r')
        # plt.plot(c_data[:,1], color='green', label='g')
        # plt.plot(c_data[:,2], color='blue', label='b')
        # # plt.plot(c_data[:,3:], color='blue', label='b')
        # plt.title('normlize')
        # plt.xlabel('Labels')
        # plt.ylabel('Predictions')
        # plt.legend()
        # plt.show()
        # 进行减鼻子上的像素
        # c_data = self.减鼻子像素(c_data)
        # plt.plot(c_data[:,0], color='red', label='r')
        # plt.plot(c_data[:,1], color='green', label='g')
        # plt.plot(c_data[:,2], color='blue', label='b')
        # plt.plot(c_data[:,3:], color='blue', label='b')
        # plt.title('normlize')
        # plt.xlabel('Labels')
        # plt.ylabel('Predictions')
        # plt.legend()
        # plt.show()
        # c_data = self.normalize(c_data)
        # z_data = np.copy(c_data)
        # z_data = self.振幅计算2(z_data,400,1.2)
        # plt.figure()
        # plt.plot(z_data[:,0], color='red', label='r')
        # plt.plot(z_data[:,1], color='green', label='g')
        # plt.plot(z_data[:,2], color='blue', label='b')
        # plt.plot(z_data[:,0]/z_data[:,2],linestyle='-.',color='red', label='r/b')
        # plt.plot(z_data[:,0]/z_data[:,1],linestyle='-.',color='green', label='r/g')
        # plt.plot(z_data[:,1]/z_data[:,2],linestyle='-.',color='blue', label='g/b')
        # plt.title('去趋势')
        # plt.xlabel('Labels')
        # plt.ylabel('Predictions')
        # plt.legend()
        # plt.show()
        # c_data = self.normalize(c_data)
        # plt.plot(c_data[:,0]/c_data[:,1], label='r/g')
        # plt.plot(c_data[:,0]/c_data[:,2], label='r/b')
        # plt.plot(c_data[:,1]/c_data[:,0], label='g/r')
        # plt.plot(c_data[:,0]-c_data[:,1], label='r-g')
        # plt.plot(c_data[:,0]-c_data[:,2], label='r-b')
        # plt.plot(c_data[:,1]-c_data[:,2], label='g-r')
        # plt.plot(c_data[:,1]/c_data[:,2], label='g/b')
        # plt.plot(c_data[:,2]/c_data[:,0], label='b/r')
        # plt.plot(c_data[:,2]/c_data[:,1], label='b/g')
        # plt.title('Scatter Plot of Labels vs Predictions')
        # plt.xlabel('Labels')
        # plt.ylabel('Predictions')
        # plt.legend()
        # plt.show()

        # 进行fastICA
        # c_data_ICA = self.fastICA_data(c_data_fft, 36)
        # if i_data is not None:
        #     # 进行滤波处理
        #     i_data = self.fft_data(i_data, 12, low = 0.5,high = 4)
        #     i_data = self.normalize(i_data)
        #     i_data = self.detrend(i_data, 12, order=1)
        #     # # 进行fastICA
        #     # i_data_ICA = self.fastICA_data(i_data_fft, 12)
        # else:
        #     i_data = None
        return ci_data,cidc_data,GT_SPO2
        #return c_data_fft,i_data_fft

    def 处理数据并保存(self,文件路径,output_path):
        output_data = output_path + "/预处理过的rPPG.csv"
        output_dc_png = output_path + "/预处理过的DC与血氧的折线图.png"
        output_o_png = output_path + "/0预处理过的原rPPG折线图.png"
        output_odc_png = output_path + "/0预处理过的原dc折线图.png"
        output_spo2_png = output_path + "/0预处理过的原血氧折线图.png"
        self.output_path = output_path
        # 这里开关是否添加数据增强的数据
        if not 'none' in output_data:
            print("不存在none")
            return 0
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("进行数据处理")
        ci_data_inter,GT_SPO2_inter,HR_inter = self.读取数据(文件路径)
        # S = 1800
        # D = 1800
        # for i in range((c_data_inter.shape[0]-D)//S + 1):
        #     c_data_切片,i_data_切片 = c_data_inter[i*S:i*S+D],c_data_inter[i*S:i*S+D]
        #     c_data,i_data,d,dc_data,idc_data,两者不同 = self.process(c_data_切片,i_data_切片)#d
        ci_data,cidc_data,GT_SPO2 = self.process(ci_data_inter,GT_SPO2_inter,HR_inter)#d
        # GT_SPO2 = GT_SPO2.reshape(-1, 1)[d//2:-1*(d//2)]
        # GT_SPO2_inter = GT_SPO2_inter[:120*(GT_SPO2_inter.shape[0]//120),:]
        # ci_data = ci_data[150:,:]
        # cidc_data = cidc_data[150:,:]
        # GT_SPO2 = GT_SPO2[:-150]
        GT_SPO2 = GT_SPO2.reshape(-1, 1)
        print("DC数据形状")
        print(cidc_data.shape)
        print("可见光数据形状")
        print(ci_data.shape)
        # if i_data is not None:
        #     print("iDC数据形状")
        #     print(idc_data.shape)
        #     print("红外光数据形状")
        #     print(i_data.shape)
        print("真实SPO2的数据形状")
        print(GT_SPO2.shape)
        
        GT_SPO2_copy = GT_SPO2.copy()
        cidc_data_copy = cidc_data.copy()

        plt.figure(figsize=(15, 10))
        plt.plot(self.snormalize(self.基线计算2(GT_SPO2_copy,win=400)), label='spo2')
        GT_SPO2_copy = GT_SPO2.copy()
        cof_r = []
        if ci_data.shape[1] >= 24:
            for i in range(24):
                # cidc_data[:,i] = self.snormalize(cidc_data[:,i])
                相关系数 = self.calculate_pearson_correlation_coefficient(cidc_data_copy[:,i],self.snormalize((self.基线计算2(GT_SPO2_copy,win=400))+0.1))
                plt.plot(self.snormalize(cidc_data_copy[:,i]), label=f'{i}:{相关系数:02f}')
                cof_r.append(相关系数)
                cidc_data_copy = cidc_data.copy()
                GT_SPO2_copy = GT_SPO2.copy()
        else:
            for i in range(12):
                # cidc_data[:,i] = self.snormalize(cidc_data[:,i])
                相关系数 = self.calculate_pearson_correlation_coefficient(cidc_data_copy[:,i],self.snormalize((self.基线计算2(GT_SPO2_copy,win=400))+0.1))
                plt.plot(self.snormalize(cidc_data_copy[:,i]), label=f'{i}:{相关系数:02f}')
                cof_r.append(相关系数)
                cidc_data_copy = cidc_data.copy()
                GT_SPO2_copy = GT_SPO2.copy()
        # cof_r = np.array(cof_r)
        # cof_g = np.array(cof_g)
        # cof_b = np.array(cof_b)
        cof_r_abs = np.abs(cof_r)
        plt.title(f'DC与标签相关系数{np.mean(cof_r_abs):.4f} +- {np.std(cof_r_abs):.4f}')
        plt.legend()
        plt.savefig(output_dc_png)
        # plt.show()
        plt.close()
        ROI = [
            '左脸颊','右脸颊','额头','全脸']
        if ci_data.shape[1] >= 24:
            for i in range(4):
                plt.figure(figsize=(15, 10))
                plt.plot(self.snormalize(cidc_data_copy[:,i*4]), color = 'r', label=f'{i*4}:{cof_r[i*4]:02f}')
                plt.plot(self.snormalize(cidc_data_copy[:,i*4+1]), color = 'g', label=f'{i*4+1}:{cof_r[i*4+1]:02f}')
                plt.plot(self.snormalize(cidc_data_copy[:,i*4+2]), color = 'b', label=f'{i*4+2}:{cof_r[i*4+2]:02f}')
                plt.plot(self.snormalize(cidc_data_copy[:,i*4+3]), color = 'gray', label=f'{i*4+3}:{cof_r[i*4+3]:02f}')
                plt.plot(self.snormalize(self.基线计算2(GT_SPO2_copy,win=400)), label='spo2')
                GT_SPO2_copy = GT_SPO2.copy()
                plt.title(f'{ROI[i]}DC与血氧标签相关系数')
                plt.legend()
                plt.savefig(output_path + f"/{ROI[i]}DC与血氧标签的折线图.png")
                # plt.show()
                plt.close()
        else:
            for i in range(4):
                plt.figure(figsize=(15, 10))
                plt.plot(self.snormalize(cidc_data_copy[:,i*3]), color = 'r', label=f'{i*3}:{cof_r[i*3]:02f}')
                plt.plot(self.snormalize(cidc_data_copy[:,i*3+1]), color = 'g', label=f'{i*3+1}:{cof_r[i*3+1]:02f}')
                plt.plot(self.snormalize(cidc_data_copy[:,i*3+2]), color = 'b', label=f'{i*3+2}:{cof_r[i*3+2]:02f}')
                plt.plot(self.snormalize(self.基线计算2(GT_SPO2_copy,win=400)), label='spo2')
                GT_SPO2_copy = GT_SPO2.copy()
                plt.title(f'{ROI[i]}DC与血氧标签相关系数')
                plt.legend()
                plt.savefig(output_path + f"/{ROI[i]}DC与血氧标签的折线图.png")
                # plt.show()
                plt.close()


        
        plt.plot(ci_data[:,0], color='red', label='r')
        plt.plot(ci_data[:,1], color='green', label='g')
        plt.plot(ci_data[:,2], color='blue', label='b')
        if ci_data.shape[1] >= 24:
            plt.plot(ci_data[:,3], color='gray', label='i')
        plt.title('data')
        plt.xlabel('Labels')
        plt.ylabel('Predictions')
        plt.legend()
        plt.savefig(output_o_png)
        plt.close()
        plt.show()

        plt.plot(cidc_data[:,0], color='red', label='r')
        plt.plot(cidc_data[:,1], color='green', label='g')
        plt.plot(cidc_data[:,2], color='blue', label='b')
        if ci_data.shape[1] >= 24:
            plt.plot(ci_data[:,3], color='gray', label='i')
        plt.title('data')
        plt.xlabel('Labels')
        plt.ylabel('Predictions')
        plt.legend()
        plt.savefig(output_odc_png)
        plt.close()
        plt.show()

        plt.plot(GT_SPO2[:,0], color='red', label='r')
        plt.title('data')
        plt.xlabel('Labels')
        plt.ylabel('Predictions')
        plt.legend()
        plt.savefig(output_spo2_png)
        plt.close()
        plt.show()

        R_all = np.zeros((1,24))
        # IR_all = np.zeros((1,36))
        R_IR_all = np.zeros((1,48))
        SPO2_all = np.zeros((1,1))
        
        SPO2_all = np.concatenate((SPO2_all,GT_SPO2),axis=0)
                        
        if ci_data.shape[1] >= 24:            
            all_segment = np.concatenate((ci_data, cidc_data), axis=1)
            R_IR_all = np.concatenate((R_IR_all, all_segment), axis=0)
            data_all = np.concatenate((R_IR_all,SPO2_all),axis=1)
        else:
            cdc_data = np.concatenate((ci_data,ci_data),axis=1)
            R_all = np.concatenate((R_all,cdc_data),axis=0)
            data_all = np.concatenate((R_all,SPO2_all),axis=1)    

        df1 = pd.DataFrame(data_all[1:])  
        df1.to_csv(output_data,index=False)
    
    def getdata(self):   
        # 遍历当前目录及其所有子目录  
        for root, dirs, files in os.walk(self.主文件夹):  
            for file in files:  
                if file == f'{self.文件名}':  
                    # 构造完整的文件路径  
                    粗制rPPG文件路径 = os.path.join(root, file)  
                    print(粗制rPPG文件路径)
                    # 相对于主文件夹的相对路径  
                    相对路径 = os.path.relpath(root, self.主文件夹)  
        
                    # 在输出路径中构建相应的目录  
                    output_subdir = os.path.join(self.输出路径, 相对路径)  
                    if not os.path.exists(output_subdir):  
                        os.makedirs(output_subdir) 

                    # 数据增强方法 = ['缩放画面','调整亮度','调整对比度','调整饱和度','添加噪声','调整色调','旋转画面','水平翻转']
                    
        
                    if os.path.exists(粗制rPPG文件路径):
                        输出的文件 = os.path.join(output_subdir, '预处理过的rPPG.csv')
                        if os.path.exists(输出的文件):
                            print(f'{输出的文件}已经存在')
                            continue
                        self.处理数据并保存(粗制rPPG文件路径,output_subdir)
                        # time.sleep(1000)
                        # if 两者心率计算不同:
                        #     记录log(f'{相对路径} 两者心率计算不一样(上面显示)',self.输出路径)

                        # for i in range(len(数据增强方法)):
                        #     if 数据增强方法[i] == '调整亮度':
                        #         for j in range(-50,51,10):
                        #             self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],j,output_path)
                        #     if 数据增强方法[i] == '调整饱和度':
                        #         for j in range(-50,51,10):
                        #             self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],j,output_path)
                        #     if 数据增强方法[i] == '调整对比度':
                        #         for j in [0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.4,1.8,2.3,3]:
                        #             self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],j,output_path)
                        #     if 数据增强方法[i] == '添加噪声':
                        #         for j in [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]:
                        #             self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],j,output_path)
                        #     if 数据增强方法[i] == '调整色调':
                        #         for j in [-30,-25,-20,-15,-10,-8,-6,-4,-2,2,4,6,8,10,15,20,25,30]:
                        #             self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],j,output_path)
                        #     if 数据增强方法[i] == '旋转画面':
                        #         for j in [-45,-30,-20,-10,10,20,30,45]:
                        #             self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],j,output_path)
                        #     if 数据增强方法[i] == '水平翻转':
                        #         self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],0,output_path)
                        #     if 数据增强方法[i] == '缩放画面':
                        #         for j in [0.5,0.4,0.3,0.2,0.1]:
                        #             self.处理数据并保存(color_path_signal,ir_path_signal,path_SPO2,数据增强方法[i],j,output_path)
                    else:
                        print("输入文件不存在")
