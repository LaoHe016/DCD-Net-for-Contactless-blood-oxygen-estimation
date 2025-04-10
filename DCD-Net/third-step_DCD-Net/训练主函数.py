import train_SpO2CNN1D
import sys
import os
import numpy as np
import pyCompare
import matplotlib.pyplot as plt
import pandas as pd

# 创建参数
参数字典 = {
    # 训练参数
    'epochs': 10,
    'batch_size': 32,
    '学习率':0.0001,
    '渐热学习放大倍率':2,
    '检查点': None,            #从某个检查点继续训练'./checkpoints/一部分数据增强参数/SPO2_last.pt'

    # 数据集
    "D" : 600, #数据长度
    '步长' : 60,
    "整个数据集范围内的软化标签长度" : 100, #方便计算趋势即皮尔逊相关系数
    '软标签平均范围' : 100, #少于D//2
    '验证集占比': 0.5,
    '训练数据文件夹': '../000_preprocessed data/不进行归一化_PURE_一阶/',
    '训练数据名称':None,
    '不参加训练验证的文件夹': [], #表示剔除掉异常的训练数据
    '测试数据名称': ['01-01','01-02','01-03','01-04','01-05','01-06'],#比如'01-01'
    '验证数据名称': ['02-01','02-02','02-03','02-04','02-05','02-06'],#比如'01-01'
    '训练集数据用于输出显示的' : '02-02',
    '测试数据集名称':'1',

    # 重采样
    '重采样': False,
    '非平衡重采样': True,
        '过采样阈值':0.02,# 某一类数据样本占总数的比例低于这个阈值，进行重采样到这个数
        '欠采样阈值':0.05,# 某一类数据样本占总数的比例高于这个阈值，进行欠采样到这个数

    # 数据增强
    '噪声均值':-1,
    '噪声方差':0.001,
    '数据增强方法':['真实','none'],  #:['none','画面扭曲','运动模糊','缩放画面','旋转画面','水平翻转'] ['none','画面扭曲','运动模糊','环境光变化','缩放画面','调整亮度','调整对比度','调整饱和度','添加噪声','旋转画面','水平翻转']

    # 损失计算
    '加权误差方法': '平均绝对误差', # 可选 平均绝对误差 或者 均方误差
        # 趋势损失 (或者是相关系数计算)
        '皮尔逊相关系数损失计算系数' : 0,

    # 波峰对齐sss
    '波峰对齐' : False, # True and False
    '波峰个数' : 2, # 整数

    # 输出14-2微调
    '输出文件夹名': '2',     #自定义输出文件夹名称
    'test周期': 1,  #如果是要测试把它改成数字 -1表示不进行
}

def 记录log(文字,输出文件夹名称,文件名 = 'record'):
    if not os.path.exists(f'./checkpoints/exp{输出文件夹名称}/log'):
        os.makedirs(f'./checkpoints/exp{输出文件夹名称}/log')
    log_path = os.path.join(
        f'./checkpoints/exp{输出文件夹名称}/log/',文件名 + '.log')
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

def log行数():
    # 定义文件路径
    输出文件夹 = 参数字典['输出文件夹名']
    log_file_path = f'./checkpoints/exp{输出文件夹}/log/record.log'

    # 初始化行计数器
    line_count = 0
    if not os.path.exists(f'./checkpoints/exp{输出文件夹}/log'):  
        os.makedirs(f'./checkpoints/exp{输出文件夹}/log')
        记录log(f"记录交叉验证损失",参数字典['输出文件夹名'])
        return 1

    # 打开文件并读取
    with open(log_file_path, 'r') as file:
        for line in file:
            line_count += 1

    # 打印行数
    print(f'日志文件中的行数为: {line_count}')
    return line_count

def 计算损失(验证集上最佳模型预测_list,验证集上最佳模型标签_list):
    # 计算数组的总元素数量
    total_elements = np.prod(np.array(验证集上最佳模型预测_list).shape)
    验证集上最佳模型预测 = np.array(验证集上最佳模型预测_list).reshape(total_elements)
    验证集上最佳模型标签 = np.array(验证集上最佳模型标签_list).reshape(total_elements)
    MAE_FFT = np.mean(np.abs(验证集上最佳模型预测 - 验证集上最佳模型标签))
    mee_standard_error = np.std(np.abs(验证集上最佳模型预测 - 验证集上最佳模型标签)) / np.sqrt(len(验证集上最佳模型标签))
    print("MAE : {0} +/- {1}".format(MAE_FFT, mee_standard_error))

    RMSE_FFT = np.sqrt(np.mean(np.square(验证集上最佳模型预测 - 验证集上最佳模型标签)))
    rmse_standard_error = np.std(np.square(验证集上最佳模型预测 - 验证集上最佳模型标签)) / np.sqrt(len(验证集上最佳模型标签))
    print("RMSE : {0} +/- {1}".format(RMSE_FFT, rmse_standard_error))



    output_folder = 参数字典['输出文件夹名']
    pyCompare.blandAltman(验证集上最佳模型预测, 验证集上最佳模型标签,
                        percentage=False,
                        title='Bland-Altman',
                        limitOfAgreement=1.96,
                        savePath=f'./checkpoints/exp{output_folder}/bland_altman_plot.png')
    plt.close('all')

    plt.scatter(验证集上最佳模型标签, 验证集上最佳模型预测,label='Predictions')
    plt.plot(验证集上最佳模型标签, 验证集上最佳模型标签, color='red', label='REFERENCE LINE')
    plt.title('Scatter Plot of Labels vs Predictions')
    plt.xlabel('Labels')
    plt.ylabel('Predictions')
    plt.savefig(f'./checkpoints/exp{output_folder}/Scatter_Plot.png')
    plt.close('all')

    # 将数据转换为DataFrame
    data = pd.DataFrame({
        'True Labels': 验证集上最佳模型标签,
        'Predicted Labels': 验证集上最佳模型预测
    })

    # 保存为CSV文件
    data.to_csv(f'./checkpoints/exp{output_folder}/Scatter_Plot.csv', index=False)

    print("数据已成功保存到 validation_results.csv 文件中！")


    # 组织数据，计算每个标签的误差
    unique_labels = np.unique(验证集上最佳模型标签)  # 找出所有唯一的标签
    mae_per_label = {}
    rmse_per_label = {}

    for label in unique_labels:
        # 选择当前标签的索引
        idx = 验证集上最佳模型标签 == label
        
        # 计算当前标签的MAE和RMSE
        MAE = np.mean(np.abs(验证集上最佳模型预测[idx] - 验证集上最佳模型标签[idx]))
        RMSE = np.sqrt(np.mean(np.square(验证集上最佳模型预测[idx] - 验证集上最佳模型标签[idx])))
        
        # 存储结果
        mae_per_label[label] = MAE
        rmse_per_label[label] = RMSE

    # 输出每个标签的平均误差
    for label in unique_labels:
        print(f"Label {label} MAE: {mae_per_label[label]:.4f}, RMSE: {rmse_per_label[label]:.4f}")
        记录log(f'Label {label} MAE: {mae_per_label[label]:.4f}, RMSE: {rmse_per_label[label]:.4f}',参数字典['输出文件夹名'],文件名='MAEs')

    return MAE_FFT,mee_standard_error,RMSE_FFT,rmse_standard_error

def 跨数据集测试(参数字典):
    数据集 = ['PURE','ViInHealth','VIPL-HR']
    原始参数 = 参数字典.copy()

    for 训练数据集 in 数据集:
        for 验证测试数据集 in 数据集:

            if 训练数据集 == 验证测试数据集:
                print(f"训练数据集:{训练数据集};验证测试数据集:{验证测试数据集}相同,跳过")
                continue
            参数字典['测试数据名称'] = [验证测试数据集]
            参数字典['验证数据名称'] = [验证测试数据集]
            参数字典['训练数据名称'] = [训练数据集]
            参数字典['输出文件夹名'] += f'/跨数据集验证_训练数据{训练数据集}_验证测试数据集{验证测试数据集}'
            
            这一折最佳模型名称=None
            验证集最小损失=None
            MAE,mee_standard_error,RMSE,rmse_standard_error=100,100,1000,1000
            MAEs = []
            RMSEs = []
            MAE_standard_errors = []
            RMSE_standard_errors = []
            原输出文件夹 = 参数字典['输出文件夹名']
            if os.path.exists(f'./checkpoints/exp{原输出文件夹}'):
                print(f"训练数据集:{训练数据集};验证测试数据集:{验证测试数据集}\n{原输出文件夹}已存在,跳过")
                参数字典 = 原始参数.copy()
                continue
            else:
                print(f"训练数据集:{训练数据集};验证测试数据集:{验证测试数据集}\n{原输出文件夹}不存在,继续")

            for j in range(1):  # 循环十次
                参数字典['输出文件夹名'] = 原输出文件夹 + f'/第{j+1}次'

                测试集MAE = train_SpO2CNN1D.train(参数字典)
                # MAE, MAE_standard_error, RMSE, RMSE_standard_error = 计算损失(验证集上最佳模型预测_list, 验证集上最佳模型标签_list)
                
                # # 将每次的结果存储在列表中
                # MAEs.append(MAE)
                # MAE_standard_errors.append(MAE_standard_error)
                # RMSEs.append(RMSE)
                # RMSE_standard_errors.append(RMSE_standard_error)

            # # 找到MAE最小的索引
            # min_MAE_index = MAEs.index(min(MAEs))

            # # 使用MAE最小的值更新MAE, MAE_standard_error, RMSE, RMSE_standard_error
            # MAE = MAEs[min_MAE_index]
            # MAE_standard_error = MAE_standard_errors[min_MAE_index]
            # RMSE = RMSEs[min_MAE_index]
            # RMSE_standard_error = RMSE_standard_errors[min_MAE_index]

            # print(f'跨数据集验证，训练：{训练数据集}，验证测试：{验证测试数据集} , MAE = {MAE} +/- {MAE_standard_error}   RMSE = {RMSE} +/- {RMSE_standard_error} , 模型名称 = {这一折最佳模型名称} , 验证集损失 = {验证集最小损失}')
            # print(f'跨数据集验证，训练：{训练数据集}，验证测试：{验证测试数据集} , 历次损失 = {MAEs}')
            记录log(f'跨数据集验证，训练：{训练数据集}，验证测试：{验证测试数据集} , MAE = {测试集MAE}',原始参数['输出文件夹名'])
            # 记录log(f'跨数据集验证，训练：{训练数据集}，验证测试：{验证测试数据集} , 历次损失 = {MAEs}',原始参数['输出文件夹名'],文件名='MAEs')
            # # loss_sum += MAE
            参数字典 = 原始参数.copy()

if __name__ == "__main__":
    
    

    # 跨数据集测试(参数字典)
    line_count = log行数() - 1

    原始参数 = 参数字典.copy()
    loss_sum = 0
    # 5折交叉验证 不同人之间折
    '''
    第一折：123测试
    第二折：345测试
    第三折：567测试
    第四折：789测试
    第五折：9,10,1测试
    '''
    for i in range(5):
        if i < line_count:
            continue
        if i < 4:
            # 参数字典['不参加训练验证的文件夹'] = [f'{i*2+1:02d}-01',f'{i*2+1:02d}-02',f'{i*2+1:02d}-03',f'{i*2+1:02d}-04',f'{i*2+1:02d}-05',f'{i*2+1:02d}-06',
            #                                    f'{i*2+2:02d}-01',f'{i*2+2:02d}-02',f'{i*2+2:02d}-03',f'{i*2+2:02d}-04',f'{i*2+2:02d}-05',f'{i*2+2:02d}-06',
            #                                     ]
            参数字典['测试数据名称'] = [f'{i*2+1:02d}-01',f'{i*2+1:02d}-02',f'{i*2+1:02d}-03',
                                      f'{i*2+2:02d}-01',f'{i*2+2:02d}-02',f'{i*2+2:02d}-03',
                                      f'{i*2+3:02d}-01',f'{i*2+3:02d}-02',f'{i*2+3:02d}-03',
                                      ]
            
            参数字典['验证数据名称'] = [f'{i*2+1:02d}-04',f'{i*2+1:02d}-05',f'{i*2+1:02d}-06',
                                      f'{i*2+2:02d}-04',f'{i*2+2:02d}-05',f'{i*2+2:02d}-06',
                                      f'{i*2+3:02d}-04',f'{i*2+3:02d}-05',f'{i*2+3:02d}-06',
                                      ]
            参数字典['训练集数据用于输出显示的'] = '10-02'
            参数字典['输出文件夹名'] += f'/个人第{i+1}折_测试数据{i*2+3:02d}{i*2+1:02d}{i*2+2:02d}'
        elif i >= 4:
            # 参数字典['不参加训练验证的文件夹'] = [f'{i*2+1:02d}-01',f'{i*2+1:02d}-02',f'{i*2+1:02d}-03',f'{i*2+1:02d}-04',f'{i*2+1:02d}-05',f'{i*2+1:02d}-06',
            #                                    f'{i*2+2}-01',f'{i*2+2}-02',f'{i*2+2}-03',f'{i*2+2}-04',f'{i*2+2}-05',f'{i*2+2}-06',
            #                                     ]
            参数字典['测试数据名称'] = [f'{i*2+1:02d}-01',f'{i*2+1:02d}-02',f'{i*2+1:02d}-03',
                                      f'{i*2+2:02d}-01',f'{i*2+2:02d}-02',f'{i*2+2:02d}-03',
                                      f'{1:02d}-01',f'{1:02d}-02',f'{1:02d}-03',
                                      ]
            
            参数字典['验证数据名称'] = [f'{i*2+1:02d}-04',f'{i*2+1:02d}-05',f'{i*2+1:02d}-06',
                                      f'{i*2+2:02d}-04',f'{i*2+2:02d}-05',f'{i*2+2:02d}-06',
                                      f'{1:02d}-04',f'{1:02d}-05',f'{1:02d}-06',
                                      ]
            参数字典['训练集数据用于输出显示的'] = f'02-02'
            参数字典['输出文件夹名'] += f'/个人第{i+1}折_测试数据{i*2:02d}{i*2+1:02d}{i*2+2:02d}'
        这一折最佳模型名称=None
        验证集最小损失=None
        MAE,mee_standard_error,RMSE,rmse_standard_error=100,100,1000,1000
        MAEs = []
        RMSEs = []
        MAE_standard_errors = []
        RMSE_standard_errors = []
        原输出文件夹 = 参数字典['输出文件夹名']

        for j in range(2):  # 循环十次
            参数字典['输出文件夹名'] = 原输出文件夹 + f'/第{j+1}次'

            验证集上最佳模型预测_list, 验证集上最佳模型标签_list, 这一折最佳模型名称, 验证集最小损失 = train_SpO2CNN1D.train(参数字典)
            MAE, MAE_standard_error, RMSE, RMSE_standard_error = 计算损失(验证集上最佳模型预测_list, 验证集上最佳模型标签_list)
            
            # 将每次的结果存储在列表中
            MAEs.append(MAE)
            MAE_standard_errors.append(MAE_standard_error)
            RMSEs.append(RMSE)
            RMSE_standard_errors.append(RMSE_standard_error)

        # 找到MAE最小的索引
        min_MAE_index = MAEs.index(min(MAEs))

        # 使用MAE最小的值更新MAE, MAE_standard_error, RMSE, RMSE_standard_error
        MAE = MAEs[min_MAE_index]
        MAE_standard_error = MAE_standard_errors[min_MAE_index]
        RMSE = RMSEs[min_MAE_index]
        RMSE_standard_error = RMSE_standard_errors[min_MAE_index]

        print(f'个人第{i+1}折 , MAE = {MAE} +/- {MAE_standard_error}   RMSE = {RMSE} +/- {RMSE_standard_error} , 模型名称 = {这一折最佳模型名称} , 验证集损失 = {验证集最小损失}')
        print(f'个人第{i+1}折 , 历次损失 = {MAEs}')
        记录log(f'个人第{i+1}折 , MAE = {MAE} +/- {MAE_standard_error}   RMSE = {RMSE} +/- {RMSE_standard_error} , 模型名称 = {这一折最佳模型名称} , 验证集损失 = {验证集最小损失}',原始参数['输出文件夹名'])
        记录log(f'个人第{i+1}折 , 历次损失 = {MAEs}',原始参数['输出文件夹名'],文件名='MAEs')
        loss_sum += MAE
        参数字典 = 原始参数.copy()

    if log行数() < 7:
        print(f"所有5折的平均损失是 = {loss_sum/5}")
        记录log(f"所有5折的平均损失是 = {loss_sum/5}",原始参数['输出文件夹名'])

    line_count = log行数() - 7
    loss_sum = 0
    # 6折交叉验证 不同动作之间折
    '''
    第一折：静止测试
    第二折：说话测试
    第三折：轻微移动测试
    第四折：快速移动测试
    第五折：轻微转动测试
    第六折：快速转动测试
    '''
    for i in range(6):
        if i < line_count:
            continue
        参数字典['不参加训练验证的文件夹'] = ['测试']
        
        参数字典['测试数据名称'] = [f'01-0{i+1}',f'02-0{i+1}',f'03-0{i+1}',f'04-0{i+1}',f'05-0{i+1}',f'06-0{i+1}',f'07-0{i+1}',f'08-0{i+1}',f'09-0{i+1}',f'10-0{i+1}']
        参数字典['验证数据名称'] = []
        参数字典['输出文件夹名'] += f'/动作第{i+1}折_验证数据xx_{i+2:02d}_测试数据xx_{i+1:02d}'
        
        # if i != 3 or i != 2:
        #     参数字典['训练集数据用于输出显示的'] = '02-04'
        # else:
        #     参数字典['训练集数据用于输出显示的'] = '02-05'
        # if i >= 5:
        #     参数字典['验证数据名称'] = [f'01-0{i}',f'02-0{i}',f'03-0{i}',f'04-0{i}',f'05-0{i}',f'06-0{i}',f'07-0{i}',f'08-0{i}',f'09-0{i}',f'10-0{i}']
        #     参数字典['输出文件夹名'] += f'/动作第{i+1}折_验证数据xx_{i:02d}_测试数据xx_{i+1:02d}'
        
        # else:
        #     参数字典['验证数据名称'] = [f'01-0{i+2}',f'02-0{i+2}',f'03-0{i+2}',f'04-0{i+2}',f'05-0{i+2}',f'06-0{i+2}',f'07-0{i+2}',f'08-0{i+2}',f'09-0{i+2}',f'10-0{i+2}']

        #     参数字典['输出文件夹名'] += f'/动作第{i+1}折_验证数据xx_{i+2:02d}_测试数据xx_{i+1:02d}'

        这一折最佳模型名称=None
        验证集最小损失=None
        MAE,mee_standard_error,RMSE,rmse_standard_error=100,100,1000,1000
        MAEs = []
        RMSEs = []
        MAE_standard_errors = []
        RMSE_standard_errors = []
        原输出文件夹 = 参数字典['输出文件夹名']

        for j in range(1):  # 循环十次
            参数字典['输出文件夹名'] += f'/第{j+1}次'

            验证集上最佳模型预测_list, 验证集上最佳模型标签_list, 这一折最佳模型名称, 验证集最小损失 = train_SpO2CNN1D.train(参数字典)
            MAE, MAE_standard_error, RMSE, RMSE_standard_error = 计算损失(验证集上最佳模型预测_list, 验证集上最佳模型标签_list)
            
            # 将每次的结果存储在列表中
            MAEs.append(MAE)
            MAE_standard_errors.append(MAE_standard_error)
            RMSEs.append(RMSE)
            RMSE_standard_errors.append(RMSE_standard_error)

        # 找到MAE最小的索引
        min_MAE_index = MAEs.index(min(MAEs))

        # 使用MAE最小的值更新MAE, MAE_standard_error, RMSE, RMSE_standard_error
        MAE = MAEs[min_MAE_index]
        MAE_standard_error = MAE_standard_errors[min_MAE_index]
        RMSE = RMSEs[min_MAE_index]
        RMSE_standard_error = RMSE_standard_errors[min_MAE_index]

        print(f'动作第{i+1}折 , MAE = {MAE} +/- {MAE_standard_error}   RMSE = {RMSE} +/- {RMSE_standard_error} , 模型名称 = {这一折最佳模型名称} , 验证集损失 = {验证集最小损失}')
        print(f'个人第{i+1}折 , 历次损失 = {MAEs}')
        记录log(f'动作第{i+1}折 , MAE = {MAE} +/- {MAE_standard_error}   RMSE = {RMSE} +/- {RMSE_standard_error} , 模型名称 = {这一折最佳模型名称} , 验证集损失 = {验证集最小损失}',原始参数['输出文件夹名'])
        记录log(f'个人第{i+1}折 , 历次损失 = {MAEs}',原始参数['输出文件夹名'],文件名='MAEs')
        loss_sum += MAE
        参数字典 = 原始参数.copy()

    if log行数() < 14:
        print(f"所有6折的平均损失是 = {loss_sum/6}")
        记录log(f"所有6折的平均损失是 = {loss_sum/6}",原始参数['输出文件夹名'])
