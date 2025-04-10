from 预处理流程 import 预处理

if __name__ == "__main__":
    '''
    main_path = "./data/self/"
    color_path_signal = "./data/self/"
    color_path_time = "./data/self/"
    ir_path_signal = "./data/self/"
    ir_path_time = "./data/self/"
    path_GT = "./data/self/"
    path_SPO2 = "./data/self/"
    output_path = "./data/output/"
    '''
    # 输入数据主文件夹 = "./data/self/"
    # 输入数据主文件夹 = "F:/Face_rPPG_Get/预处理/HYXTest_1DCNN+A/视频处理完输出的信号-原始信号/"
    输入数据主文件夹 = "../0-0、1输出的数据/实例测试/"
    文件名 = "获取的rPPG.csv"
    # 输出文件夹路径 = "./data/output/"
    输出文件夹路径 = "../0-1、预处理后的数据/30实例测试/"
    
    processor = 预处理(输入数据主文件夹,文件名, 输出文件夹路径)
    processor.getdata()