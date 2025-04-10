from Preprocess import Preprocessing

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
    输入数据主文件夹 = "F:/Face_rPPG_Get/Dataset/VIPL-HR/" # 存放数据的主文件夹
    可见光视频路径 = "/video.avi" # 可见光视频的文件名
    红外光视频路径 = "/video_ir11111.mp4"
    血氧参考值文件路径 = "/gt_SpO2.csv"     # 血氧值文件的文件名
    输出文件夹路径 = "../0-0、1输出的数据/4ROI/VIPL-HR/" # 数据输出文件夹
    
    processor = Preprocessing(输入数据主文件夹, 可见光视频路径, 红外光视频路径, 血氧参考值文件路径, 输出文件夹路径)
    processor.getdata()