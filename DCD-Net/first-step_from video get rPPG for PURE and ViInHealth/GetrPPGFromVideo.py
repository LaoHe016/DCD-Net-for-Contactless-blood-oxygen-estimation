import cv2
import time
from tqdm import tqdm  
import pandas as pd
from FaceMeshDetector import FaceMeshDetector
import numpy as np
import os
import imgaug.augmenters as iaa

上次亮度 = 0

def 直方图均衡化(image):
    # print("直方图均衡化！")
    # 将图像分为三个颜色通道
    b, g, r = cv2.split(image)

    # 单独对每个通道进行直方图均衡化
    b_histeq = cv2.equalizeHist(b)
    g_histeq = cv2.equalizeHist(g)
    r_histeq = cv2.equalizeHist(r)

    # 合并均衡化后的通道
    equalized_image = cv2.merge((b_histeq, g_histeq, r_histeq))

    return equalized_image

def 运动模糊(图像,参数):
    seq = iaa.Sequential([
        iaa.MotionBlur(k = 参数[0],angle = 参数[1])
    ])
    输出图像 = seq(image = 图像)
    return 输出图像

def 画面扭曲(图像,参数):
    seq = iaa.Sequential([
        iaa.PiecewiseAffine(scale=(参数[0], 参数[1]))
    ])
    输出图像 = seq(image = 图像)
    return 输出图像

def color_transform(frame, brightness_delta, contrast_delta, saturation_delta):
    # 将BGR图像转换为HSV图像
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 调整亮度
    v = np.clip(v + brightness_delta, 0, 255).astype(np.uint8)

    # 调整对比度
    v = np.clip((v - 128) * contrast_delta + 128, 0, 255).astype(np.uint8)

    # 调整饱和度
    s = np.clip(s + saturation_delta, 0, 255).astype(np.uint8)

    # 合并通道并转换回BGR颜色空间
    final_hsv = cv2.merge((h, s, v))
    img_color = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img_color

def add_noise(frame, noise_intensity):
    noise = np.random.randn(*frame.shape) * 255 * noise_intensity
    noisy_frame = frame + noise.astype(np.uint8)
    noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
    return noisy_frame

def hue_transform(frame, hue_delta):
    # 色调变换
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 调整色调
    h = (h.astype(int) + hue_delta) % 180  # 确保色调值在 0 到 179 之间
    h = h.astype(np.uint8)

    # 合并通道并转换回BGR颜色空间
    final_hsv = cv2.merge((h, s, v))
    img_hue = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img_hue

def data_augmentation(frame, method, params):
    # print(f"数据增强方法：{method},参数：{params}")
    augmented_frame = frame
    if method == '调整亮度':
        augmented_frame = color_transform(frame, params, 1, 0)
    elif method == '环境光变化':
        augmented_frame = color_transform(frame, params, 1, 0)
    elif method == '画面扭曲':
        augmented_frame = 画面扭曲(frame, params)
    elif method == '调整对比度':
        augmented_frame = color_transform(frame, 0, params, 0)
    elif method == '调整饱和度':
        augmented_frame = color_transform(frame, 0, 1, params)
    elif method == '添加噪声':
        augmented_frame = add_noise(frame, params)
    elif method == '调整色调':
        augmented_frame = hue_transform(frame, params)
    elif method == '旋转画面':
        # 旋转视频帧
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        angle = params
        rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented_frame = cv2.warpAffine(frame, rotation_mat, frame.shape[1::-1], flags=cv2.INTER_LINEAR)
    elif method == '水平翻转':
        # 水平翻转视频帧
        augmented_frame = cv2.flip(frame, 1)
    elif method == '缩放画面':
        # 计算新的宽度和高度
        new_width = params
        new_height = int(frame.shape[0])

        # 使用cv2.resize进行缩放
        augmented_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    elif method == '运动模糊':
        augmented_frame = 运动模糊(frame,params)
    else:
        augmented_frame = frame
    return augmented_frame
    
def 计算角度(index,参数):
    if 参数 < 0:
        if index // (-2*参数) % 2 == 0:
            return 参数+index%(-2*参数)
        else:
            return -1*参数-index%(-2*参数)

    else:
        if 2*index // (2*参数) % 2 == 0:
            return 参数-(2*index)%(2*参数)
        else:
            return -1*参数+(2*index)%(2*参数)
        
def 计算宽度(index,参数,原宽度):
    新目标宽度 = int(原宽度 * 参数)
    if index // (原宽度 - 新目标宽度) % 2 == 0:
        return 原宽度-index%(原宽度-新目标宽度)
    else:
        return 新目标宽度+index%(原宽度-新目标宽度)

def 计算亮度(参数):
    global 上次亮度
    变化 = np.random.randint(low=-1*参数, high=参数+1)
    亮度 = np.clip(上次亮度 + 变化, -50, 50)
    上次亮度 = 亮度
    return 亮度
    


class GetrPPGFromVideo():
    # 输入：input_video表示视频文件的路径，比如./video/UBFC-Phy
    def __init__(self,input_video):
        self.input_video = input_video
    
    def getrppgfromvideo(self,数据增强方法 = 'none',参数 = 0,output_mp4 = None,cunshipin = False):
        global 上次亮度
        cap = cv2.VideoCapture(self.input_video)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = None
        # 裁剪比例 = 360/frame_width
        裁剪比例 = 1
        pTime = 0
        detector = FaceMeshDetector(maxFaces=1)
        index = 0
        avg_all = []
        # 获取视频的总帧数  
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        # 创建一个进度条对象  
        pbar = tqdm(total=total_frames)  # total参数表示总任务量 

        # print("直方图均衡化！")
        print(f"数据增强方法：{数据增强方法},参数：{参数}")
        while True:
            success, img = cap.read()
            if not success:
                break
            img = cv2.resize(img, (0, 0), fx=裁剪比例, fy=裁剪比例, interpolation=cv2.INTER_LINEAR)
            
            # img = 直方图均衡化(img)
            # 应用数据增强
            if 数据增强方法 != 'none':
                if 数据增强方法 == '旋转画面':
                    img = data_augmentation(img, 数据增强方法,params=计算角度(index,参数))
                elif 数据增强方法 == '缩放画面':
                    img = data_augmentation(img, 数据增强方法,params=计算宽度(index,参数,原宽度 = img.shape[0]))
                elif 数据增强方法 == "环境光变化":
                    if index % 参数[0] == 0:
                        img = data_augmentation(img, 数据增强方法,params=计算亮度(参数[1]))
                else:
                    img = data_augmentation(img, 数据增强方法,params=参数)

            img, faces, avg_one = detector.获取视频人脸rPPG(img,True)
            # 裁剪比例 = 360/img.shape[1]
            if avg_one != None:
                avg_all.append(avg_one)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            
            if 数据增强方法 == '环境光变化':
                cv2.putText(img, f'FPS{int(fps)} light{int(上次亮度)}', (2, 7), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            else:
                cv2.putText(img, f'FPS{int(fps)}', (2, 7), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.imshow("Image", img)

            if cunshipin:
                output_dir = output_mp4+"/输出处理过程图片"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # 使用正斜杠构建文件路径
                frame_filename = output_dir+f'/frame_{index:04d}.jpg'
                cv2.imencode('.jpg', img)[1].tofile(frame_filename)


            index = index + 1
            if index >= 2500:
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            pbar.set_postfix(ordered_dict = {
                "img_size": img.shape,
                'avg_shape': np.array(avg_all).shape
            })
            # 更新进度条  
            pbar.update(1)  # update方法表示完成了多少任务量 

        cap.release()
        cv2.destroyAllWindows()
        if avg_all is None:  
            raise ValueError("avg_all is None, cannot create DataFrame")  
        else:
            print(f"rppg信号的长度：{len(avg_all)}")
        
        df = pd.DataFrame(avg_all)  
        上次亮度 = 0
        return df