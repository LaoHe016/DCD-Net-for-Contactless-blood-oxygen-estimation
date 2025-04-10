import cv2
import mediapipe as mp
import numpy as np


class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, False, self.minDetectionCon,
                                                 self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def 获取视频人脸rPPG(self, img, draw=False):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        faces = []
        avg_one = None
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                minx = 10000
                miny = 10000
                maxx = 0
                maxy = 0
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                    minx = x if x < minx else minx
                    miny = y if y < miny else miny
                    maxx = x if x > maxx else maxx
                    maxy = y if y > maxy else maxy
                    # if draw:
                    #     cv2.circle(img, (x, y), 1, (255, 255, 255), cv2.FILLED)
                    #     cv2.putText(img, f'{int(id)}', (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, id//2, 0), 1)
                # img = self.segment_skin_ycrcb(img)
                # 截取人脸部分
                # 裁剪图片
                o_img = img.copy()
                # o_face = face.copy()
                img = img[miny:maxy, minx:maxx]
                for i in range(len(face)):
                    face[i][0] -= minx
                    face[i][1] -= miny
                
                # 分割整个脸
                img = self.cut_face(img,face,162,132,136,152,365,401,389,332,10,103) 
                img = self.cut_eye(img,face,121,143,70,63,105,66,55) 
                img = self.cut_eye(img,face,372,350,336,296,334,293,300) 
                img = self.cut_eye(img,face,270,269,267,0,37,39,40,57,182,18,406,287) 
                
                # 计算多边形ROI区域的rPPG
                pts1,avg_r1,avg_g1,avg_b1 = self.get_avg(img,face,116,123,50,36,100,118,117) # 左脸颊 0,1,2
                pts2,avg_r2,avg_g2,avg_b2 = self.get_avg(img,face,329,371,266,280,352,345,347)  # 右脸颊  3,4,5
                pts3,avg_r3,avg_g3,avg_b3 = self.get_avg(img,face,108,107,55,8,285,336,337,151) # 额头    6,7,8
                # pts4,avg_r4,avg_g4,avg_b4 = self.get_avg(img,face,229,119,100,47,114,245,233,232,231,230) # 左眼角   9,10,11
                # pts5,avg_r5,avg_g5,avg_b5 = self.get_avg(img,face,452,350,277,329,348,347,346,449,450,451,452) # 额头      12,13,14
                
                # pts6,avg_r6,avg_g6,avg_b6 = self.get_avg(img,face,107,189,413,336) # 眉中      15,16,17
                # pts7,avg_r7,avg_g7,avg_b7 = self.get_avg(img,face,189,413,360,131) # 鼻子      18,19,20
                # pts8,avg_r8,avg_g8,avg_b8 = self.get_avg(img,face,203,2,423,270,269,267,0,37,39,40,203)   # 人中  21,22,23
                # pts9,avg_r9,avg_g9,avg_b9 = self.get_avg(img,face,182,406,369,175,140)  # 下巴  24,25,26
                # pts10,avg_r10,avg_g10,avg_b10 = self.get_avg(img,face,103,105,66,107,109)  # 左额头  27,28,29
                # pts11,avg_r11,avg_g11,avg_b11 = self.get_avg(img,face,338,336,296,334,332)  # 右额头 30,31,32

                pts12,avg_r12,avg_g12,avg_b12 = self.get_avg(img,face,162,132,136,152,365,401,389,332,10,103)  # 整个脸  33,34,35
                
                o_img = o_img[miny-50:maxy+50, minx-50:maxx+50]
                for i in range(len(face)):
                    face[i][0] += 50
                    face[i][1] += 50

                pts13,avg_r13,avg_g13,avg_b13 = self.get_out_avg(o_img,face,162,132,136,152,365,401,389,332,10,103)  # 背景噪声  33,34,35
                for i in range(len(face)):
                    face[i][0] -= 50
                    face[i][1] -= 50

                cv2.polylines(img, [pts1], True, (0, 255, 0), 1)
                cv2.polylines(img, [pts2], True, (0, 255, 0), 1)
                cv2.polylines(img, [pts3], True, (0, 255, 0), 1)
                # cv2.polylines(img, [pts4], True, (0, 255, 0), 1)
                # cv2.polylines(img, [pts5], True, (0, 255, 0), 1)

                # cv2.polylines(img, [pts6], True, (0, 255, 0), 1)
                # cv2.polylines(img, [pts7], True, (0, 255, 0), 1)
                # cv2.polylines(img, [pts8], True, (0, 255, 0), 1)
                # cv2.polylines(img, [pts9], True, (0, 255, 0), 1)
                # cv2.polylines(img, [pts10], True, (0, 255, 0), 1)
                # cv2.polylines(img, [pts11], True, (0, 255, 0), 1)

                cv2.polylines(img, [pts12], True, (0, 0, 255), 1)
                

                avg_one = [avg_r1,avg_g1,avg_b1,
                           avg_r2,avg_g2,avg_b2,
                           avg_r3,avg_g3,avg_b3,
                           avg_r12,avg_g12,avg_b12,
                           #avg_r13,avg_g13,avg_b13
                           ]

                # faces.append(face)

        return img, faces,avg_one
    
    def get_avg(self,img,face,*x_y):
        xys = []
        for i in range(len(x_y)):
            xys.append(face[x_y[i]])


        # 绘制四边形的pts
        pts = np.array(xys, np.int32)
        pts = pts.reshape((-1, 1, 2))
        # 利用掩码填充计算ROI平均值
        mask = np.zeros_like(img, dtype=np.uint8)  
        cv2.fillPoly(mask, [pts], (255,255,255))
        # cv2.imshow("mask", mask)
        # cv2.waitKey(1000)   
        # 计算四边形区域内每个RGB通道的平均值  
        sum_b = np.sum(img[:, :, 0] * (mask[:, :, 0] > 0))  
        sum_g = np.sum(img[:, :, 1] * (mask[:, :, 1] > 0))  
        sum_r = np.sum(img[:, :, 2] * (mask[:, :, 2] > 0))  
        count_all = np.sum(mask[:, :, 0] > 0)
        count_of_pixels = np.sum((mask[:, :, 0] > 0) & (img[:, :, 0] > 0))  # 只检查一个通道即可，因为它们都应该是一样的  
        
        average_blue = sum_b / count_of_pixels if count_of_pixels > 0 else 0  
        average_green = sum_g / count_of_pixels if count_of_pixels > 0 else 0  
        average_red = sum_r / count_of_pixels if count_of_pixels > 0 else 0 
        cv2.putText(img, f'{average_red:.4f}', (xys[0][0],xys[0][1]), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 255), 1)
        cv2.putText(img, f'{average_green:.4f}', (xys[0][0],xys[0][1]+10), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 1)
        cv2.putText(img, f'{average_blue:.4f}', (xys[0][0],xys[0][1]+20), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 0, 0), 1)
        cv2.putText(img, f'{count_of_pixels}/{count_all}', (xys[0][0],xys[0][1]+30), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 1)

        return pts,average_red,average_green,average_blue
    
    def get_out_avg(self,img,face,*x_y):
        xys = []
        for i in range(len(x_y)):
            xys.append(face[x_y[i]])


        # 绘制四边形的pts
        pts = np.array(xys, np.int32)
        pts = pts.reshape((-1, 1, 2))
        # 利用掩码填充计算ROI平均值
        mask = np.zeros_like(img, dtype=np.uint8)  
        cv2.fillPoly(mask, [pts], (255,255,255))   
        # 创建外部区域的掩码（反色）
        mask = cv2.bitwise_not(mask)[:, :, :]  # 提取单通道
        cv2.imshow("outmask", mask)
        cv2.imshow("0img",img)
        # print(mask.shape,img.shape)
        # cv2.waitKey(1000)
        # 计算四边形区域内每个RGB通道的平均值  
        sum_b = np.sum(img[:, :, 0] * (mask[:, :, 0] > 0))  
        sum_g = np.sum(img[:, :, 1] * (mask[:, :, 1] > 0))  
        sum_r = np.sum(img[:, :, 2] * (mask[:, :, 2] > 0))  
        count_all = np.sum(mask[:, :, 0] > 0)
        count_of_pixels = np.sum((mask[:, :, 0] > 0) & (img[:, :, 0] > 0))  # 只检查一个通道即可，因为它们都应该是一样的  
        
        average_blue = sum_b / count_of_pixels if count_of_pixels > 0 else 0  
        average_green = sum_g / count_of_pixels if count_of_pixels > 0 else 0  
        average_red = sum_r / count_of_pixels if count_of_pixels > 0 else 0 
        cv2.putText(img, f'{average_red:.4f}', (xys[0][0],xys[0][1]), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 255), 1)
        cv2.putText(img, f'{average_green:.4f}', (xys[0][0],xys[0][1]+10), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 1)
        cv2.putText(img, f'{average_blue:.4f}', (xys[0][0],xys[0][1]+20), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 0, 0), 1)
        cv2.putText(img, f'{count_of_pixels}/{count_all}', (xys[0][0],xys[0][1]+30), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 1)

        return pts,average_red,average_green,average_blue

    def cut_face(self,img,face,*x_y):
        # 提取坐标
        xys = [face[x_y[i]] for i in range(len(x_y))]

        # 绘制四边形的pts
        pts = np.array(xys, np.int32)
        pts = pts.reshape((-1, 1, 2))

        # 利用掩码填充计算ROI
        mask = np.zeros_like(img, dtype=np.uint8)  
        cv2.fillPoly(mask, [pts], (255, 255, 255)) 

        # 将四边形外部的像素置为0
        masked_img = cv2.bitwise_and(img, mask)

        return masked_img
    
    def cut_eye(self,img,face,*x_y):
        # 提取坐标
        xys = [face[x_y[i]] for i in range(len(x_y))]

        # 绘制四边形的pts
        pts = np.array(xys, np.int32)
        pts = pts.reshape((-1, 1, 2))

        # 创建一个全白的掩码
        mask = np.ones_like(img, dtype=np.uint8) * 255

        # 利用掩码填充计算ROI
        cv2.fillPoly(mask, [pts], (0, 0, 0))  # 将四边形内部填充为0

        # 将四边形内部的像素置为0
        masked_img = cv2.bitwise_and(img, mask)

        return masked_img
    
    def segment_skin_ycrcb(self,image):  
        # 将图像从BGR转换为YCrCb颜色空间  
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)  
    
        # 分离Y, Cr, Cb通道  
        (_, cr, cb) = cv2.split(ycrcb_image)  
    
        # 设定Cr和Cb的阈值范围（这些值可能需要根据具体图像进行调整）  
        cr_threshold_lower = 138
        cr_threshold_upper = 173  
        cb_threshold_lower = 77  
        cb_threshold_upper = 127  
    
        # 创建一个肤色掩模  
        skin_mask = np.zeros(cr.shape, dtype=np.uint8)  
        skin_mask[((cr >= cr_threshold_lower) & (cr <= cr_threshold_upper)) &  
                ((cb >= cb_threshold_lower) & (cb <= cb_threshold_upper))] = 255  
    
        # 将肤色掩模应用到原始图像上，这里我们仅展示肤色区域  
        skin_image = cv2.bitwise_and(image, image, mask=skin_mask)  
    
        return skin_image 