import torch
import torch.nn as nn

class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()

    def forward(self, preds, labels):       # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss += 1 - pearson
            # print(f"sum_x={sum_x} sum_y={sum_y} sum_xy={sum_xy} sum_x2={sum_x2} sum_y2={sum_y2} pearson={pearson} ")
            
        loss = loss/preds.shape[0]
        return loss
    
class Heyuanxia_Loss(nn.Module): 
    def __init__(self,参数):
        super(Heyuanxia_Loss,self).__init__()
        self.参数 = 参数
        self.criterion_Pearson = Neg_Pearson()
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()  # 添加均方误差损失
    def forward(self, pred_SpO2, labels):    
        # loss_time = self.criterion_Pearson(pred_SpO2.view(1,-1) , labels.view(1,-1))    
        # print(pred_SpO2,labels)
        loss_mae = self.mae_loss(pred_SpO2,labels)
        # loss_mse = self.mse_loss(pred_SpO2, labels)  # 计算均方误差
        
        # if torch.isnan(loss_time) or torch.isinf(loss_time):
        #     loss_time = 0
        if torch.isnan(loss_mae) or torch.isinf(loss_mae):
            print("有问题")
            loss_mae = 0
        # if torch.isnan(loss_mse) or torch.isinf(loss_mse):  # 检查MSE是否为NaN或无穷大
        #     loss_mse = 0

        # 打印损失值，可以根据需要取消注释
        # print(f'\n负皮尔逊损失 = {loss_time}  MAE = {loss_mae}  MSE = {loss_mse}')
        
        # # 计算总损失，你可以根据自己的需求调整各个损失的权重
        # loss = self.参数['皮尔逊相关系数损失计算系数'] * loss_time + 1.0 * loss_mae
        loss = 1.0 * loss_mae
        return loss,loss_mae