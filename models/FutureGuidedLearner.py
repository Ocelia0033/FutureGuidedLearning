import torch
import torch.nn as nn
import torch.nn.functional as F

class FutureGuidedLearner(nn.Module):
    """
    Future-Guided Learning (FGL)模型用于时间序列预测
    基于项目中描述的FGL方法，实现一个适合光伏功率预测的模型
    """
    def __init__(self, input_dim, output_dim, input_len, pred_len):
        super(FutureGuidedLearner, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_len = input_len
        self.pred_len = pred_len
        
        # LSTM编码器用于提取历史信息
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 解码器层用于生成预测
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, pred_len * output_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入序列，形状为 [batch_size, input_len, input_dim]
        
        返回:
            pred: 预测序列，形状为 [batch_size, pred_len, output_dim]
        """
        # 通过LSTM编码器
        _, (h_n, _) = self.encoder(x)
        
        # 使用最后一层的隐藏状态作为上下文向量
        context = h_n[-1]
        
        # 通过解码器生成预测
        pred = self.decoder(context)
        
        # 重塑预测结果为所需的输出形状
        batch_size = x.size(0)
        pred = pred.view(batch_size, self.pred_len, self.output_dim)
        
        return pred