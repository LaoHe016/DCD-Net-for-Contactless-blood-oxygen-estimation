import torch.nn as nn
import torch

from attention import MultiHeadedAttention
from utils import SublayerConnection, PositionwiseFeedForward
from embedding import BERTEmbedding

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

class Multiscale_feature_fusion(nn.Module):
    def __init__(self,h=36):
        super(Multiscale_feature_fusion, self).__init__()
        self.avgpool_list = nn.ModuleList()
        for i in range(4):
            avgpool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化
            self.avgpool_list.append(avgpool)

        self.bk = nn.Sequential(nn.Conv1d(h*4, h, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(h, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        """Definition of 多尺度特征融合.
        Args:
          x [BS 600 36]
        Returns:
          fusion_x [BS,600,36]
        """
        B, L, C, = x[0].shape
        x_len = len(x)
        # print(x[0].shape)
        for i in range(len(x)):
            x[i] = x[i].permute(0,2,1)
            x_0 = x[i]
            x_0 = self.avgpool_list[i](x_0)
            # x_0 = torch.mean(x_0, dim=1, keepdim=True)
            x_0 = torch.sigmoid(x_0)
            # print(x_0.shape)  # 打印出x_0的形状
            # print(x[i].shape)  # 打印出x[i]的形状
            x[i] = x_0 * x[i]
            x[i] = x[i]
        x_fusion = torch.cat(x,1)
        x_result = self.bk(x_fusion)
        
        return x_result

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, 
                vocab_size = 45000,
                hidden= 36,
                n_layers=[9,5,5,5],
                attn_heads=  [
                    [1,2,3,4,6,9,12,18,36],
                    [1,2,3,4,6],
                    [6,9,12,18,36],
                    [3,4,6,9,12]
                ],
                dropout=0.1):
        """
        ：param vocab_size：总单词的vocab_side
        ：param hidden：BERT模型隐藏大小
        ：param n_layers：transformer块（层）的数量
        ：param attn_heads：注意力头数量
        ：param dropout：丢弃率
        较大的隐藏层大小可以捕获更丰富的信息
        较小的隐藏层大小可能会导致模型的表示能力受限，但也可能减少过拟合的风险，并降低计算成本。
        较小的模型可能在某些情况下泛化能力更好，因为它们不太可能过拟合训练数据。

        更多的头，可以学习到更多的子空间，能够捕获的信息多样性就越大
        在资源有限的情况下，可能需要在头数和隐藏层大小之间做出权衡，以平衡模型的性能和计算成本。
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # # 该论文指出，他们使用4*hidden_size作为ff_network_hidden_size
        # self.feed_forward_hidden = hidden * 4

        # BERT的嵌入，位置、分段、令牌嵌入的总和
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # 多层transformer块，深度网络
        self.transformer_block_0 = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads[0][i], hidden * 4, dropout) for i in range(n_layers[0])])
        self.transformer_block_1 = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads[1][i], hidden * 4, dropout) for i in range(n_layers[1])])
        self.transformer_block_2 = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads[2][i], hidden * 4, dropout) for i in range(n_layers[2])])
        self.transformer_block_3 = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads[3][i], hidden * 4, dropout) for i in range(n_layers[3])])
    
        # 多尺度特征融合
        self.Multiscale_FF = Multiscale_feature_fusion(h=hidden)

    def forward(self, x):
        '''
        输入：bs,36,600
        输出：bs,36,600
        '''
        B,C,L = x.shape
        # x = x.transpose(1,2)#bs,9，128
        # x = x.view(B,C*L)
        # 填充令牌的注意力掩蔽  bs,9,128
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # 将索引序列嵌入向量序列
        # print("词嵌入之前")
        # print(x.shape)

        x = self.embedding(x.transpose(1,2))
        # print("词嵌入之后")
        # print(x.shape)

        muti_att = []
        # 在多个transformer块上运行
        for i , transformer in enumerate(self.transformer_block_0):
            # print(i)
            if i == 0:
                x0 = transformer.forward(x, mask=None)
            else:
                x0 = transformer.forward(x0, mask=None)
        muti_att.append(x0)

        for i , transformer in enumerate(self.transformer_block_1):
            # print(i)
            if i == 0:
                x0 = transformer.forward(x, mask=None)
            else:
                x0 = transformer.forward(x0, mask=None)
        muti_att.append(x0)

        for i , transformer in enumerate(self.transformer_block_2):
            # print(i)
            if i == 0:
                x0 = transformer.forward(x, mask=None)
            else:
                x0 = transformer.forward(x0, mask=None)
        muti_att.append(x0)

        for i , transformer in enumerate(self.transformer_block_3):
            # print(i)
            if i == 0:
                x0 = transformer.forward(x, mask=None)
            else:
                x0 = transformer.forward(x0, mask=None)
        muti_att.append(x0)
        # print('transformer之后的')
        # print(x0.shape)

        x = self.Multiscale_FF(muti_att)

        return x
