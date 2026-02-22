import torch.nn as nn
import torch
from .Bert import Bert_Model
from .ResNet import resnet

# ============================================================
# Hierarchical Attention (HA)
# 分层注意力模块 —— 对应论文 Section 3.2
# ============================================================
class Attention(nn.Module):
    """
        Hierarchical Attention for Text Encoding
        分层注意力文本编码模块

        Input:
            layer  : CLS representations from last 4 BERT layers
                     来自BERT最后四层的CLS向量
            input  : token representations from last 4 BERT layers
                     来自BERT最后四层的token特征

        Output:
            text_encoding : enhanced text representation after HA + BiLSTM
                            经HA与BiLSTM增强后的文本表示
        """
    def __init__(self,inplane=4):
        super(Attention, self).__init__()

        # 全局平均池化用于生成层权重
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # 两层全连接生成 attention 权重（论文公式2）
        # α = σ(W2 ReLU(W1 H_cls))
        self.fc = nn.Sequential(nn.Linear(inplane,inplane,bias=False),
                                nn.ReLU(),
                                nn.Linear(inplane,inplane,bias=False),
                                nn.Sigmoid())# BiLSTM 用于建模上下文依赖
        # BiLSTM 用于建模上下文依赖
        self.shared_text_encoding = FastLSTM()

    def forward(self,layer,input):
        # 拼接4层CLS向量  [B,4,768]
        x = torch.cat(layer, dim=1)
        # 生成层权重 α
        y = self.avgpool(x).permute(0,2,1)
        y = self.fc(y).permute(0,2,1).unsqueeze(-1)
        # 拼接4层token特征
        hidden = torch.cat(input,dim=1)

        # 加权求和
        out = y.expand_as(hidden) * hidden
        out = torch.sum(out,dim=1).squeeze(1)

        #BiLSTM
        text_encoding = self.shared_text_encoding(out)

        return text_encoding

# ============================================================
# BiLSTM for Context Modeling
# 双向LSTM上下文建模
# ============================================================
class FastLSTM(nn.Module):
    def __init__(self):
        super(FastLSTM, self).__init__()

        # hidden_size=64, bidirectional=True
        # 输出维度为 128
        self.LSTM = nn.LSTM(input_size=768,
                            hidden_size=64,
                            num_layers=1,
                            bias=False,
                            batch_first=False,
                            bidirectional=True)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        out,_ = self.LSTM(x)
        out_final = out.permute(1,0,2)
        return out_final

# ============================================================
# Shared Projection for Text (Ip / Tp)
# 文本对齐投影模块
# ============================================================
class EncodingPart_T(nn.Module):
    """
    Project text feature into shared embedding space (D=32)
    将文本特征映射到共享对齐空间
    """
    def __init__(self,shared_text_dim=32,drop=0.1):
        super(EncodingPart_T, self).__init__()
        self.shared_text_linear = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(32, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim)
        )

    def forward(self, text): #text:[b,128]
        text_shared = self.shared_text_linear(text)
        return text_shared  #text:[b,64]

# ============================================================
# Shared Projection for Image (Ip / Tp)
# 图像对齐投影模块
# ============================================================
class EncodingPart_I(nn.Module):
    """
    Project image feature into shared embedding space (D=32)
    将图像特征映射到共享对齐空间
    """
    def __init__(self,shared_image_dim=32,drop=0.1):
        super(EncodingPart_I, self).__init__()
        self.shared_image_linear = nn.Sequential(
                nn.Linear(128, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(32, shared_image_dim),
                nn.BatchNorm1d(shared_image_dim)
            )

    def forward(self,image):
        image_shared = self.shared_image_linear(image)
        return image_shared

# ============================================================
# MSA_Model (SPP-SCL Framework)
# 整体多模态情感分析模型
# ============================================================
class MSA_Model(nn.Module):
    def __init__(self,config):
        super(MSA_Model, self).__init__()

        # ================= Image Encoder =================
        # ResNet-50 backbone
        self.image_encoder = resnet()
        for params in self.image_encoder.parameters():
            params.requires_grad = False # 冻结图像主干

        # ================= Text Encoder ==================
        # BERT backbone
        self.text_encoder = Bert_Model(config)
        for params in self.text_encoder.parameters():
            params.requires_grad = False # 冻结文本主干

        # ================= Hierarchical Attention =========
        self.Attention = Attention()

        # 1x1 conv + GAP for image feature reduction
        self.Conv_I = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.AdaptiveAvgPool2d(1))

        # ================= Shared Alignment Space =========
        self.Align_Module_T = EncodingPart_T(drop=config.dropout)
        self.Align_Module_I = EncodingPart_I(drop=config.dropout)

        # ================= CMF (Cross-Modal Fusion) =======
        # Q, K, V projections
        self.Q_T = nn.Sequential(nn.Linear(32,32),nn.Dropout(config.dropout))
        self.K_T = nn.Sequential(nn.Linear(32,32),nn.Dropout(config.dropout))
        self.V_T = nn.Sequential(nn.Linear(32,32),nn.Dropout(config.dropout))

        self.Q_I = nn.Sequential(nn.Linear(32,32),nn.Dropout(config.dropout))
        self.K_I = nn.Sequential(nn.Linear(32,32),nn.Dropout(config.dropout))
        self.V_I = nn.Sequential(nn.Linear(32,32),nn.Dropout(config.dropout))

        self.scale = 32 ** -0.5  # 自注意力层的scale操作
        self.attend = nn.Softmax(dim=-1)  # 自注意力层的softmax归一化操作

        # Alignment refinement
        self.Align_I = nn.Sequential(nn.Linear(32,32),nn.BatchNorm1d(32),
                                    nn.ReLU(),nn.Dropout(config.dropout))
        self.Align_T = nn.Sequential(nn.Linear(32,32),nn.BatchNorm1d(32),
                                    nn.ReLU(),nn.Dropout(config.dropout))

        self.classifier = nn.Sequential(nn.Linear(32*4,32*2),nn.BatchNorm1d(32*2),
                                    nn.ReLU(),nn.Dropout(config.dropout))

        # ================= Classifier ======================
        self.fc = nn.Linear(32*2,3)

    def forward(self,image,input_ids,token_type_ids,mask):

        # -------- Image branch --------
        I_L = self.image_encoder(image)
        CLS_T,layer3,hidden3 = self.text_encoder(input_ids, token_type_ids, mask)

        # -------- Text branch --------
        T_L = self.Attention(layer3, hidden3)
        I_L = self.Conv_I(I_L).squeeze(-1).squeeze(-1)

        # -------- Shared embedding --------
        T_Contra = self.Align_Module_T(T_L.mean(1)) #[b,64]
        I_Contra = self.Align_Module_I(I_L) #[b,64]

        T_Align = self.Align_T(T_Contra)
        I_Align = self.Align_I(I_Contra)

        # -------- Cross-Modal Fusion (CMF) --------
        I_Q = self.Q_I(I_Align)
        I_K = self.K_I(I_Align)
        I_V = self.V_I(I_Align)

        T_I_Q = T_Align * I_Q
        T_I_K = T_Align * I_K

        dot_T_I = self.attend(torch.matmul(T_I_Q,T_I_K.T)*self.scale)
        value_T_I = torch.matmul(dot_T_I ,I_V)

        T_Q = self.Q_T(T_Align)
        T_K = self.K_T(T_Align)
        T_V = self.V_T(T_Align)

        I_T_Q = I_Align * T_Q
        I_T_K = I_Align * T_K

        dot_I_T = self.attend(torch.matmul(I_T_Q,I_T_K.T)*self.scale)
        value_I_T = torch.matmul(dot_I_T,T_V)

        # 拼接 [Im, Tm, Ip, Tp]
        value = torch.cat((value_I_T, value_T_I),1)
        fusion = torch.cat((value,I_Align,T_Align),1)

        CL_out = self.classifier(fusion)
        output = self.fc(CL_out)

        return output, CL_out, T_Align, I_Align