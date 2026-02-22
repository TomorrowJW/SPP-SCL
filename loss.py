import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Intra-modal Supervised Contrastive Loss (L_cli / L_clt)
# 同模态监督对比损失 —— 对应论文 Eq.(6)
# ============================================================
class CLoss(nn.Module):
    """
    Supervised Contrastive Loss for single modality.

    用于单模态（Image 或 Text）的监督对比学习。

    Paper formulation:
        L = - 1/|P(i)| * sum_{p in P(i)}
            log( exp(sim(i,p)/τ) / sum_{a≠i} exp(sim(i,a)/τ) )

    Args:
        temperature (τ): temperature scaling factor
    """
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature


    def forward(self, output, label):
        """
         Args:
             output : feature embeddings [B, D]
             label  : class labels [B]

         Returns:
             loss : supervised contrastive loss
         """
        batch_size = label.size()[0]
        label = label.unsqueeze(-1)
        assert label.size() == torch.Size([batch_size,1])

        # ====================================================
        # Step 1: Construct positive mask
        # 构造同类别正样本掩码
        # mask[i,j] = 1 if y_i == y_j else 0
        # ====================================================
        with torch.no_grad():
            mask = torch.eq(label, label.transpose(0, 1))
            mask = mask.float().to(output.device)
            diag_mask = torch.scatter(torch.ones(batch_size,batch_size),1,torch.arange(batch_size).view(-1,1),0).to(output.device)
            mask = mask * diag_mask

        # ====================================================
        # Step 2: L2 normalize embeddings
        # 特征向量归一化（论文中要求 ℓ2-normalized）
        # ====================================================
        output = F.normalize(output,p=2,dim=-1)

        # ====================================================
        # Step 3: Compute similarity matrix
        # 计算余弦相似度矩阵
        # ====================================================
        dot = torch.matmul(output, output.transpose(0, 1))
        logits = torch.div(dot, self.temperature)

        assert logits.size() == torch.Size([batch_size,batch_size])

        # Remove diagonal from logits
        logits = logits * diag_mask

        # ====================================================
        # Step 4: Numerical stability trick
        # 数值稳定性处理（减去每行最大值）
        # ====================================================
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)

        # ====================================================
        # Step 5: Compute log-probability
        # 计算对数概率
        # ====================================================
        logits = logits * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True)+1e-12)

        # ====================================================
        # Step 6: Average over positive samples
        # 对所有正样本取平均
        # ====================================================
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum==0,torch.ones_like(mask_sum),mask_sum)
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = pos_logits.mean()
        loss = -loss

        return loss


# ============================================================
# Cross-modal Supervised Contrastive Loss (L_clm)
# 跨模态监督对比损失 —— 对应论文 Eq.(7)
# ============================================================
class CLoss_T_I(nn.Module):
    """
        Supervised contrastive loss between Image and Text features.

        用于图像-文本跨模态对比学习。

        Paper formulation:
            L = - 1/|P(i)| * sum_{p in P(i)}
                log( exp(sim(I_i,T_p)/τ) /
                     sum_j exp(sim(I_i,T_j)/τ) )
        """
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, Image_feature, Text_feature, label):
        """
            Args:
                Image_feature : [B, D]
                Text_feature  : [B, D]
                label         : [B]

            Returns:
                loss : cross-modal supervised contrastive loss
        """

        batch_size = label.size()[0]
        label = label.unsqueeze(-1)
        assert label.size() == torch.Size([batch_size,1])

        # ====================================================
        # Step 1: Construct positive mask
        # 构造跨模态同类别掩码
        # ====================================================

        with torch.no_grad():
            mask = torch.eq(label, label.transpose(0, 1))
            mask = mask.float().to(Image_feature.device)

        # ====================================================
        # Step 2: L2 normalization
        # 特征归一化
        # ====================================================

        Image_feature = F.normalize(Image_feature,p=2,dim=-1)
        Text_feature = F.normalize(Text_feature,p=2,dim=-1)

        # ====================================================
        # Step 3: Cross-modal similarity matrix
        # 计算图像-文本相似度矩阵
        # ====================================================
        dot = torch.matmul(Image_feature, Text_feature.transpose(0, 1))
        logits_all = torch.div(dot, self.temperature)
        assert logits_all.size() == torch.Size([batch_size,batch_size])

        # Numerical stability
        logits_max, _ = torch.max(logits_all, dim=1, keepdim=True)
        logits_all = logits_all - logits_max.detach()

        exp_logits = torch.exp(logits_all)

        # ====================================================
        # Step 4: Compute log-probability
        # ====================================================
        logits_p = logits_all * mask
        log_prob = logits_p - torch.log(exp_logits.sum(dim=1, keepdim=True)+1e-12)

        # ====================================================
        # Step 5: Average positive samples
        # ====================================================
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum==0,torch.ones_like(mask_sum),mask_sum)
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()

        loss = pos_logits.mean()
        loss = -loss
        return loss
