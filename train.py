from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
import torch.nn as nn
import os
import random
import numpy as np
from config import Con_fig
from model.Model import MSA_Model
from Dataloader import get_loader
from loss import CLoss,CLoss_T_I
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# Environment Configuration
# 环境与设备配置
# ================================================================

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_seed(seed):
    # ================================================================
    # Environment Configuration
    # 环境与设备配置
    # ================================================================
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(42)

# ================================================================
# Configuration & Model Initialization
# 配置与模型初始化
# ================================================================
cfg = Con_fig()
model = MSA_Model(cfg).to(cfg.device)

# ================================================================
# Configuration & Model Initialization
# 配置与模型初始化
# ================================================================
CLos = CLoss(cfg.temperature).to(cfg.device)  # Text SCL
CLos_T = CLoss(cfg.temperature).to(cfg.device) # Image SCL
CLos_I = CLoss(cfg.temperature).to(cfg.device) # Cross-modal SCL
T_I_Clos = CLoss_T_I(cfg.temperature).to(cfg.device)

# Classification loss
# 分类损失
BCE = nn.CrossEntropyLoss().to(cfg.device)

# ================================================================
# DataLoader
# 数据加载
# ================================================================
train_dataloader = get_loader(cfg, mode="Train", shuffle=True, pin_memory=True)
val_dataloader = get_loader(cfg, mode="Valid", shuffle=False, pin_memory=True)

total_step = len(train_dataloader)

# ================================================================
# Optimizer & Scheduler
# 优化器与学习率调度器
# ================================================================

params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(params, cfg.lr, betas=(0.9, 0.999), eps=1e-8,weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ================================================================
# Validation Function
# 验证阶段
# ================================================================
def valid(model,dataloader):
    """
    Model evaluation on validation set.
    在验证集上评估模型性能。
    """
    pre_value = []
    true_value = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader, start=1):
            images, input_ids, token_type_ids, masks, labels = data
            images = images.to(cfg.device)
            input_ids = input_ids.to(cfg.device)
            token_type_ids = token_type_ids.to(cfg.device)
            masks = masks.to(cfg.device)
            labels = labels.to(cfg.device)

            output, CL_out, T_Align, I_Align = model(images,input_ids,token_type_ids,masks)

            pre_val = torch.argmax(output,dim=1)
            true_value.extend(list(labels.cpu().numpy()))
            pre_value.extend(list(pre_val.cpu().numpy()))

        val_accuracy = accuracy_score(true_value, pre_value)
        val_precision = precision_score(true_value, pre_value,average='weighted')
        val_recall = recall_score(true_value, pre_value,average='weighted')
        val_f1 = f1_score(true_value, pre_value,average='weighted')

        val_report = classification_report(true_value,pre_value,digits=6)

    model.train()

    return val_accuracy,val_precision,val_recall,val_f1, val_report

# ================================================================
# Training Function (Two-step Optimization)
# 训练函数（两阶段优化策略）
# ================================================================

def train():
    """
    Two-step optimization:
    Step 1: Update CE + Intra-modal SCL
    Step 2: Conditionally update Cross-modal SCL

    两阶段优化：
    第一步：更新分类损失 + 模态内对比损失
    第二步：根据对齐质量判断是否更新跨模态对比损失
    """
    print('Let us start to train the model:')

    for epoch in range(cfg.num_epochs):

        train_pre_value = []
        train_true_value = []

        model.train()

        for i, data in enumerate(train_dataloader, start=1):
            optimizer.zero_grad()

            images,input_ids,token_type_ids,masks,labels = data
            images = images.to(cfg.device)
            input_ids = input_ids.to(cfg.device)
            token_type_ids = token_type_ids.to(cfg.device)
            masks = masks.to(cfg.device)
            labels = labels.to(cfg.device)

            # ========================================================
            # Step 1: Intra-modal + CE update
            # 第一步：分类损失 + 模态内对比损失
            # ========================================================
            output, CL_out, T_Align, I_Align = model(images,input_ids,token_type_ids,masks)
            loss_ce = 5*BCE(output,labels)

            loss_cl_t = CLos_T(T_Align,labels)
            loss_cl_i = CLos_I(I_Align,labels)

            loss1 = cfg.a * loss_ce + cfg.c * loss_cl_t + cfg.d * loss_cl_i
            loss1.backward()
            optimizer.step()
            
            # ========================================================
            # Step 2: Conditional cross-modal update
            # 第二步：根据对齐质量决定是否更新跨模态损失
            # ========================================================
            output_post, CL_out_post, T_Align_post, I_Align_post = \
                model(images, input_ids, token_type_ids, masks)

            similarity = torch.sigmoid(torch.matmul(T_Align_post, I_Align_post.T))
            dig = similarity.diag().sum() / output_post.size(0)
            count = (similarity < dig).float().sum().item()
            mask = torch.eq(labels.unsqueeze(-1), labels.unsqueeze(-1).transpose(0, 1))
            mask = mask.float().to(cfg.device).sum()


            if count >= int((mask.item()) *(2/3)):
                loss_cm = torch.zeros(1, device=cfg.device)
                loss = loss1
            else:
                loss_cm = T_I_Clos(T_Align_post, I_Align_post, labels)
                loss2 = cfg.e * loss_cm

                optimizer.zero_grad()
                loss2.backward()
                optimizer.step()
                loss = loss1 + loss2

            # ========================================================
            # Metrics
            # ========================================================
            pre_val = torch.argmax(output,dim=1)
            train_true_value.extend(list(labels.cpu().numpy()))
            train_pre_value.extend(list(pre_val.cpu().numpy()))

            if i % 50 == 0 or i == total_step:
                print(
                    'Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Lossce: {:.6f}, Losscl_t: {:.6f}, Losscl_i: {:.6f}, Losscm: {:.6f}, Loss:{:.6f}'.
                    format(epoch, cfg.num_epochs, i, total_step, optimizer.param_groups[0]['lr'], loss_ce.item(),
                        loss_cl_t.item(), loss_cl_i.item(), loss_cm.item(), loss.item()))
        
        # ============================================================
        # Epoch-level Evaluation
        # ============================================================
        train_accuracy = accuracy_score(train_true_value, train_pre_value)
        train_f1 = f1_score(train_true_value, train_pre_value,average='weighted')

        train_report = classification_report(train_true_value, train_pre_value, digits=6)

        val_accuracy, val_precision, val_recall, val_f1, val_report = \
        valid(model,val_dataloader)

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), cfg.save_model_path + '%d' % epoch + 'Net.pth')
            print('Train acc:',train_accuracy)
            print('Valid acc:',val_accuracy)
            print('Train F1:', train_f1)
            print('Valid F1:', val_f1)
            print('Train report:',train_report)
            print('Valid report:', val_report)


if __name__ == '__main__':
    train()