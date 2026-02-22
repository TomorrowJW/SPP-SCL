from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
from config import Con_fig
from model.Model import MSA_Model
from Dataloader import get_loader

cfg = Con_fig()
test_dataloader = get_loader(cfg, mode="Test", shuffle=False, pin_memory=True)
model = MSA_Model(cfg).to(cfg.device)
model.load_state_dict(torch.load(cfg.save_model_path + 'MVSA-S.pth',  map_location=cfg.device), strict=False)
model.eval()

def test(model,dataloader):
    print('Let us start to test the model')

    pre_value = []
    true_value = []


    with torch.no_grad():
        for i, data in enumerate(dataloader, start=1):
            images,input_ids,token_type_ids,masks,labels = data
            images = images.to(cfg.device)
            input_ids = input_ids.to(cfg.device)
            token_type_ids = token_type_ids.to(cfg.device)
            masks = masks.to(cfg.device)
            labels = labels.to(cfg.device)


            out, CL_out, T_Align, I_Align = model(images,input_ids,token_type_ids,masks)

            #输出预测值
            pre_test = torch.argmax(out,dim=-1)
            #存到列表中，为了打印预测报告
            true_value.extend(list(labels.cpu().numpy()))
            pre_value.extend(list(pre_test.cpu().numpy()))


        test_accuracy = accuracy_score(true_value, pre_value)
        test_precision_w = precision_score(true_value, pre_value,average='weighted')
        test_recall_w = recall_score(true_value, pre_value,average='weighted')
        test_f1_w = f1_score(true_value, pre_value,average='weighted')

        test_precision_mi = precision_score(true_value, pre_value,average='micro')
        test_recall_mi = recall_score(true_value, pre_value,average='micro')
        test_f1_mi = f1_score(true_value, pre_value,average='micro')

        test_precision_ma = precision_score(true_value, pre_value,average='macro')
        test_recall_ma = recall_score(true_value, pre_value,average='macro')
        test_f1_ma = f1_score(true_value, pre_value,average='macro')

        test_report = classification_report(true_value,pre_value,digits=6)

        w = [test_precision_w,test_recall_w,test_f1_w]
        i = [test_precision_mi,test_recall_mi,test_f1_mi]
        a = [test_precision_ma,test_recall_ma,test_f1_ma]


    return test_accuracy,w,i,a,test_report

if __name__ == "__main__":
    test_accuracy, w,i,a, test_report = test(model,test_dataloader)
    print('the weighted value is :')
    print('Accuracy:[{:06f}], Precision:[{:06f}], Recall:[{:.6f}], F1:[{:.8f}]'.
                    format(test_accuracy, w[0], w[1], w[2]))
    print('\n')
    print('the micro value is :')
    print('Accuracy:[{:06f}], Precision:[{:06f}], Recall:[{:.6f}], F1:[{:.8f}]'.
          format(test_accuracy, i[0], i[1], i[2]))
    print('\n')
    print('the macro value is :')
    print('Accuracy:[{:06f}], Precision:[{:06f}], Recall:[{:.6f}], F1:[{:.8f}]'.
          format(test_accuracy, a[0], a[1], a[2]))
    print('\n')
    print(test_report)

