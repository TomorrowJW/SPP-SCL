from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MSADataset(Dataset):
    def __init__(self, config, mode="Train"):

        self.vocab = config.Bert_path
        self.image_path = config.image_path
        self.max_length = config.max_length

        self.tokenizer = BertTokenizer.from_pretrained(self.vocab)  # 导入分词器

        self.train_transform = transforms.Compose(
            [
                transforms.Resize(self.get_resize(config.image_size)),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])

        self.dev_test_transform = transforms.Compose(
        [
            transforms.Resize(self.get_resize(config.image_size)),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

        self.classes_for_all_samplers = []

        if mode == "Train":
            with open(config.data_path + 'train.json', 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.transform = self.train_transform

        elif mode == "Valid":
            with open(config.data_path + 'dev.json', 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.transform = self.dev_test_transform

        else:
            with open(config.data_path + 'test.json', 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.transform = self.dev_test_transform

    def __len__(self):
        return len(self.data)

    def image_loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def get_resize(self,image_size):
        for i in range(20):
            if 2**i >= image_size:
                return 2**i
        return image_size

    def text_to_id_mask(self,text):
        max_length = self.max_length  #定义文本的最大长度

        bert_input = self.tokenizer(text,padding='max_length',max_length=max_length,
                               truncation=True,return_tensors='pt')
        input_ids = bert_input['input_ids']
        token_type_ids = bert_input['token_type_ids']
        attention_mask = bert_input['attention_mask']

        return input_ids, token_type_ids, attention_mask

    def __getitem__(self, index):
        sample = self.data[index]
        image_path = os.path.join(self.image_path, sample['id']+'.jpg')
        image = self.transform(self.image_loader(image_path))

        text = sample['text']
        label = int(sample['emotion_label'])

        input_ids, token_type_ids, masks = self.text_to_id_mask(text)

        return image,input_ids.squeeze(),token_type_ids.squeeze(), masks.squeeze(),label


def get_loader(config, mode='Train', shuffle=True, pin_memory=True):
    dataset = MSADataset(config=config, mode=mode)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle,
                                num_workers=config.num_workers, pin_memory=pin_memory)
    return dataloader