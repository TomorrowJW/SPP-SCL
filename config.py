import torch

class Con_fig():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------- Dataset --------
    data_path = '/data/MVSA-S/'
    image_path = '/data/MVSA-S/data/'

    # -------- Dataset --------
    image_size = 224
    num_workers = 8
    batch_size = 64

    lr = 1e-4
    weight_decay = 1e-6
    num_epochs = 150

    # -------- Training --------
    Bert_path = '/bert-base-uncased/'
    max_length = 202

    dropout = 0.2

    #------- Contrastive --------
    temperature = 0.07

    save_model_path = '/save_models/'

    # -------- Loss weights --------
    a = 1   # CE
    c = 1   # Text SCL
    d = 1   # Image SCL
    e = 1   # Cross-modal




