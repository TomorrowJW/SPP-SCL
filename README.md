# SPP-SCL: Semi-Push-Pull Supervised Contrastive Learning for Image-Text Sentiment Analysis and Beyond 
### AAAI Conference on Artificial Intelligence (AAAI), 2026

> **Authors:**  
> Jiesheng Wu¹, Shengrong Li² (Corresponding author)
> ¹ School of Computer and Information, Anhui Normal University, Wuhu, China 
> ² College of Artificial Intelligence, Nanjing University of Aeronautics and Astronautics, Nanjing, China

## 1. Preface

- This repository provides code for "_**SPP-SCL: Semi-Push-Pull Supervised Contrastive Learning for Image-Text Sentiment Analysis and Beyond **_" AAAI 2026. [Paper](https://dl.acm.org/doi/10.1145/3768584) 


## 2. Overview

### 2.1. Introduction

Existing Image-Text Sentiment Analysis (ITSA) methods may suffer from inconsistent intra-modal and inter-modal sentiment relationships. Therefore, we develop a method that balances before fusing to solve the issue of vision-language imbalance intra-modal and inter-modal sentiment relationships; that is, a Semi-Push-Pull Supervised Contrastive Learning (SPP-SCL) method is proposed. Specifically, the method is implemented using a novel two-step strategy, namely first using the proposed intra-modal supervised contrastive learning to pull the relationships between the intra-modal and then performing a well-designed conditional execution statement. If the statement result is false, our method will perform the second step, which is inter-modal supervised contrastive learning to push away the relationships between inter-modal. The two-step strategy will balance the intra-modal and inter-modal relationships to achieve the purpose of relationship consistency and finally perform cross-modal feature fusion for sentiment analysis and detection. Experimental studies on three public image-text sentiment and sarcasm detection datasets demonstrate that SPP-SCL significantly outperforms state-of-the-art methods by a large margin and is more discriminative in sentiment.

### 2.2. Framework Overview

<p align="center">
    <img src="imgs/SPP-SCL.jpg"/> <br />
    <em> 
    Figure 1:  Overall architecture of SPP-SCL. The framework includes two main steps: intra-modal sentiment alignment via supervised contrastive learning, and conditional inter-modal sentiment alignment.
    </em>
</p>

### 2.3. Quantitative Results

<p align="center">
    <img src="imgs/results.jpg"/> <br />
    <em> 
    Figure 2: Quantitative Results
    </em>
</p>

### 2.4. Qualitative Results

<p align="center">
    <img src="imgs/vis.jpg"/> <br />
    <em> 
    Figure 3: Qualitative Results.
    </em>
</p>

## 3. Proposed Method

### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with an NVIDIA Tesla V100 GPU of 16 GB Memory.

1. Configuring your environment (Prerequisites):
       
    + Installing necessary packages: `pip install -r requirements.txt`.

1. Downloading necessary data:

    + downloading training dataset and move it into `./data/`, 
    which can be found from [Baidu Drive](https://pan.baidu.com/s/1zLYWsxxluq1elQuyY7gg3w) (extraction code: ekd2). 

    + downloading testing dataset and move it into `./data/`, 
    which can be found from [Baidu Drive](https://pan.baidu.com/s/1xnaiHnAuj4UVTPRak9oU2g) (extraction code: nhwe). 
        
    + downloading our weights and move it into `./save_models/PVT-V2-B4-384.pth`, 
    which can be found from [(Baidu Drive)](https://pan.baidu.com/s/1ibKdnGDf4_vCGY_zmqyyAA?pwd=2855) (extraction code: 2855). 
    
    + downloading weights and move it into `./pre_train/pvt_v2_b4.pth`,
    which can be found from [Baidu Drive](https://pan.baidu.com/s/1aWXw0O7vMXRYWQK5MvHBIA?pwd=u1u6) (extraction code: u1u6). 

1. Training Configuration:

    + After you download training dataset, just run `MyTrain.py` to train our model.


1. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `MyTest.py` to generate the final prediction maps.
    
    + You can also download prediction maps and edge prediction maps ('CHAMELEON', 'CAMO', 'COD10K', 'NC4K') from [Baidu Drive](https://pan.baidu.com/s/1MhEAX396p9cFGbehCxgYlA?pwd=w2mw) (extraction code: w2mw)).

Note:  If you have difficulty accessing Baidu Drive, please contact us for alternative download links.
### 3.2 Evaluating your trained model:

One evaluation is written in Python codes ([link](https://github.com/lartpang/PySODMetrics)), or Matlab codes ([link](https://github.com/DengPingFan/CODToolbox)).
please follow this the instructions in `MyEval.py` and just run it to generate the evaluation results.

## 4. Citation

Please cite our paper if you find the work useful, thanks!
	
	@article{wu2025boosting,
	  title={Boosting Foreground-Background Disentanglement for Camouflaged Object Detection},
	  author={Wu, Jiesheng and Hao, Fangwei and Xu, Jing},
	  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
	  volume={21},
	  number={12},
	  pages={1--23},
	  year={2025},
	  publisher={ACM New York, NY}
	}

## 5. Contact

For any questions, discussions, or collaboration opportunities, please contact:

**Jiesheng Wu**  
School of Computer and Information, Anhui Normal University, Wuhu, China
Email: jasonwu@mail.nankai.edu.cn

**[⬆ back to top](#1-preface)**
