# Cross-attention
**Cross-Attention Guided Loss-Based Deep Dual-Branch Fusion Network for Liver Tumor Classification**

![image](https://github.com/user-attachments/assets/011d7e53-5b62-454e-ac43-4a4a0f0c83c9)

**Environment**

    pip3 install -r requirements.txt 

**Train**

    bash do_train.sh

**Test**

    bash do_test.sh


**Citation**

    @article{WANG2024102713,
    title = {Cross-attention guided loss-based deep dual-branch fusion network for liver tumor classification},
    journal = {Information Fusion},
    pages = {102713},
    year = {2024},
    issn = {1566-2535},
    doi = {https://doi.org/10.1016/j.inffus.2024.102713},
    url = {https://www.sciencedirect.com/science/article/pii/S1566253524004913},
    author = {Rui Wang and Xiaoshuang Shi and Shuting Pang and Yidi Chen and Xiaofeng Zhu and Wentao Wang and Jiabin Cai and Danjun Song and Kang Li},
    keywords = {Deep multiple instance learning, Attention, Cross-modality medical image, Liver tumor classification},
    abstract = {Recently, convolutional neural networks (CNNs) and multiple instance learning (MIL) methods have been successfully applied to MRI images. However, CNNs directly utilize the whole image as the model input and the downsampling strategy (like max or mean pooling) to reduce the size of the feature map, thereby possibly neglecting some local details. And MIL methods learn instance-level or local features without considering spatial information. To overcome these issues, in this paper, we propose a novel cross-attention guided loss-based dual-branch framework (LCA-DB) to leverage spatial and local image information simultaneously, which is composed of an image-based attention network (IA-Net), a patch-based attention network (PA-Net) and a cross-attention module (CA). Specifically, IA-Net directly learns image features with loss-based attention to mine significant regions, meanwhile, PA-Net captures patch-specific representations to extract crucial patches related to the tumor. Additionally, the cross-attention module is designed to integrate patch-level features by using attention weights generated from each other, thereby assisting them in mining supplement region information and enhancing the interactive collaboration of the two branches. Moreover, we employ an attention similarity loss to further reduce the semantic inconsistency of attention weights obtained from the two branches. Finally, extensive experiments on three liver tumor classification tasks demonstrate the effectiveness of the proposed framework, e.g., on the LLD-MMRI–7, our method achieves 69.2%, 65.9% and 88.5% on the seven-class liver tumor classification tasks in terms of accuracy, F1 score and AUC, with the superior classification and interpretation performance over recent state-of-the-art methods. The source code of LCA-DB is available at https://github.com/Wangrui-berry/Cross-attention.}
    }

