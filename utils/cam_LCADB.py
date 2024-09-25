import argparse
import cv2
import numpy as np
import torch
from torchvision import models
import torch.nn.functional as F

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import os
from torch import Tensor

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import monai
from monai.utils import GridSampleMode, GridSamplePadMode,InterpolateMode,NumpyPadMode
from monai.transforms import RandGaussianNoise, SqueezeDim, Rand3DElastic, RandGaussianSmooth, RandZoom, RandAffine, LoadImage, AddChannel, ToTensor, Resize, NormalizeIntensity, RandRotate90, RandFlip
from monai.transforms import Compose, SpatialCropd, CenterSpatialCropd, RandSpatialCropd, Rand3DElasticd, RandGaussianNoised, SqueezeDimd, \
                             LoadImaged, AddChanneld, EnsureChannelFirstd, ToTensord, Resized, NormalizeIntensityd, \
                             RandRotate90d, RandFlipd, ConcatItemsd, RandZoomd, RandGaussianSmoothd, AdjustContrastd,RandAdjustContrastd,\
                             RandAffined, ScaleIntensityd, RandSpatialCropSamplesd,RandRotated,ScaleIntensityRanged,RandScaleCropd,RandSpatialCropd
import yaml
config_path = os.path.dirname(os.path.abspath(__file__)) + '/config.yaml'
with open(config_path,'r',encoding='utf8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

from skimage.transform import resize
from matplotlib import pyplot as plt
from einops import rearrange,repeat
from torchvision.utils import save_image

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def train_transform_sim(image):

    train_transform_sim = Compose(
        [
                LoadImaged(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp"), image_only=True),
                EnsureChannelFirstd(keys=["t2","dwi","in_","out_","pre","ap","pvp","dp"]),
                Resized(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp"), spatial_size=(128,128,16)),
                RandRotated(keys=["t2","dwi","in_","out_","pre","ap","pvp","dp"],range_x=[- np.pi/18, np.pi/18], prob=0.3, mode=GridSampleMode.BILINEAR, padding_mode=GridSamplePadMode.BORDER, keep_size=True),
                RandFlipd(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp"), prob=0.3),         
                NormalizeIntensityd(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp")),
                ToTensord(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp"))
            ]
        )
    image = train_transform_sim(image)
    return image


class LayerActivations:
    features = None
 
    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

 
    def hook_fn(self, module, nii_img, output):
        self.features = output.cpu()
 
    def remove(self):
        self.hook.remove()

def getCAM(layer_activation, weight_fc, target_cls): #layer_activation为类的实例
    """
    layer_activation: the activations and gradients of target_layer
        activations: feature map before GAP.  shape => (N, C, T, H, W) 最后一个卷积层输出的特征图 [1, 512, 2, 10, 10]
    weight_fc: the weight of fully connected layer.  shape => (num_classes, C) [2, 512]
    target_cls: predicted class id
    cam: class activation map.  shape => (1, num_classes, H, W)
    """
    print("value:", layer_activation.shape)#[1,512,2,16,16]
    print("weight:", weight_fc[:, :, None, None, None].shape)#[n_classes,512,1,1,1]
    if layer_activation.shape[0] != 1:  #P-Net/MIL
        layer_activation = rearrange(layer_activation,'(b h w) c d p1 p2->b c d (h p1) (w p2)',h=4,w=4) # =c=M(8)*C
        print("pnet-value:", layer_activation.shape)#([16, 512, 1, 2, 2])->[1, 512, 1, 8, 8])


    cam = F.conv3d(layer_activation, weight=weight_fc[:, :, None, None, None])# ([1, 1, 2, 16, 16])
    _, _, t, h, w = cam.shape       #[1,n_classes,d,h,w]       
    print("cam_new",cam.shape)

    cam -= torch.min(cam)
    cam /= torch.max(cam)
    cam = cam.view(1, 1, t, h, w)

    return cam.data

def visualize(images_t2,images_dwi, images_in, images_out, images_pre, images_ap, images_pvp, images_dp, cam1, cam2, data_dict, target_cls):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        clip: (Tensor) shape => (1, 3, T, H, W)
        cam: (Tensor) shape => (1, 1, T, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, T, H, W)
    """

    _, _, T, H, W = images_pvp.shape
    cam1 = F.interpolate(cam1, size=(T, H, W), mode="trilinear", align_corners=False)
    cam2 = F.interpolate(cam2, size=(T, H, W), mode="trilinear", align_corners=False)

    cam1 = 255 * cam1.squeeze()
    cam2 = 255 * cam2.squeeze()
    heatmaps = []
    for t in range(T):
        c = cam1[t] + cam2[t]
        heatmap = cv2.applyColorMap(np.uint8(c), cv2.COLORMAP_JET)
        # print("heatmap11",heatmap.shape)            #(192,)
        heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
        # print("heatmap22",heatmap.shape)
        
        heatmap = heatmap.float() / 255
        b, g, r = heatmap.split(1)
        heatmap = torch.cat([r, g, b])
        heatmaps.append(heatmap)

    heatmaps = torch.stack(heatmaps)
    heatmaps = heatmaps.transpose(1, 0).unsqueeze(0)
    print("heatmaps",heatmaps.shape)    #([1, 3, 16, 128, 128])
    print("images_pvp",images_pvp.shape)        #([1, 1, 16, 128, 128])
    result = 0.6*images_pvp + 0.8*heatmaps.cpu()          #todo
    result = result.div(result.max())
    print("result",result.shape)        #([1, 3, 16, 128, 128])
    
    print("data_dict['pvp']",data_dict['pvp'].split('/')[-2])      #.split('/')[-1]

    for i in range(T):
        # print(i)
        nii_img = images_pvp[:, :, i].squeeze()
        heatmap = result[:, :, i].squeeze()
        save_path = '//wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/cam/LCADB_onlyFC/{}_target_cls{}_pvp'.format(data_dict['pvp'].split('/')[-2], int(target_cls))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_image(nii_img, os.path.join(save_path,'nii_{}.eps'.format(str(i))))
        save_image(nii_img, os.path.join(save_path,'nii_{}.png'.format(str(i)))) 
        save_image(heatmap, os.path.join(save_path,'{}.eps'.format(str(i))))
        save_image(heatmap, os.path.join(save_path,'{}.png'.format(str(i))))
    return result


if __name__ == '__main__':


    seed = 5
    # seed_everything(seed)
    print("seed_everything:",seed)
    from monai.utils import set_determinism
    set_determinism(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True,warn_only=True)
    
    parser = argparse.ArgumentParser()
# Dataset parameters
    parser.add_argument('--data_dir', default='//wangrui/MVI/experiment/LLDMRI_1011/2data_roi_register/images', type=str)
    parser.add_argument('--csv_path', default='', type=str)
    parser.add_argument('--num_classes', type=int, default=7, metavar='N',
                    help='number of label classes (Model default if None)')
    parser.add_argument('--result_dir', default='', type=str)
    parser.add_argument('--json_dir', default='', type=str)
    parser.add_argument('--csv_dir', default='', type=str)
        
# Model parameters
    parser.add_argument('--net', type=str, default="3dres", 
                        help='[brain, Song,DAMIDL,deepganet];[ours: (Rbase, RANet,PANet),3dres/crossatten]')
    parser.add_argument('--cuda',type=str_to_bool,default=True, 
                        help='load in GPU')
#Pretrained model
    parser.add_argument('--resume',type=str_to_bool,default=True,
                        help='read model.state.dict()')
    parser.add_argument('--weight_path',default='') 
#Parallel
    #Parser.add_argument('--device',type=str,default='cuda:0',help='')
    parser.add_argument('--data_parallel', type=str_to_bool, default=False)
    # parser.add_argument('--device_id',type=list,default=[0,1],help='')
#Parameter
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='batch size for dataloader')

    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='weight decay rate')
    parser.add_argument('--num_epoch', type=int, default=1, metavar='N')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
    
    parser.add_argument('--n_fold', type=str, default='5fold', 
                        help='1fold/5fold')
    args = parser.parse_args()
    
    exp_info = vars(args)
    print("exp_info:", exp_info)

    # writer = SummaryWriter(log_dir=syspath+"/log/"+starttime[:10]+"/"+starttime[11:],flush_secs=60)

    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """


# 1.加载模型

    print("--------Load - Model - Start!------")
    if args.net == "3dres":
        from model.crossatten import generate_crossatten
        model = generate_crossatten(10)
        print("batch_size=",args.batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        print("3dres net as baseline:","generate_model")
        # model = nn.DataParallel(model)  
        if args.resume:
            print("Load the best.pt model",args.weight_path)
            # utils.set_logger(os.path.join(experiment_path, 'load_train.log'))
            checkpoint = torch.load(args.weight_path)
            raw_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])       #!!!
            optimizer.load_state_dict(checkpoint['optimizer'])
            # model.load_state_dict(checkpoint, strict=False)


            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            best_f1 = checkpoint['best_f1']
            best_acc = checkpoint['best_acc']
            model.eval()
            print("------Load - Model -Done!------Have - Pretrain:",raw_epoch,"epochs")

# 2.选择目标层
    #################111#################
    # target_layer1 = [model.layer4_compose]
    # target_layer2 = [model.layer4_patch_com]
    #################222#################
    target_layer1 = [model.conv6]
    target_layer2 = [model.layer4_patch_com]
    #################!!!#################
    attention_layer = [model.attention_layer]


# 3. 构建输入图像的Tensor形式

    for data_path in os.listdir(args.data_dir): 
        patient_path = os.path.join(args.data_dir,data_path)

        phase_list = ['T2WI', 'DWI', 'In Phase', 'Out Phase', 'C-pre', 'C+A', 'C+V', 'C+Delay']
        
        t2path = os.path.join(patient_path,'T2WI.nii.gz')
        dwipath = os.path.join(patient_path,'DWI.nii.gz')
        in_path = os.path.join(patient_path,'In Phase.nii.gz')
        out_path = os.path.join(patient_path,'Out Phase.nii.gz')
        prepath = os.path.join(patient_path,'C-pre.nii.gz')
        apath = os.path.join(patient_path,'C+A.nii.gz')
        dpath = os.path.join(patient_path,'C+Delay.nii.gz')
        pvpath = os.path.join(patient_path,'C+V.nii.gz')
        data_dict = {'t2': t2path, 'dwi':dwipath, 'in_':in_path, 'out_':out_path, 'pre': prepath, 'ap':apath, 'dp':dpath,'pvp':pvpath}

        images  = train_transform_sim(data_dict)     #torch.Size([1(unsqueeze), 1, H, W, D])

        images_t2,images_dwi, images_in, images_out, images_pre, images_ap, images_pvp, images_dp = images['t2'].unsqueeze(0), images['dwi'].unsqueeze(0), images['in_'].unsqueeze(0), images['out_'].unsqueeze(0), images['pre'].unsqueeze(0), images['ap'].unsqueeze(0), images['pvp'].unsqueeze(0), images['dp'].unsqueeze(0)
        # print("image,",nii_img.shape)
        b,c,h,w,d = images_pvp.shape       
        print("nii_img.shape: ",b,c,h,w,d)  #1, C=1, H=128,W=128,D=16


        conv_out1 = LayerActivations(target_layer1, 0)  #特征图/激活图A_{k}
        conv_out2 = LayerActivations(target_layer2, 0)      #[b,c,d,h,w]



# 4. 数据输入模型
        output, out1, f_compose, alpha_compose, out2, f_patch_com, alpha_patch_com = model(images_t2.to(torch.float32),images_dwi.to(torch.float32), images_in.to(torch.float32), images_out.to(torch.float32), images_pre.to(torch.float32), images_ap.to(torch.float32), images_pvp.to(torch.float32), images_dp.to(torch.float32))      #(patch_num,C=1,H,W,D)



        prob = F.softmax(output, dim=1)
        target_category = np.argmax(output.cpu().data.numpy(), axis=-1)     #预测类别
        max_prob, target_cls = torch.max(prob, dim=1)  # 获取预测类别编码target_cls///raw: target_cls = torch.argmax(prob).item()  
        print(
                "predicted action ids {}\t probability {}".format(
                    target_cls.item(), max_prob.item()
                )
            )



        weight_fc = model.state_dict()['bag_classifier.weight'][target_cls,:]  
        cam1 = getCAM(conv_out1.features,weight_fc,target_cls.item())   #[B,C=M=8,16,128,128]->conv6 [1, 1, 2, 16, 16])
        print('cam1.shape1', cam1.shape)      ##cam1: conv6 [1, 1, 2, 16, 16])//conv6[B,512,1,4,4]
        #P-Net
        # mil: cam rearrange->拼【patch】
        cam2 = getCAM(conv_out2.features,weight_fc,target_cls.item())       # [b,c,d,h,w] = [b,c,d,h,w] , [1,c]
        print('cam2.shape1', cam2.shape)      #cam2.shape1 torch.Size([1, 1, 1, 8, 8])


        ps = 32
        images_t2 = rearrange(images_t2,'b c h w d-> b c d h w') 
        images_dwi = rearrange(images_dwi,'b c h w d-> b c d h w') 
        images_in = rearrange(images_in,'b c h w d-> b c d h w')
        images_out = rearrange(images_out,'b c h w d-> b c d h w')       
        images_pre = rearrange(images_pre,'b c h w d-> b c d h w')        
        images_ap = rearrange(images_ap,'b c h w d-> b c d h w')
        images_pvp = rearrange(images_pvp,'b c h w d-> b c d h w')   #其中(b h w) 是bs*pn     [b * 192/64 * 192/64,1,64,64,64]
        images_dp = rearrange(images_dp,'b c h w d-> b c d h w')
        # img_array = rearrange(nii_img,'b c h w d-> b c d h w')


        heatmap = visualize(images_t2,images_dwi, images_in, images_out, images_pre, images_ap, images_pvp, images_dp, cam1, cam2, data_dict,target_cls)

        print('nii_img', images_pvp.shape)
        print('heatmap', heatmap.shape)
