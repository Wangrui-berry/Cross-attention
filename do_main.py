# Code of the paper Cross-Attention Guided Loss-Based Deep Dual-Branch Fusion Network for Liver Tumor Classification
# If any question, please contact the author.
# https://github.com/Wangrui-berry/Cross-attention.

import torch
from cmath import exp
import os
from site import abs_paths
import sys
import numpy as np

from feeder_8modal_7class import MRIDataset

from model.crossatten import generate_crossatten
from model.ablation.Rnet import generate_rnet
from model.ablation.PA_Net import generate_panet
print("===========================================================================")    
# IA_Net -> crossatten
from model.fuxian.MIL.deepganet import generate_deepganet
from model.fuxian.MIL.DAMIDL8m_semi import DAMIDL
from model.fuxian.MIL.Resmil import generate_resmil


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time
import datetime
starttime = datetime.datetime.now()+datetime.timedelta(hours=8)
starttime = starttime.strftime("%Y-%m-%d_%H_%M_%S")

import argparse
import yaml
syspath = os.path.dirname(os.path.abspath(__file__))


import json
from monai.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

from torch.utils.tensorboard import SummaryWriter 


from utils.losses import BCEFocalLoss, MultiCEFocalLoss
from utils.losses import CrossEntropyLoss as WeightCE
device_now = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

#######################loss-based attention(P/I weight)################
def rampup(global_step, rampup_length=24):
    if global_step <rampup_length:
        global_step = np.float(global_step)
        rampup_length = np.float(rampup_length)
        phase = 1.0 - np.maximum(0.0, global_step) / rampup_length
    else:
        phase = 0.0
    return np.exp(-5.0 * phase * phase)

def train(train_loader, model, lossCE, loss_patch, optimizer,args):
    
    start = time.time()
    model.train()
    train_loss = 0
    correct = 0
    train_pre = 0
    train_rec = 0
    train_f1 = 0

    rampup_value = rampup(epoch)

    if epoch==0:
        u_w = 0
    else:
        u_w = 0.2*rampup_value

    u_w = torch.autograd.Variable(torch.FloatTensor([u_w]).cuda(), requires_grad=False) 

    for images_t2, images_dwi, images_in, images_out, images_pre, images_ap, images_pvp, images_dp, label in train_loader:
        
        if args.cuda:

            images_t2, images_dwi, images_in, images_out, images_pre, images_ap, images_pvp, images_dp, labels = images_t2.to(device_now,non_blocking=True), images_dwi.to(device_now,non_blocking=True), images_in.to(device_now,non_blocking=True), images_out.to(device_now,non_blocking=True), images_pre.to(device_now,non_blocking=True), images_ap.to(device_now,non_blocking=True), images_pvp.to(device_now,non_blocking=True), images_dp.to(device_now,non_blocking=True), label.to(device_now,non_blocking=True)

        #######################cross atten################
        bag_preds, out1, f_compose, alpha_compose, out2, f_patch_com, alpha_patch_com = model(images_t2.to(torch.float32),images_dwi.to(torch.float32), images_in.to(torch.float32), images_out.to(torch.float32), images_pre.to(torch.float32), images_ap.to(torch.float32), images_pvp.to(torch.float32), images_dp.to(torch.float32))      #(patch_num,C=1,H,W,D)


        loss_1 = lossCE(bag_preds, labels.long())

        loss_2_R = loss_patch(f_compose, labels.repeat(4*4,1).permute(1,0).contiguous().view(-1), weights= alpha_compose.view(-1))
        loss_2_R_alphapatchcom = loss_patch(f_compose, labels.repeat(4*4,1).permute(1,0).contiguous().view(-1), weights= alpha_patch_com.view(-1))
        loss_2_P_alphacom = loss_patch(f_patch_com, labels.repeat(4*4,1).permute(1,0).contiguous().view(-1), weights=alpha_compose.view(-1))
        loss_2_P = loss_patch(f_patch_com, labels.repeat(4*4,1).permute(1,0).contiguous().view(-1), weights=alpha_patch_com.view(-1))


        ############+sim_loss##########
        simlabel = torch.ones(alpha_compose.shape[0]).to(device_now)
        sim_loss = torch.nn.CosineEmbeddingLoss()(alpha_compose,alpha_patch_com,simlabel)
        
        loss = loss_1 + u_w*loss_2_P/bag_preds.size(0) + u_w*loss_2_R/bag_preds.size(0) + 0.25*u_w* loss_2_R_alphapatchcom +  0.25*u_w* loss_2_P_alphacom + 0.25* sim_loss  
        # loss = loss_1 + u_w*loss_2_R/bag_preds.size(0) + u_w*loss_2_R/bag_preds.size(0)   #without cross-attention

        _, predicted = torch.max(bag_preds.data, 1)

        correct += predicted.eq(labels).sum().float().item()
        train_pre += metrics.precision_score(labels.tolist(), predicted.tolist(),average='micro')
        train_rec += metrics.recall_score(labels.tolist(), predicted.tolist(),average='micro')
        train_f1 += metrics.f1_score(labels.tolist(), predicted.tolist(),average='micro')


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    


    finish = time.time()
    print('Train set: Epoch: {}, len(train_loader.dataset)): {}, Train_Accuracy: {}, Average loss: {:.4f}, Precision Score: {:.4f}, Recall Score :{:.4f}, F1 Score: {:.4f}, Time consumed:{:.2f}s'.format(
    epoch,
    len(train_loader.dataset),
    int(correct) / len(train_loader.dataset),
    train_loss / len(train_loader),
    train_pre / len(train_loader),
    train_rec / len(train_loader),
    train_f1 / len(train_loader),
    finish - start
))

    train_acc = correct / len(train_loader.dataset)
    train_f1 = train_f1 / len(train_loader)

    return train_loss, train_acc, train_f1


def validate(val_loader, model, lossCE, loss_patch, optimizer,args):

    start = time.time()
    model.eval()

    val_loss = 0.0
    correct = 0.0
    test_pre = 0.0
    test_rec =0.0
    val_f1 = 0.0

    for (images_t2, images_dwi, images_in, images_out, images_pre, images_ap, images_pvp, images_dp, labels) in val_loader:

        if args.cuda:
            images_t2, images_dwi, images_in, images_out, images_pre, images_ap, images_pvp, images_dp, labels = images_t2.to(device_now,non_blocking=True), images_dwi.to(device_now,non_blocking=True), images_in.to(device_now,non_blocking=True), images_out.to(device_now,non_blocking=True), images_pre.to(device_now,non_blocking=True), images_ap.to(device_now,non_blocking=True), images_pvp.to(device_now,non_blocking=True), images_dp.to(device_now,non_blocking=True), labels.to(device_now,non_blocking=True)

        outputs,_,_,_,_,_,_ = model(images_t2.to(torch.float32),images_dwi.to(torch.float32), images_in.to(torch.float32), images_out.to(torch.float32), images_pre.to(torch.float32), images_ap.to(torch.float32), images_pvp.to(torch.float32), images_dp.to(torch.float32))    #(patch_num,C=1,H,W,D)

        loss = lossCE(outputs, labels.long())

        val_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().float().item()

        test_pre += metrics.precision_score(labels.tolist(), preds.tolist(),average='micro')
        test_rec += metrics.recall_score(labels.tolist(), preds.tolist(),average='micro')
        val_f1 += metrics.f1_score(labels.tolist(), preds.tolist(),average='micro')



    finish = time.time()

    print('Evaluating Network.....')
    print('Test set: Epoch: {}, len(val_loader.dataset)): {}, Val_ACC: {}, Average loss: {:.4f}, Precision Score: {:.4f}, Recall Score :{:.4f}, F1 Score: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        len(val_loader.dataset),
        correct / len(val_loader.dataset),
        val_loss / len(val_loader),
        test_pre/ len(val_loader),
        test_rec/ len(val_loader),
        val_f1 / len(val_loader),
        finish - start
    ))
    val_acc = correct / len(val_loader.dataset)
    val_f1 = val_f1 / len(val_loader)

    return val_loss, val_acc, val_f1


if __name__ == '__main__':

    print("Start experiment:", starttime)
    seed = 5
    print("seed_everything:",seed)
    from monai.utils import set_determinism
    set_determinism(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True,warn_only=True)
    
    parser = argparse.ArgumentParser()
# Dataset parameters
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--csv_path', default='', type=str, 
                        help='contain both train and val data')
    parser.add_argument('--num_classes', type=int, default=7, metavar='N',
                    help='number of label classes (Model default if None)')
    parser.add_argument('--result_dir', default='', type=str, 
                        help='contain both train and val data')
    parser.add_argument('--json_dir', default='', type=str, 
                        help='contain both train and val data')
        
# Model parameters
    parser.add_argument('--net', type=str, default="crossatten", 
                        help='brain,DAMIDL,deepganet,Rbase, RANet,PANet,crossatten]')
    parser.add_argument('--cuda',type=str_to_bool,default=True, 
                        help='load in GPU')
#Pretrained model
    parser.add_argument('--resume',type=str_to_bool,default=False,
                        help='read model.state.dict()')
    parser.add_argument('--weight_path',default='',help='2023-04-03_23_51_03_epoch500; load model path: results/2023-03-29_15_59_04.pth')  
#Parallel
    #Parser.add_argument('--device',type=str,default='cuda:0',help='')
    parser.add_argument('--data_parallel', type=str_to_bool, default=False)
    # parser.add_argument('--device_id',type=list,default=[0,1],help='')
#Parameter
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='weight decay rate')
    parser.add_argument('--num_epoch', type=int, default=65, metavar='N',
                    help='number of epochs to train (default: 300)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
    
    parser.add_argument('--n_fold', type=str, default='5fold', 
                        help='1fold/5fold')
    args = parser.parse_args()
    
    exp_info = vars(args)
    print("exp_info:", exp_info)

    writer = SummaryWriter(log_dir=syspath+"/log/"+starttime[:10]+"/"+starttime[11:],flush_secs=60)

    if args.net == "Rbase":
        model = generate_rnet(10)
        print("batch_size=",args.batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        print("resnet 3d baseline:","generate_resbase")



    elif args.net =="deepganet":
        model = generate_deepganet(18)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        print("deepganet:,patch_num=16")
    elif args.net =="DAMIDL":
        model = DAMIDL(patch_num=4, feature_depth=None, num_classes=7)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        print("DAMIDL:,patch_num=16,batch_size =4")

    elif args.net == "3dres":
        model = generate_crossatten(10)
        print("batch_size=",args.batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        print("3dres net as baseline:","generate_model")
 
        if args.resume:
            print("Load the best.pt model",args.weight_path)
            checkpoint = torch.load(args.weight_path)
            raw_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
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

    print("NOT PARALLEL!!!")

    model = model.to(device_now,non_blocking=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.333, last_epoch= -1)

    #######################loss-CE###############
    # lossCE= torch.nn.CrossEntropyLoss()   # if binary classification
    lossCE= MultiCEFocalLoss(class_num=7,device_now=device_now)
    #######################loss-based attention###############
    loss_patch = WeightCE(aggregate='sum',device_now=device_now)

    train_dataset = MRIDataset(args, flag='Train')
    val_dataset = MRIDataset(args, flag='Val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=6, worker_init_fn=np.random.seed(seed),pin_memory=True)   #worker_init_fn=np.random.seed(seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=6,worker_init_fn=np.random.seed(seed),pin_memory=True)       #worker_init_fn = worker_init_fn(42),
  
    best_epoch = 0
    best_acc = 0.0
    best_f1 = 0.0

    for epoch in range(0, args.num_epoch):
        
        scheduler.step()
        
        print("lr of epoch", epoch, "=>", scheduler.get_last_lr())
        train_loss, train_acc, train_f1 = train(train_loader, model, lossCE, loss_patch, optimizer,args)
        val_loss, val_acc, val_f1 = validate(val_loader, model, lossCE, loss_patch, optimizer,args)

        writer.add_scalar('train/train_loss', train_loss, epoch)
        writer.add_scalar('train/train_acc', train_acc, epoch)
        writer.add_scalar('val/val_loss', val_loss, epoch)
        writer.add_scalar('val/val_acc', val_acc, epoch)
        writer.add_scalar('train/train_f1', train_f1, epoch)
        writer.add_scalar('val/val_f1', val_f1, epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_f1 = val_f1
            best_epoch = epoch + 1
            torch.save({
                'epoch': best_epoch,
                'model': model.state_dict(), 
                'best_acc': best_acc,
                'best_f1': val_f1,
                'optimizer': optimizer.state_dict(),
                },
                os.path.join(args.result_dir.format(model), starttime[:19] +'.pth'))
    print("Starttime:",starttime)
    print("best_epoch",best_epoch)
    print("best_acc",best_acc)
    print("best_f1",best_f1)
    print("best_acc_f1",best_acc_f1)


    eval_metrics = {'Starttime': starttime, 
                    'fold': args.n_fold,
                    'num_classes': args.num_classes,
                    'model': args.net,
                    'batch_size': args.batch_size,
                    'lr': args.lr,
                    'num_epoch': args.num_epoch,
                    'best_epoch': best_epoch,
                    'best_acc': best_acc,
                    'best_f1': best_f1,
                    'best_acc_f1': best_acc_f1,
                      }
    json_str = json.dumps(eval_metrics, indent=1)
    with open(os.path.join(args.result_dir, starttime[:10] +'_'+ args.json_dir), 'a') as f:
        f.write(json_str)