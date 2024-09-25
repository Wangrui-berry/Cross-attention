import torch
from cmath import exp
import os
from site import abs_paths
import sys
import numpy as np

#todo
from feeder_8modal_7class import MRIDataset

from model.crossatten import generate_crossatten

from model.ablation.Rnet import generate_rnet
from model.ablation.PA_Net import generate_panet
from model.ablation.RA_Net import generate_ranet
print("===========================================================================")    
# IA_Net -> crossatten
from model.fuxian.MIL.deepganet import generate_deepganet
from model.fuxian.baseline.Song8m import SongNet
from model.fuxian.MIL.DAMIDL8m_semi import DAMIDL
from model.fuxian.baseline.brain import generate_brain
from model.fuxian.MA import generate_ma

from model.fuxian.MIL.Resmil import generate_resmil


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time
import datetime
starttime = datetime.datetime.now()+datetime.timedelta(hours=8)
starttime = starttime.strftime("%Y-%m-%d_%H_%M_%S")
import argparse

import json
from monai.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

from torch.utils.tensorboard import SummaryWriter 

# from metrics import FocalLoss
from utils.losses import BCEFocalLoss, MultiCEFocalLoss
from utils.losses import CrossEntropyLoss as WeightCE
device_now = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random


def confusion_matrix(preds, labels, conf_matrix):
    # preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def Cohen_Kappa(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.cohen_kappa_score(y_true, y_pred)

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

#######################loss-based attention(P/R weight)################
def rampup(global_step, rampup_length=24):
    if global_step <rampup_length:
        global_step = np.float(global_step)
        rampup_length = np.float(rampup_length)
        phase = 1.0 - np.maximum(0.0, global_step) / rampup_length
    else:
        phase = 0.0
    return np.exp(-5.0 * phase * phase)


def compute_micro_auc(labels, preds, args):

    # preds = torch.stack(preds)
    labels = torch.nn.functional.one_hot(labels,args.num_classes)
    if not isinstance(preds, np.ndarray):
        preds = preds.cpu().detach().numpy()
    # labels = torch.stack(labels)
    if not isinstance(labels, np.ndarray):
        labels = labels.cpu().detach().numpy()




    auc_micro_list = []

    for i in range(preds.shape[0]):
 
        fpr_micro, tpr_micro, _ = metrics.roc_curve(labels[i], preds[i])
        
        auc_micro = metrics.auc(fpr_micro, tpr_micro)
        auc_micro_list.append(auc_micro)

    auc_macro = np.array(auc_micro).mean()


    fpr_micro, tpr_micro, _ = metrics.roc_curve(labels.ravel(), preds.ravel())
    auc_micro = metrics.auc(fpr_micro, tpr_micro)

    return auc_macro, auc_micro

#######################validation################
def validate(val_loader, model, lossCE, loss_patch, optimizer,args):

    start = time.time()
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct = 0.0

        in_micro_f1 = 0.0
        in_micro_pre = 0.0       #raw micro
        in_micro_rec = 0.0        #raw micro

        out_micro_f1=0.0
        out_micro_pre = 0.0
        out_micro_rec = 0.0

        in_macro_f1=0.0
        in_macro_pre = 0.0
        in_macro_rec = 0.0

        out_macro_f1 = 0.0
        out_macro_pre = 0.0
        out_macro_rec = 0.0
        preds_collect = []
        label_collect = []
        output_collect = []

        in_weighted_f1 = 0.0
        in_weighted_pre = 0.0
        in_weighted_rec = 0.0


        kappa = 0.0
        auc1 = 0.0
        auc2 = 0.0

        conf_matrix = torch.zeros(args.num_classes,args.num_classes)          #(num_classes, num_classes)==(3,3)
        

        
        for (images_t2, images_dwi, images_in, images_out, images_pre, images_ap, images_pvp, images_dp, labels) in val_loader:

            if args.cuda:
                images_t2, images_dwi, images_in, images_out, images_pre, images_ap, images_pvp, images_dp, labels = images_t2.to(device_now,non_blocking=True), images_dwi.to(device_now,non_blocking=True), images_in.to(device_now,non_blocking=True), images_out.to(device_now,non_blocking=True), images_pre.to(device_now,non_blocking=True), images_ap.to(device_now,non_blocking=True), images_pvp.to(device_now,non_blocking=True), images_dp.to(device_now,non_blocking=True), labels.to(device_now,non_blocking=True)
            #todo:multimodal
            #3loss
            # with amp_autocast():
            # outputs = model(images_t2.to(torch.float32),images_dwi.to(torch.float32), images_in.to(torch.float32), images_out.to(torch.float32), images_pre.to(torch.float32), images_ap.to(torch.float32), images_pvp.to(torch.float32), images_dp.to(torch.float32))    #(patch_num,C=1,H,W,D)
            # outputs,_,_   = model(images_t2.to(torch.float32),images_dwi.to(torch.float32), images_in.to(torch.float32), images_out.to(torch.float32), images_pre.to(torch.float32), images_ap.to(torch.float32), images_pvp.to(torch.float32), images_dp.to(torch.float32))    #(patch_num,C=1,H,W,D)
            
            outputs,_,_,_,_,_,_  = model(images_t2.to(torch.float32),images_dwi.to(torch.float32), images_in.to(torch.float32), images_out.to(torch.float32), images_pre.to(torch.float32), images_ap.to(torch.float32), images_pvp.to(torch.float32), images_dp.to(torch.float32))    #(patch_num,C=1,H,W,D)


            loss = lossCE(outputs, labels.long())

            val_loss += loss.item()
            _, preds = outputs.max(1)

            preds_collect.append(preds)
            label_collect.append(labels)
            output_collect.append(outputs)

            correct += preds.eq(labels).sum().float().item()


            in_micro_pre += metrics.precision_score(labels.tolist(), preds.tolist(),average='micro')
            in_micro_rec += metrics.recall_score(labels.tolist(), preds.tolist(),average='micro')
            in_micro_f1 += metrics.f1_score(labels.tolist(), preds.tolist(),average='micro')


            in_macro_pre += metrics.precision_score(labels.tolist(), preds.tolist(),average='macro')
            in_macro_rec += metrics.recall_score(labels.tolist(), preds.tolist(),average='macro')
            in_macro_f1 += metrics.f1_score(labels.tolist(), preds.tolist(),average='macro')



            in_weighted_pre += metrics.precision_score(labels.tolist(), preds.tolist(),average='weighted')
            in_weighted_rec += metrics.recall_score(labels.tolist(), preds.tolist(),average='weighted')
            in_weighted_f1 += metrics.f1_score(labels.tolist(), preds.tolist(),average='weighted')

            
            kappa += Cohen_Kappa(outputs.cpu().detach(), labels.cpu().detach())
    

            auc_macro, auc_micro = compute_micro_auc(labels, outputs,args)
            
            auc1 += auc_macro
            auc2 += auc_micro

        finish = time.time()



        in_micro_f1 = in_micro_f1 / len(val_loader)
        in_micro_pre = in_micro_pre / len(val_loader)
        in_micro_rec = in_micro_rec / len(val_loader)

        in_macro_pre = in_macro_pre / len(val_loader)
        in_macro_rec = in_macro_rec / len(val_loader)
        in_macro_f1 = in_macro_f1 / len(val_loader)


        in_weighted_f1 = in_weighted_f1 / len(val_loader)
        in_weighted_pre = in_weighted_pre / len(val_loader)
        in_weighted_rec = in_weighted_rec / len(val_loader)   

        label_collect = torch.cat(label_collect, dim=0).cpu().detach().cpu().numpy()
        preds_collect = torch.cat(preds_collect, dim=0).cpu().detach().cpu().numpy()
        output_collect = torch.cat(output_collect, dim=0).cpu().detach().cpu().numpy()


        out_micro_pre = metrics.precision_score(label_collect, preds_collect,average='micro')
        out_micro_rec = metrics.recall_score(label_collect, preds_collect,average='micro')
        out_micro_f1 = metrics.f1_score(label_collect, preds_collect,average='micro')

        out_macro_pre = metrics.precision_score(label_collect, preds_collect,average='macro')
        out_macro_rec = metrics.recall_score(label_collect, preds_collect,average='macro')
        out_macro_f1 = metrics.f1_score(label_collect, preds_collect,average='macro')

        out_weighted_pre = metrics.precision_score(label_collect, preds_collect,average='weighted')
        out_weighted_rec = metrics.recall_score(label_collect, preds_collect,average='weighted')
        out_weighted_f1 = metrics.f1_score(label_collect, preds_collect,average='weighted')

        print("val_loader",len(val_loader))
        print("val_loader",len(val_loader.dataset))
        val_acc = correct / len(val_loader.dataset)

        kappa = kappa / len(val_loader)
        auc1 = auc1 / len(val_loader)
        auc2 = auc2 / len(val_loader)
    return val_loss, val_acc, in_micro_f1,in_micro_pre,in_micro_rec, out_micro_f1, out_micro_pre, out_micro_rec, in_macro_f1, in_macro_pre, in_macro_rec, out_macro_f1, out_macro_pre, out_macro_rec, in_weighted_f1,in_weighted_pre, in_weighted_rec, out_weighted_f1,out_weighted_pre, out_weighted_rec,kappa,auc1,auc2#,external_auc3,external_auc4,external_auc5,external_auc6



if __name__ == '__main__':

    print("Start experiment:", starttime)
    seed = 5

    from monai.utils import set_determinism
    set_determinism(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True,warn_only=True)
    
    parser = argparse.ArgumentParser()
# Dataset parameters
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--csv_path', default='', type=str)
    parser.add_argument('--num_classes', type=int, default=7, metavar='N',
                    help='number of label classes (Model default if None)')
    parser.add_argument('--result_dir', default='', type=str)
    parser.add_argument('--json_dir', default='', type=str)
    parser.add_argument('--csv_dir', default='', type=str)
        
# Model parameters
    parser.add_argument('--net', type=str, default="crossatten", 
                        help='[brain, Song,DAMIDL,deepganet];[ours: (Rbase, RANet,PANet),3dres/crossatten]')
    parser.add_argument('--cuda',type=str_to_bool,default=True, 
                        help='load in GPU')
#Pretrained model
    parser.add_argument('--resume',type=str_to_bool,default=True,
                        help='read model.state.dict()')
    parser.add_argument('--weight_path',default='/wangrui/MVI/code/MVI3d_multimodal_0320/results/2023-03-29_15_59_04.pth',help='2023-04-03_23_51_03_epoch500; load model path: results/2023-03-29_15_59_04.pth')  
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
    parser.add_argument('--num_epoch', type=int, default=6, metavar='N')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
    
    parser.add_argument('--n_fold', type=str, default='5fold', 
                        help='1fold/5fold')
    args = parser.parse_args()
    
    exp_info = vars(args)
    print("exp_info:", exp_info)

    # writer = SummaryWriter(log_dir=syspath+"/log/"+starttime[:10]+"/"+starttime[11:],flush_secs=60)

    if args.net == "brain":
        model = generate_brain(34)
        print("batch_size=",args.batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        print("brain baseline:","resnet34, generate_brain")

        if args.resume:
            print("Load the best.pt model",args.weight_path)

            checkpoint = torch.load(args.weight_path)
            raw_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])



            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            best_f1 = checkpoint['best_f1']
            best_acc = checkpoint['best_acc']
            model.eval()
            print("------Load - Model -Done!------Have - Pretrain:",raw_epoch,"epochs")





    elif args.net == "Rbase":
        model = generate_rnet(10)
        print("batch_size=",args.batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        print("resnet 3d baseli  ne:","generate_resbase")
        if args.resume:
            print("Load the best.pt model",args.weight_path)

            checkpoint = torch.load(args.weight_path)
            raw_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])


            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            best_f1 = checkpoint['best_f1']
            best_acc = checkpoint['best_acc']
            model.eval()
            print("------Load - Model -Done!------Have - Pretrain:",raw_epoch,"epochs")


    elif args.net == "Song":
        model = SongNet(in_channels=1, n_classes=7, feature_depth=None, bn_momentum=0.05)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        print("SongNet,batch_size =1")

    elif args.net =="deepganet":
        model = generate_deepganet(18)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        print("deepganet:,patch_num=16")
        if args.resume:
            print("Load the best.pt model",args.weight_path)
            checkpoint = torch.load(args.weight_path)
            raw_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])


            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            best_f1 = checkpoint['best_f1']
            best_acc = checkpoint['best_acc']
            model.eval()
            print("------Load - Model -Done!------Have - Pretrain:",raw_epoch,"epochs")


    elif args.net =="DAMIDL":
        model = DAMIDL(patch_num=4, feature_depth=None, num_classes=7)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        print("DAMIDL:,patch_num=16,batch_size =4")

        if args.resume:
            print("Load the best.pt model",args.weight_path)
 
            checkpoint = torch.load(args.weight_path)
            raw_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model']) 
            optimizer.load_state_dict(checkpoint['optimizer'])
 


            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            best_f1 = checkpoint['best_f1']
            best_acc = checkpoint['best_acc']
            model.eval()
            print("------Load - Model -Done!------Have - Pretrain:",raw_epoch,"epochs")


    elif args.net =="ma":
        model = generate_ma(10)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        print("generate_ma:,patch_num=16")
        if args.resume:
            print("Load the best.pt model",args.weight_path)
 
            checkpoint = torch.load(args.weight_path)
            raw_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model']) 
            optimizer.load_state_dict(checkpoint['optimizer'])
 


            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            best_f1 = checkpoint['best_f1']
            best_acc = checkpoint['best_acc']
            model.eval()
            print("------Load - Model -Done!------Have - Pretrain:",raw_epoch,"epochs")


    elif args.net =="resmil":
        model = generate_resmil(10)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        print("generate_resmil")
        if args.resume:
            print("Load the best.pt model",args.weight_path)
 
            checkpoint = torch.load(args.weight_path)
            raw_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model']) 
            optimizer.load_state_dict(checkpoint['optimizer'])
 


            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            best_f1 = checkpoint['best_f1']
            best_acc = checkpoint['best_acc']
            model.eval()
            print("------Load - Model -Done!------Have - Pretrain:",raw_epoch,"epochs")


    elif args.net =="panet":
        model = generate_panet(10)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        print("PANET:,patch_num=32,batch_size =8")
        if args.resume:
            print("Load the best.pt model",args.weight_path)
 
            checkpoint = torch.load(args.weight_path)
            raw_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model']) 
            optimizer.load_state_dict(checkpoint['optimizer'])
 


            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            best_f1 = checkpoint['best_f1']
            best_acc = checkpoint['best_acc']
            model.eval()
            print("------Load - Model -Done!------Have - Pretrain:",raw_epoch,"epochs")

    elif args.net =="ranet":
        model = generate_ranet(10)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        print("RANET:,patch_num=32,batch_size =8")
        if args.resume:
            print("Load the best.pt model",args.weight_path)
 
            checkpoint = torch.load(args.weight_path)
            raw_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model']) 
            optimizer.load_state_dict(checkpoint['optimizer'])
 


            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            best_f1 = checkpoint['best_f1']
            best_acc = checkpoint['best_acc']
            model.eval()
            print("------Load - Model -Done!------Have - Pretrain:",raw_epoch,"epochs")

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
 


            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            best_f1 = checkpoint['best_f1']
            best_acc = checkpoint['best_acc']
            model.eval()
            print("------Load - Model -Done!------Have - Pretrain:",raw_epoch,"epochs")

    elif args.net == "crossatten":
        model = generate_crossatten(34)
        print("batch_size=",args.batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        print("3dres net as baseline:","generate_model")

        if args.resume:
            print("Load the best.pt model",args.weight_path)
 
            checkpoint = torch.load(args.weight_path)
            raw_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model']) 
            optimizer.load_state_dict(checkpoint['optimizer'])
 


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
    # lossCE= torch.nn.CrossEntropyLoss()
    lossCE= MultiCEFocalLoss(class_num=7,device_now=device_now)
    #######################loss-based attention(R-Net)###############
    loss_patch = WeightCE(aggregate='sum',device_now=device_now)


    val_dataset = MRIDataset(args, flag='test')# MRIDataset(**config['dataset']['val'])


    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=6,worker_init_fn=np.random.seed(seed),pin_memory=True)       #worker_init_fn = worker_init_fn(42),
  
    best_epoch = 0
    best_acc = 0.0
    best_f1 = 0.0
    acc_kappa = 0.0
    for epoch in range(0, args.num_epoch):
        
        scheduler.step()
        
        print("lr of epoch", epoch, "=>", scheduler.get_last_lr())
        val_loss, val_acc, in_micro_f1, in_micro_pre,in_micro_rec, out_micro_f1, out_micro_pre,out_micro_rec, in_macro_f1, in_macro_pre, in_macro_rec, out_macro_f1, out_macro_pre, out_macro_rec, in_weighted_f1,in_weighted_pre, in_weighted_rec, out_weighted_f1,out_weighted_pre, out_weighted_rec,kappa, auc1, auc2 = validate(val_loader, model, lossCE, loss_patch, optimizer, args)
 

        if in_micro_f1 > best_f1:
            best_f1 = in_micro_f1
        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_f1 = in_micro_f1
            best_epoch = epoch + 1
            best_auc1 = auc1
            best_auc2 = auc2
            acc_kappa = kappa
            torch.save({
                'epoch': best_epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'best_f1': in_micro_f1,
                'optimizer': optimizer.state_dict(),
                },
                os.path.join(args.result_dir.format(model), starttime[:19] +'.pth'))
        print("Starttime:",starttime)
        print("epoch",epoch)
        print("best_acc",best_acc)
        print("best_f1",best_f1)

        print("in_mic_f1",in_micro_f1)
        print("in_micro_pre",in_micro_pre)
        print("in_micro_recall",in_micro_rec)

        print("out_micro_f1",out_micro_f1)
        print("out_micro_pre",out_micro_pre)
        print("out_micro_recall",out_micro_rec)

        print("in_macro_f1",in_macro_f1)
        print("in_macro_pre",in_macro_pre)
        print("in_macro_recall",in_macro_rec)

        print("out_macro_f1",out_macro_f1)
        print("out_macro_pre",out_macro_pre)
        print("out_macro_recall",out_macro_rec)

        print("in_weighted_f1",in_weighted_f1)
        print("in_weighted_pre",in_weighted_pre)
        print("in_weighted_recall",in_weighted_rec)

        print("out_weighted_f1",out_weighted_f1)
        print("out_weighted_pre",out_weighted_pre)
        print("out_weighted_recall",out_weighted_rec)

        print("best_acc_f1",best_acc_f1)
        print("acc_kappa",acc_kappa)
        print("micro_auc",best_auc1)
        print("macro_auc",best_auc2)

        

        eval_metrics = {'weight_path':args.weight_path,
                        'Starttime': starttime, 
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
                        "in_micro_f1": in_micro_f1,
                        'in_micro_precision': in_micro_pre,
                        'in_micro_recall': in_micro_rec,
                        "out_micro_f1": out_micro_f1,
                        'out_micro_precision': out_micro_pre,
                        'out_micro_Recall':out_micro_rec,
                        'in_macro_f1': in_macro_f1,
                        'in_macro_precision': in_macro_pre,
                        'in_macro_Recall': in_macro_rec,
                        'out_macro_f1': out_macro_f1,
                        'out_macro_precision': out_macro_pre,
                        'out_acro_Recall': out_macro_rec,
                        'in_weighted_f1': in_weighted_f1,
                        'in_weighted_precision': in_weighted_pre,
                        'in_weighted_recall': in_weighted_rec,
                        'out_weighted_f1': out_weighted_f1,
                        'out_weighted_precision': out_weighted_pre,
                        'out_weighted_recall': out_weighted_rec,
                        'micro_auc': best_auc1,
                        'macro_auc': best_auc2,
                        'kappa':acc_kappa
                        }
        json_str = json.dumps(eval_metrics, indent=1)
        with open(os.path.join(args.result_dir, starttime[:10] +'_'+ args.json_dir), 'a') as f:
            f.write(json_str)
        with open(os.path.join(args.result_dir, args.csv_dir), 'a') as f:
            import csv
            csv_write = csv.writer(f)
            data_row = ["model","test","acc","best_acc_f1","in_micro_f1","in_micro_pre","in_micro_rec","out_micro_f1","out_micro_pre", "out_micro_rec","in_macro_f1","in_macro_pre", "in_macro_rec","out_macro_f1","out_macro_pre", "out_macro_rec","in_weighted_f1","in_weighted_pre", "in_weighted_rec","out_weighted_f1","out_weighted_pre", "out_weighted_rec","kappa","microauc","macroauc"]
            csv_write.writerow(data_row)
            data_row = [str(args.result_dir.split('/')[-2:]),"test",best_acc, best_acc_f1,in_micro_f1,in_micro_pre,in_micro_rec, out_micro_f1, out_micro_pre, out_micro_rec, in_macro_f1, in_macro_pre, in_macro_rec, out_macro_f1, out_macro_pre, out_macro_rec, in_weighted_f1,in_weighted_pre, in_weighted_rec, out_weighted_f1,out_weighted_pre, out_weighted_rec,kappa,best_auc1,best_auc2]
            csv_write.writerow(data_row)