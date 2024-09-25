import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange,repeat

def get_inplanes():
    return [64, 128, 256, 512] 


def conv3d(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def ConvBnReLU(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3d(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3d(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = ConvBnReLU(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3d(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = ConvBnReLU(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class RR3d_MIL(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=8,
                 conv1_t_size=3,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=7):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1_netb = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 3, 3),
                               stride=(conv1_t_stride, 1,1),
                               padding=(conv1_t_size // 2, 1,1),
                               bias=False)
        self.bn1_netb = nn.BatchNorm3d(self.in_planes)
        self.relu_netb = nn.ReLU(inplace=True)
        self.maxpool_netb = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1_netb = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2_netb = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3_netb = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4_netb = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)


        self.conv1_neta = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 3, 3),
                               stride=(conv1_t_stride, 1,1),
                               padding=(conv1_t_size // 2, 1,1),
                               bias=False)
        self.bn1_neta = nn.BatchNorm3d(self.in_planes)
        self.relu_neta = nn.ReLU(inplace=True)
        self.maxpool_neta = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1_neta = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2_neta = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)

        self.layer3_neta = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4_neta = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)


        self.expansion = block.expansion
        
        if self.expansion>1:
            self.conv4 = nn.Conv3d(block_inplanes[3]*self.expansion, block_inplanes[3], kernel_size=1, stride=1, padding=0, bias=False)
            self.bn4 = nn.BatchNorm3d(block_inplanes[3]) 

        self.conv5 = nn.Conv3d(block_inplanes[3], block_inplanes[3], kernel_size=(1,4,4), stride=4, padding=0, bias=False)
        self.bn5 = nn.BatchNorm3d(block_inplanes[3])

        self.conv6 = nn.Conv3d(block_inplanes[3], block_inplanes[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm3d(block_inplanes[3])


        self.L = 512
        self.D = 64
        self.K = 1
        self.J = 1


        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.bag_classifier = nn.Sequential(
            # nn.ReLU(True),
            # nn.Dropout(p=0.5),     
            nn.Linear(block_inplanes[3],512),
            nn.Dropout(p=0.2), 
            nn.Linear(512,7),
        )




        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    ConvBnReLU(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_in, x_adc, x_b500, x_ap, x_pvp, x_dp, x_t1, x_t2):

        b,c,h,w,d = x_pvp.shape  
        x_neta = torch.cat((x_in, x_adc, x_b500, x_ap,x_pvp,x_dp,x_t1,x_t2),dim=1).permute(0,1,4,2,3)
        x_netb = torch.cat((x_in, x_adc, x_b500, x_ap,x_pvp,x_dp,x_t1,x_t2),dim=1).permute(0,1,4,2,3)

    ################_neta###################

        x_neta = self.conv1_neta(x_neta)
        x_neta = self.bn1_neta(x_neta)
        x_neta = self.relu_neta(x_neta)


        x_neta = self.layer1_neta(x_neta)
        x_neta = self.layer2_neta(x_neta)
        x_neta = self.layer3_neta(x_neta)
        x_neta = self.layer4_neta(x_neta)

#########################patch_netb########################



        x_netb = self.conv1_netb(x_netb)
        x_netb = self.bn1_netb(x_netb)
        x_netb = self.relu_netb(x_netb)


        x_netb = self.layer1_netb(x_netb)
        x_netb = self.layer2_netb(x_netb)
        x_netb = self.layer3_netb(x_netb)
        x_netb = self.layer4_netb(x_netb)

    
################Cross Attention#############
        atten_neta = torch.sum(x_neta.unsqueeze(dim=2), dim=2)
        atten_neta = torch.nn.functional.normalize(atten_neta,dim=1)

        atten_netb = torch.sum(x_netb.unsqueeze(dim=2), dim=2)
        atten_netb = torch.nn.functional.normalize(atten_netb,dim=1)


 
        hadamard_atten = atten_neta * atten_netb

    ########################Neta % add % Netb########################
        x_multi = x_neta + x_netb 
     
    #########################Neta % cat % Netb#########################
        # x_multi = torch.cat((x_neta, hadamard_atten, x_netb),dim=1)
    #########################Neta % add % Netb#########################
        # x_multi = x_neta + hadamard_atten + x_netb


        x_multi = self.avgpool(x_multi).view(b,-1)    #([b, 7])



        bag_preds = self.bag_classifier(x_multi)

        return bag_preds, atten_neta, atten_netb


def generate_ma(model_depth, **args):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = RR3d_MIL(BasicBlock, [1, 1, 1, 1], get_inplanes(), **args)
    elif model_depth == 18:
        model = RR3d_MIL(BasicBlock, [2, 2, 2, 2], get_inplanes(), **args)
    elif model_depth == 34:
        model = RR3d_MIL(BasicBlock, [3, 4, 6, 3], get_inplanes(), **args)
    elif model_depth == 50:
        model = RR3d_MIL(Bottleneck, [3, 4, 6, 3], get_inplanes(), **args)
    elif model_depth == 101:
        model = RR3d_MIL(Bottleneck, [3, 4, 23, 3], get_inplanes(), **args)
    elif model_depth == 152:
        model = RR3d_MIL(Bottleneck, [3, 8, 36, 3], get_inplanes(), **args)
    elif model_depth == 200:
        model = RR3d_MIL(Bottleneck, [3, 24, 36, 3], get_inplanes(), **args)

    return model