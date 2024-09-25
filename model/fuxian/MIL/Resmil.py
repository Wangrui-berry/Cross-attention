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


        self.conv1_patch_com = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 3, 3),
                               stride=(conv1_t_stride, 1,1),
                               padding=(conv1_t_size // 2, 1,1),
                               bias=False)
        self.bn1_patch_com = nn.BatchNorm3d(self.in_planes)
        self.relu_patch_com = nn.ReLU(inplace=True)
        self.maxpool_patch_com = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1_patch_com = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2_patch_com = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        
        self.L = 512
        self.D = 64
        self.K = 1
        self.J = 1

        
        self.attention = nn.Sequential(
                nn.Linear(block_inplanes[3], block_inplanes[1]),
                nn.ReLU(),#nn.Tanh(),
                nn.Linear(block_inplanes[1], self.D),
                nn.Tanh(),
                nn.Linear(self.D, self.K)
            )

        self.attention_all = nn.Sequential(
                nn.Linear(block_inplanes[3]*self.K, block_inplanes[1]),
                nn.ReLU(),
                nn.Linear(block_inplanes[1], self.D),
                nn.Tanh(),
                nn.Linear(self.D, self.J)
            )
        
        self.attention_patch_com = nn.Sequential(
                nn.Linear(block_inplanes[3], block_inplanes[1]),
                nn.ReLU(),#nn.Tanh(),
                nn.Linear(block_inplanes[1], self.D),
                nn.Tanh(),
                nn.Linear(self.D, self.K)
            )

        self.whichatten = nn.Sequential(
                nn.Linear(block_inplanes[3], block_inplanes[1]),
                nn.ReLU(),#nn.Tanh(),
                nn.Linear(block_inplanes[1], self.D),
                nn.Tanh(),
                nn.Linear(self.D, self.K)
            )
 
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bag_classifier = nn.Linear(block_inplanes[3]*self.K*self.J, n_classes)




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
        
        ps = 32
        x_in = rearrange(x_in,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps) 
        x_adc = rearrange(x_adc,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps) 
        x_b500 = rearrange(x_b500,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps)
        x_ap = rearrange(x_ap,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps)       
        x_pvp = rearrange(x_pvp,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps)
        x_dp = rearrange(x_dp,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps)        
        x_t1 = rearrange(x_t1,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps)
        x_t2 = rearrange(x_t2,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps)



        x_patch_com = torch.cat((x_in, x_adc, x_b500, x_ap,x_pvp,x_dp,x_t1,x_t2),dim=1)
        x_patch_com = self.conv1_patch_com(x_patch_com)
        x_patch_com = self.bn1_patch_com(x_patch_com)
        x_patch_com = self.relu_patch_com(x_patch_com)
        if not self.no_max_pool:
            x_patch_com = self.maxpool_patch_com(x_patch_com)

        x_patch_com = self.layer1_patch_com(x_patch_com)
        x_patch_com = self.layer2_patch_com(x_patch_com)
        x_patch_com = self.layer3(x_patch_com)
        x_patch_com = self.layer4(x_patch_com)

        x_patch_com = self.avgpool(x_patch_com).view(b, int(h*w/ps/ps), -1)

        A_patch_com = self.attention_patch_com(x_patch_com).transpose(1,2)
        A_patch_com = torch.softmax(A_patch_com, dim=2)
    
        x_patch_com = torch.bmm(A_patch_com,x_patch_com).view(b,-1)

        bag_preds = self.bag_classifier(x_patch_com)
        return bag_preds


def generate_resmil(model_depth, **args):
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
