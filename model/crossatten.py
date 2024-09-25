import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange,repeat

def get_inplanes():
    return [64, 128, 256, 512] #[32,64,128,256] #


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

class  Attention_Layer(nn.Module):
    def __init__(self, ):
        super(Attention_Layer, self).__init__()
    def forward(self, x, w, bias, gamma):   #[bs,pn,C]
        out = x.contiguous().view(x.size(0)*x.size(1), x.size(2))

        out_f = F.linear(out, w, bias)  #out_f,bs*pn*7 = b,pn,7

        out = out_f.view(x.size(0),x.size(1), out_f.size(1))    #[bs,pn,7]

        out= torch.sqrt((out**2).sum(2))

        alpha_01 = out /out.sum(1, keepdim=True).expand_as(out)

        alpha_01 = F.relu(alpha_01- 0.1/float(gamma))
        
        alpha_01 = alpha_01/alpha_01.sum(1, keepdim=True).expand_as(alpha_01)

        alpha = torch.unsqueeze(alpha_01, dim=2)    #bs,pn,1
        out = alpha.expand_as(x)*x 

        return out, out_f, alpha_01
    

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
        self.layer3_patch_com = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4_patch_com = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)


        self.conv1_compose = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 3, 3),
                               stride=(conv1_t_stride, 1,1),
                               padding=(conv1_t_size // 2, 1,1),
                               bias=False)
        self.bn1_compose = nn.BatchNorm3d(self.in_planes)
        self.relu_compose = nn.ReLU(inplace=True)
        self.maxpool_compose = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1_compose = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2_compose = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)

        self.layer3_compose = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4_compose = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.attention_layer = Attention_Layer()


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
        self.K = 1     #n_classes
        self.J = 1


        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bag_classifier = nn.Linear(block_inplanes[3]*self.K*self.J, n_classes)

        self.linear1 = nn.Linear(block_inplanes[3]*self.K*self.J, n_classes)
        self.linear2 = nn.Linear(block_inplanes[3]*self.K*self.J, n_classes)


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

        b,c,h,w,d = x_pvp.shape     #batch,channel=1,whole_image_shape      [b,1,192,192,64] 
        x_compose = torch.cat((x_in, x_adc, x_b500, x_ap,x_pvp,x_dp,x_t1,x_t2),dim=1).permute(0,1,4,2,3)

    ################_compose###################

        x_compose = self.conv1_compose(x_compose)
        x_compose = self.bn1_compose(x_compose)
        x_compose = self.relu_compose(x_compose)

        x_compose = self.layer1_compose(x_compose)
        x_compose = self.layer2_compose(x_compose)
        x_compose = self.layer3_compose(x_compose)
        x_compose = self.layer4_compose(x_compose) 
        gamma = x_compose.size(2)*x_compose.size(3)*x_compose.size(4)
        if self.expansion>1:

            x_compose = F.relu(self.bn4(self.conv4(x_compose)))   
        
        x_compose = F.relu(self.bn5(self.conv5(x_compose)))
        x_compose = F.relu(self.bn6(self.conv6(x_compose)))
        x_compose = x_compose.view(x_compose.size(0), x_compose.size(1),-1)
        x_compose = x_compose.transpose(2,1)

        x_compose, R_compose, alpha_compose = self.attention_layer(x_compose, self.linear1.weight, self.linear1.bias, gamma)
        x_compose = x_compose.transpose(2,1).sum(2)
        x_compose = x_compose.view(x_compose.size(0),-1)


#########################patch_compose########################

        ps = 32        # 设定ps=patch size
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
        x_patch_com = self.layer3_patch_com(x_patch_com)
        x_patch_com = self.layer4_patch_com(x_patch_com)
        x_patch_com = self.avgpool(x_patch_com).view(b, int(h*w/ps/ps), -1)

        x_patch_com, P_patch_com, alpha_patch_com= self.attention_layer(x_patch_com, self.linear2.weight, self.linear2.bias, int(h*w/ps/ps))
        x_patch_com = x_patch_com.transpose(2,1).sum(2)
        x_patch_com = x_patch_com.view(x_patch_com.size(0),-1)

################RA + PA#############
        out_R = self.linear1(x_compose)
        out_P = self.linear2(x_patch_com)

        x_multi = x_compose + x_patch_com

        bag_preds = self.bag_classifier(x_multi)

        return bag_preds, out_R, R_compose, alpha_compose, out_P, P_patch_com, alpha_patch_com


def generate_crossatten(model_depth, **args):
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
