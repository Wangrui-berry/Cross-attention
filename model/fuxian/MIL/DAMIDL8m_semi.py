import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch

from einops import rearrange,repeat
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.cs = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)        #local_feature
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_out, max_out], dim=1)
        a = self.cs(a)
        return x*a


class AttentionBlock(nn.Module):
    def __init__(self, patch_num):
        super(AttentionBlock, self).__init__()
        self.patch_num = patch_num * patch_num
        self.GlobalAveragePool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.GlobalMaxPool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))
        self.Attn = nn.Sequential(
            nn.Conv3d(self.patch_num, self.patch_num // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.patch_num // 2, self.patch_num, kernel_size=1)
        )
        self.pearson_attn = nn.Linear(self.patch_num - 1, 1)

    def forward(self, x_adc, patch_pred):
        mean_x_adc = x_adc.mean(2)      #([8, 16, 128, 1, 9, 9]) -- ([128, 1])
        attn1 = self.Attn(self.GlobalAveragePool(mean_x_adc))
        attn2 = self.Attn(self.GlobalMaxPool(mean_x_adc))       #torch.Size([8, 16, 1, 1, 1])
        patch_pred = patch_pred.unsqueeze(-1)
        patch_pred = patch_pred.unsqueeze(-1)
        patch_pred = patch_pred.unsqueeze(-1)
        a = attn1 + attn2 + patch_pred
        a = torch.sigmoid(a)
        #bug:kanyixia ,mean_x_adc and a
        return mean_x_adc*a, a.flatten(1)


class BaseNet(nn.Module):
    def __init__(self, feature_depth):
        super(BaseNet, self).__init__()
        self.feature_depth = feature_depth
        self.spatial_attention = SpatialAttention()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, self.feature_depth[0], kernel_size=4)),
            ('norm1', nn.BatchNorm3d(self.feature_depth[0])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(self.feature_depth[0], self.feature_depth[1], kernel_size=3)),
            ('norm2', nn.BatchNorm3d(self.feature_depth[1])),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool3d(kernel_size=2)),
            ('conv3', nn.Conv3d(self.feature_depth[1], self.feature_depth[2], kernel_size=3)),
            ('norm3', nn.BatchNorm3d(self.feature_depth[2])),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv3d(self.feature_depth[2], self.feature_depth[3], kernel_size=3)),
            ('norm4', nn.BatchNorm3d(self.feature_depth[3])),
            ('relu4', nn.ReLU(inplace=True)),
        ]))
        self.classify = nn.Sequential(
            nn.Linear(self.feature_depth[3], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        local_feature = self.features(x)
        attended_feature = self.spatial_attention(local_feature)    #(512,128,1,1,1)
        feature_ = F.adaptive_avg_pool3d(local_feature, (1, 1, 1))      #([128, 128, 1, 1, 1])
        score = self.classify(feature_.flatten(1, -1))  #feature_.flatten(1, -1).shape=[128=b*p,128=c]
        return [attended_feature, score]


class DAMIDL(nn.Module):
    def __init__(self, patch_num=None, feature_depth=None, num_classes=None):
        super(DAMIDL, self).__init__()
        self.patch_num = patch_num
        print("self.patch_num:",self.patch_num)
        if feature_depth is None:
            feature_depth = [64,128,256,512]#[32,64,128,256] #[32, 64, 128, 128]#
        self.patch_net_in = BaseNet(feature_depth)
        self.patch_net_adc = BaseNet(feature_depth)
        self.patch_net_b500 = BaseNet(feature_depth)
        self.patch_net_ap = BaseNet(feature_depth)
        self.patch_net_pvp = BaseNet(feature_depth)
        self.patch_net_dp = BaseNet(feature_depth)
        self.patch_net_t1 = BaseNet(feature_depth)
        self.patch_net_t2 = BaseNet(feature_depth)




        self.attention_net = AttentionBlock(self.patch_num)
        self.reduce_channels = nn.Sequential(
            nn.Conv3d(self.patch_num * self.patch_num, 128, kernel_size=(1,2,2)),#RR-change kern_size
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.Conv3d(128, 64, kernel_size=(1,2,2)),#RR-change kern_size
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1),
        )

        # self.fc = nn.Linear(64, num_classes)
            # nn.Softmax(dim=1),
        
        self.attention_all = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )
    def forward(self, x_in, x_adc, x_b500, x_ap, x_pvp, x_dp, x_t1, x_t2):
        
        b,c,h,w,d = x_pvp.shape     #batch,channel=1,whole_image_shape      [b,1,192,192,64] 
        ps = int(h/ (self.patch_num))       # 设定ps=patch size; patch_num=8,so ps =16
        # print("b",b)
        # print("ps",ps)      #32
    #########################input#####################
        x_in = rearrange(x_in,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps) #[8, 1, 128, 128, 16])->[512, 1, 16, 16, 16])
        x_adc = rearrange(x_adc,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps) 
        x_b500 = rearrange(x_b500,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps)
        x_ap = rearrange(x_ap,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps)       
        x_pvp = rearrange(x_pvp,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps)   #其中(b h w) 是bs*pn     [b * 192/64 * 192/64,1,64,64,64]
        x_dp = rearrange(x_dp,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps)        
        x_t1 = rearrange(x_t1,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps)
        x_t2 = rearrange(x_t2,'b c (h p1) (w p2) d-> (b h w) c d p1 p2', p1=ps, p2=ps)   
        ################in###################

        feature, score = self.patch_net_in(x_in)    #[512, C=128, 1, 1, 1]); score:[128,128]

        b_,c_,d_,h_,w_= feature.shape
        feature_maps = feature.view(b,(self.patch_num)*(self.patch_num),c_,d_,h_,w_)     #patch_feature.unsqueeze(1).view(b,-1)
        
        patch_scores = score.view(b,(self.patch_num)*(self.patch_num))  #torch.Size([8, 16])
        attn_feat_in, ca_in = self.attention_net(feature_maps, patch_scores)        #torch.Size([8, 16, 1, 9, 9])=b,pn,d,h,w
        feature_in = self.reduce_channels(attn_feat_in).flatten(1)  #torch.Size([8, 64, 1, 1, 1])->[8,64]=b,c

        ################adc###################
        feature, score = self.patch_net_adc(x_adc)    #[512, C=128, 1, 1, 1])

        b_,c_,d_,h_,w_= feature.shape
        feature_maps = feature.view(b,(self.patch_num)*(self.patch_num),c_,d_,h_,w_)     #patch_feature.unsqueeze(1).view(b,-1)
        
        patch_scores = score.view(b,(self.patch_num)*(self.patch_num))  #torch.Size([8, 16])
        attn_feat_adc, ca_adc = self.attention_net(feature_maps, patch_scores)        #torch.Size([8, 16, 1, 9, 9])=b,pn,d,h,w
        feature_adc = self.reduce_channels(attn_feat_adc).flatten(1)  #torch.Size([8, 64, 1, 1, 1])->[8,64]=b,c

        
        ################b500###################
        feature, score = self.patch_net_b500(x_b500)    #[512, C=128, 1, 1, 1])

        b_,c_,d_,h_,w_= feature.shape
        feature_maps = feature.view(b,(self.patch_num)*(self.patch_num),c_,d_,h_,w_)     #patch_feature.unsqueeze(1).view(b,-1)
        
        patch_scores = score.view(b,(self.patch_num)*(self.patch_num))  #torch.Size([8, 16])
        attn_feat_b500, ca_b500 = self.attention_net(feature_maps, patch_scores)        #torch.Size([8, 16, 1, 9, 9])=b,pn,d,h,w
        feature_b500 = self.reduce_channels(attn_feat_b500).flatten(1)  #torch.Size([8, 64, 1, 1, 1])->[8,64]=b,c

        
        ################ap###################
        feature, score = self.patch_net_ap(x_ap)    #[512, C=128, 1, 1, 1])

        b_,c_,d_,h_,w_= feature.shape
        feature_maps = feature.view(b,(self.patch_num)*(self.patch_num),c_,d_,h_,w_)     #patch_feature.unsqueeze(1).view(b,-1)
        
        patch_scores = score.view(b,(self.patch_num)*(self.patch_num))  #torch.Size([8, 16])
        attn_feat_ap, ca_ap = self.attention_net(feature_maps, patch_scores)        #torch.Size([8, 16, 1, 9, 9])=b,pn,d,h,w
        feature_ap = self.reduce_channels(attn_feat_ap).flatten(1)  #torch.Size([8, 64, 1, 1, 1])->[8,64]=b,c

        ################pvp###################
        feature, score = self.patch_net_pvp(x_pvp)    #[512, C=128, 1, 1, 1])

        b_,c_,d_,h_,w_= feature.shape
        feature_maps = feature.view(b,(self.patch_num)*(self.patch_num),c_,d_,h_,w_)     #patch_feature.unsqueeze(1).view(b,-1)
        
        patch_scores = score.view(b,(self.patch_num)*(self.patch_num))  #torch.Size([8, 16])
        attn_feat_pvp, ca_pvp = self.attention_net(feature_maps, patch_scores)        #torch.Size([8, 16, 1, 9, 9])=b,pn,d,h,w
        feature_pvp = self.reduce_channels(attn_feat_pvp).flatten(1)  #torch.Size([8, 64, 1, 1, 1])->[8,64]=b,c


        ################dp###################
        feature, score = self.patch_net_dp(x_dp)    #[512, C=128, 1, 1, 1])

        b_,c_,d_,h_,w_= feature.shape
        feature_maps = feature.view(b,(self.patch_num)*(self.patch_num),c_,d_,h_,w_)     #patch_feature.unsqueeze(1).view(b,-1)
        
        patch_scores = score.view(b,(self.patch_num)*(self.patch_num))  #torch.Size([8, 16])
        attn_feat_dp, ca_dp = self.attention_net(feature_maps, patch_scores)        #torch.Size([8, 16, 1, 9, 9])=b,pn,d,h,w
        feature_dp = self.reduce_channels(attn_feat_dp).flatten(1)  #torch.Size([8, 64, 1, 1, 1])->[8,64]=b,c


        ################t1###################
        feature, score = self.patch_net_t1(x_t1)    #[512, C=128, 1, 1, 1])

        b_,c_,d_,h_,w_= feature.shape
        feature_maps = feature.view(b,(self.patch_num)*(self.patch_num),c_,d_,h_,w_)     #patch_feature.unsqueeze(1).view(b,-1)
        
        patch_scores = score.view(b,(self.patch_num)*(self.patch_num))  #torch.Size([8, 16])
        attn_feat_t1, ca_t1 = self.attention_net(feature_maps, patch_scores)        #torch.Size([8, 16, 1, 9, 9])=b,pn,d,h,w
        feature_t1 = self.reduce_channels(attn_feat_t1).flatten(1)  #torch.Size([8, 64, 1, 1, 1])->[8,64]=b,c


        ################t2###################
        feature, score = self.patch_net_t2(x_t2)    #[512, C=128, 1, 1, 1])

        b_,c_,d_,h_,w_= feature.shape
        feature_maps = feature.view(b,(self.patch_num)*(self.patch_num),c_,d_,h_,w_)     #patch_feature.unsqueeze(1).view(b,-1)
        
        patch_scores = score.view(b,(self.patch_num)*(self.patch_num))  #torch.Size([8, 16])
        attn_feat_t2, ca_t2 = self.attention_net(feature_maps, patch_scores)        #torch.Size([8, 16, 1, 9, 9])=b,pn,d,h,w
        feature_t2 = self.reduce_channels(attn_feat_t2).flatten(1)  #torch.Size([8, 64, 1, 1, 1])->[8,64]=b,c

        
    ##################################(三)###################################
        feature_multi = torch.cat((feature_in.unsqueeze(1), feature_adc.unsqueeze(1), feature_b500.unsqueeze(1), feature_ap.unsqueeze(1),feature_pvp.unsqueeze(1), feature_dp.unsqueeze(1), feature_t1.unsqueeze(1), feature_t2.unsqueeze(1)),dim=1)        #[B,3,C]
        modal_A = self.attention_all(feature_multi).transpose(1,2)      #[b,8,64]
        modal_A = torch.softmax(modal_A, dim=2) #[B,1,]

        # print("modal_A:",modal_A.shape)
        feature_multi = torch.bmm(modal_A,feature_multi).view(b,-1)   #[B,C]=[4,64]


        

        subject_pred = self.fc(feature_multi)

        return subject_pred
