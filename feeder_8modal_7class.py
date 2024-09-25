from monai.utils import GridSampleMode, GridSamplePadMode,InterpolateMode,NumpyPadMode
from monai.transforms import RandGaussianNoise, SqueezeDim, Rand3DElastic, RandGaussianSmooth, RandZoom, RandAffine, LoadImage, AddChannel, ToTensor, Resize, NormalizeIntensity, RandRotate90, RandFlip
from monai.transforms import Compose, SpatialCropd, CenterSpatialCropd, RandSpatialCropd, Rand3DElasticd, RandGaussianNoised, SqueezeDimd, \
                             LoadImaged, AddChanneld, EnsureChannelFirstd, ToTensord, Resized, NormalizeIntensityd, \
                             RandRotate90d, RandFlipd, ConcatItemsd, RandZoomd, RandGaussianSmoothd, AdjustContrastd,RandAdjustContrastd,\
                             RandAffined, ScaleIntensityd, RandSpatialCropSamplesd,RandRotated,ScaleIntensityRanged,RandScaleCropd,RandSpatialCropd
import numpy as np
import monai
import os


class MRIDataset(monai.data.Dataset):     #train/test
    
    def __init__(self,
                args,
                flag):
      
        self.csv_path = args.csv_path
        self.data_dir = args.data_dir
        self.flag = flag    
        self.data_dict = self.load_data()

        self.train_transform = Compose(
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

        self.val_transform = Compose(
            [
                LoadImaged(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp"), image_only=True),
                EnsureChannelFirstd(keys=["t2","dwi","in_","out_","pre","ap","pvp","dp"]),
                Resized(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp"), spatial_size=(128,128,16)),
                NormalizeIntensityd(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp")),
                ToTensord(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp"))
            ])
        
 
    def load_data(self):
      
        t2pathlst = []
        dwipathlst = []
        in_pathlst = []
        out_pathlst = []
        prepathlst = []
        apathlst = []
        pvpathlst = []
        dpathlst = []
        self.labellst = []
        phase_list = ['T2WI', 'DWI', 'In Phase', 'Out Phase', 'C-pre', 'C+A', 'C+V', 'C+Delay']

        csvFile=open(self.csv_path,encoding='utf-8')
        lines=csvFile.readlines()   #lines[0] = patient_ID,dataset,AP,pre,PVP,0-1,0-2
        for i in range(1,len(lines)):
            s=lines[i]
            s=s.replace('\n','')
            patient_inf=s.split(',')
            patient_flag = patient_inf[1]


            if patient_flag == str(self.flag):

                t2pathlst.append(f'{self.data_dir}/{patient_inf[0]}/T2WI.nii.gz')
                dwipathlst.append(f'{self.data_dir}/{patient_inf[0]}/DWI.nii.gz')
                in_pathlst.append(f'{self.data_dir}/{patient_inf[0]}/In Phase.nii.gz')
                out_pathlst.append(f'{self.data_dir}/{patient_inf[0]}/Out Phase.nii.gz')
                prepathlst.append(f'{self.data_dir}/{patient_inf[0]}/C-pre.nii.gz')
                apathlst.append(f'{self.data_dir}/{patient_inf[0]}/C+A.nii.gz')
                pvpathlst.append(f'{self.data_dir}/{patient_inf[0]}/C+Delay.nii.gz')
                dpathlst.append(f'{self.data_dir}/{patient_inf[0]}/C+V.nii.gz')

                self.labellst.append(int(patient_inf[2]))



        data_dict = [{'t2': t2path, 'dwi':dwipath, 'in_':in_path, 'out_':out_path, 'pre': prepath, 'ap':apath, 'pvp':pvpath, 'dp':dpath, 'label':label} for t2path, dwipath, in_path, out_path, prepath, apath, pvpath, dpath ,label in 
                        zip(t2pathlst, dwipathlst, in_pathlst, out_pathlst, prepathlst, apathlst, pvpathlst, dpathlst, self.labellst)]
        
        return data_dict
        


    def __getitem__(self,index):
        

        if self.flag == "Train":
            image = self.train_transform(self.data_dict[index])

        else:
            image = self.val_transform(self.data_dict[index])

        label =self.labellst[index]

        return image['t2'], image['dwi'], image['in_'], image['out_'], image['pre'], image['ap'], image['pvp'], image['dp'], label  #self.label = class_label


    def __len__(self):
        return len(self.labellst)
