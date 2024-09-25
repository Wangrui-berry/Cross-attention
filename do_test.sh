#!/bin/bash
# conda activate ruirui &&

#data_dir: 1data_roi // 2data_roi_register


#num_epoch

# fold1
python //wangrui/MVI/experiment/LLDMRI_1011/main/test.py  --data_dir //wangrui/MVI/experiment/LLDMRI_1011/2data_roi_register/images \
--resume True --weight_path /wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/raw_save/3CA+sim//fold1.pth \
--csv_path //wangrui/MVI/experiment/LLDMRI_1011/3subfold_lesionclassifier/test.csv \
--result_dir //wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/new_test/3CA+sim/ \
--json_dir results.json --num_classes=7 --lr=1e-4 \
--csv_dir //wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/result.csv \
--net 3dres --batch_size=8 --num_epoch=1 --n_fold fold-1
#-warmup-epochs 5 


# fold2
python //wangrui/MVI/experiment/LLDMRI_1011/main/test.py --data_dir //wangrui/MVI/experiment/LLDMRI_1011/2data_roi_register/images \
--resume True --weight_path /wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/raw_save/3CA+sim//fold2.pth \
--csv_path //wangrui/MVI/experiment/LLDMRI_1011/3subfold_lesionclassifier/test.csv \
--result_dir //wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/new_test/3CA+sim/ \
--json_dir results.json --num_classes=7 --lr=1e-4 \
--csv_dir //wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/result.csv \
--net 3dres --batch_size=8 --num_epoch=1 --n_fold fold-2
#-warmup-epochs 5 


# fold3
python //wangrui/MVI/experiment/LLDMRI_1011/main/test.py --data_dir //wangrui/MVI/experiment/LLDMRI_1011/2data_roi_register/images \
--resume True --weight_path /wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/raw_save/3CA+sim//fold3.pth \
--csv_path //wangrui/MVI/experiment/LLDMRI_1011/3subfold_lesionclassifier/test.csv \
--result_dir //wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/new_test/3CA+sim/ \
--json_dir results.json --num_classes=7 --lr=1e-4 \
--csv_dir //wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/result.csv \
--net 3dres --batch_size=8 --num_epoch=1 --n_fold fold-3
#-warmup-epochs 5 




# fold4-4
python //wangrui/MVI/experiment/LLDMRI_1011/main/test.py --data_dir //wangrui/MVI/experiment/LLDMRI_1011/2data_roi_register/images \
--resume True --weight_path /wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/raw_save/3CA+sim//fold4.pth \
--csv_path //wangrui/MVI/experiment/LLDMRI_1011/3subfold_lesionclassifier/test.csv \
--result_dir //wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/new_test/3CA+sim/ \
--json_dir results.json --num_classes=7 --lr=1e-4 \
--csv_dir //wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/result.csv \
--net 3dres --batch_size=8 --num_epoch=1 --n_fold fold-4
#-warmup-epochs 5 




# fold5
python //wangrui/MVI/experiment/LLDMRI_1011/main/test.py --data_dir //wangrui/MVI/experiment/LLDMRI_1011/2data_roi_register/images \
--resume True --weight_path /wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/raw_save/3CA+sim//fold5.pth  \
--csv_path //wangrui/MVI/experiment/LLDMRI_1011/3subfold_lesionclassifier/test.csv \
--result_dir //wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/new_test/3CA+sim/ \
--json_dir results.json --num_classes=7 --lr=1e-4 \
--csv_dir //wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/result.csv \
--net 3dres --batch_size=8 --num_epoch=1 --n_fold fold-5
#-warmup-epochs 5 


# python //wangrui/MVI/experiment/LLDMRI_1011/main/test.py --data_dir //wangrui/MVI/experiment/LLDMRI_1011/2data_roi_register/images \
# --resume True --weight_path /wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/raw_save/3CA+sim//2023-11-02_04_06_49.pth \
# --csv_path //wangrui/MVI/experiment/LLDMRI_1011/3subfold_lesionclassifier/fold5.csv \
# --result_dir //wangrui/MVI/experiment/LLDMRI_1011/main/results/2data_final/new_test/3CA+sim/ \
# --json_dir results.json --num_classes=7 --lr=1e-4 \
# --net 3dres --batch_size=8 --num_epoch=1 --n_fold fold-5