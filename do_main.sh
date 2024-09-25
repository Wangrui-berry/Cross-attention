
# fold1
python ./Crossattention/do_main.py --data_dir ./Crossattention/2data_roi_register/images \
--csv_path ./Crossattention/3subfold_lesionclassifier/fold1.csv \
--resume False --weight_path /wangrui/MVI/experiment/Crossattention/results/2data_final/raw_save/fold1.pth \
--result_dir ./Crossattention/results/  \
--json_dir ps32_res10_lr1e-4_depth64_epoch100.json --num_classes=7 --lr=1e-4 \
--net 3dres --batch_size=8 --num_epoch=100 --n_fold 5fold
#-warmup-epochs 5 


# fold2
python ./Crossattention/do_main.py --data_dir ./Crossattention/2data_roi_register/images \
--csv_path ./Crossattention/3subfold_lesionclassifier/fold2.csv \
--resume False --weight_path /wangrui/MVI/experiment/Crossattention/results/2data_final/raw_save/fold2.pth \
--result_dir ./Crossattention/results/  \
--json_dir ps32_res10_lr1e-4_depth64_epoch100.json --num_classes=7 --lr=1e-4 \
--net 3dres --batch_size=8 --num_epoch=100 --n_fold 5fold
#-warmup-epochs 5 


# fold3
python ./Crossattention/do_main.py --data_dir ./Crossattention/2data_roi_register/images \
--csv_path ./Crossattention/3subfold_lesionclassifier/fold3.csv \
--resume False --weight_path /wangrui/MVI/experiment/Crossattention/results/2data_final/raw_save/fold3.pth \
--result_dir ./Crossattention/results/  \
--json_dir ps32_res10_lr1e-4_depth64_epoch100.json --num_classes=7 --lr=1e-4 \
--net 3dres --batch_size=8 --num_epoch=100 --n_fold 5fold
#-warmup-epochs 5 




# fold4
python ./Crossattention/do_main.py --data_dir ./Crossattention/2data_roi_register/images \
--csv_path ./Crossattention/3subfold_lesionclassifier/fold4.csv \
--resume False --weight_path /wangrui/MVI/experiment/Crossattention/results/2data_final/raw_save/fold4.pth \
--result_dir ./Crossattention/results/  \
--json_dir ps32_res10_lr1e-4_depth64_epoch100.json --num_classes=7 --lr=1e-4 \
--net 3dres --batch_size=8 --num_epoch=100 --n_fold 5fold
#-warmup-epochs 5 




# fold5
python ./Crossattention/do_main.py --data_dir ./Crossattention/2data_roi_register/images \
--csv_path ./Crossattention/3subfold_lesionclassifier/fold5.csv \
--resume False --weight_path /wangrui/MVI/experiment/Crossattention/results/2data_final/raw_save/fold5.pth \
--result_dir ./Crossattention/results/  \
--json_dir ps32_res10_lr1e-4_depth64_epoch100.json --num_classes=7 --lr=1e-4 \
--net 3dres --batch_size=8 --num_epoch=100 --n_fold 5fold
#-warmup-epochs 5 