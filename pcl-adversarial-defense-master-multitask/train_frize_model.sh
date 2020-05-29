# train frize model after softermax training
#python3  softmax_training_frize.py  --max-epoch=100 --t-max=100  --gpu=0  --lr=0.002 --eval-freq=10 --save-filename=Only_Softmax_1024_avg_pool --filename=Models_Softmax/CIFAR10_Softmax.pth.tar

python3  softmax_training_frize.py  --max-epoch=100 --t-max=100  --gpu=2  --lr=0.002 --eval-freq=10 --save-filename=Only_PCL_1024_avg_pool --filename=Models_PCL/CIFAR10_PCL.pth.tar

# train frize model after pcl training
#python3  softmax_training_frize.py  --max-epoch=100 --t-max=100  --gpu=0  --lr=0.002 --save-filename=PLC_avg_pool  --filename=Models_PCL/CIFAR10_PCL_352_88.29999542236328.pth.tar

# train frize model after only fgsm training
#python3  softmax_training_frize.py  --max-epoch=100 --t-max=100  --gpu=0  --lr=0.002 --save-filename=FGSM  --filename=Models_PCL/CIFAR10_FGSM.pth.tar

# train frize model after only pgd training
#python3  softmax_training_frize.py  --max-epoch=20 --t-max=20  --gpu=2  --lr=0.002 --eval-freq=10  --save-filename=pcl_adversarial_free_1024_sgd_avg_pool_20  --filename=PCL_Models_Adversarial_training_free/CIFAR10_PCL_4_0.0_202_88.86000061035156.pth.tar 

# train frize model after pcl and fgsm training
#python3  softmax_training_frize.py  --max-epoch=100 --t-max=100  --gpu=1  --lr=0.002 --save-filename=PLC_FGSM_351_2  --filename=Models_PCL_AdvTrain_FGSM/CIFAR10_PCL_AdvTrain_FGSM_351_93.18999481201172.pth.tar

# train frize model after pcl and pgd training
#python3  softmax_training_frize.py  --max-epoch=100 --t-max=100  --gpu=0  --lr=0.002 --save-filename=PLC_FGSM_avg_pool  --filename=Models_PCL_AdvTrain_PGD/CIFAR10_PCL_PGD_0.002_0.5_64_399_93.22999572753906.pth.tar

# train frize model after pcl and pgd training
#python3  softmax_training_frize.py  --max-epoch=100 --t-max=100  --gpu=0 --eval-freq=10 --lr=0.002 --save-filename=robust_model_original_1024_adam_full  --filename=robust_model.pth.tar

#python3  softmax_training_frize.py  --max-epoch=150 --t-max=150  --gpu=1  --lr=0.002 --save-filename=adversarial_training_free_avg_out_cutout_google_11  --filename=Models_Adversarial_training_free/CIFAR10_Softmax_-11-0.005-_75_89.31999969482422.pth.tar

#python3  softmax_training_frize.py  --max-epoch=150 --t-max=150  --gpu=1  --lr=0.002 --save-filename=adversarial_training_free_11  --filename=Models_Adversarial_training_free/CIFAR10_Softmax_-11-0.005-_75_89.31999969482422.pth.tar


#python3  softmax_training_frize.py  --max-epoch=150 --t-max=150  --gpu=1  --lr=0.002 --save-filename=adversarial_training_free_avg_out_1024_11  --filename=Models_Adversarial_training_free/CIFAR10_Softmax_-11-0.005-_75_89.31999969482422.pth.tar


