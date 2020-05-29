#export CUDA_VISIBLE_DEVICES=0
#python3  pcl_adversarial_training_free.py   --n-repeats=4 --eval-freq=1  \
#	--model_name=Models_Softmax/CIFAR10_Softmax.pth.tar --gpu=0  --lr_model=0.002  \
#	--lr_conprox=0.00000 --weight-conprox=0.00000  --weight-prox=1


#python3  pcl_adversarial_training_free.py   --n-repeats=4 --eval-freq=1  \
#        --model_name=Models_Softmax/CIFAR10_Softmax.pth.tar --gpu=1  --lr_model=0.002  \
#        --lr_conprox=0.00005 --weight-conprox=0.00005  --weight-prox=1


#python3  pcl_adversarial_training_free.py   --n-repeats=8 --eval-freq=1  \
#        --model_name=Models_Softmax/CIFAR10_Softmax.pth.tar --gpu=2  --lr_model=0.002  \
#        --lr_conprox=0.00000 --weight-conprox=0.00000  --weight-prox=1



python3  pcl_adversarial_training_free_multitask_multiadv_free.py --train-batch=128  --n-repeats=2 --eval-freq=1  \
        --model_name=Models_Softmax/CIFAR10_Softmax.pth.tar --gpu=0  --lr_model=0.002  \
        --lr_conprox=0.0 --weight-conprox=0.0  --weight-prox=1 --save_name=256


