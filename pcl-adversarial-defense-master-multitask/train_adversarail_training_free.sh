python3  adversarial_training_free.py   --n-repeats=8 --eval-freq=1 --t-max=320000  --save_name=-8-0.01- \
	--model_name=Models_Softmax/CIFAR10_Softmax.pth.tar --gpu=0  --lr=0.002


#python3  adversarial_training_free.py  --max-epoch=1000 --n-repeats=9 --eval-freq=1 --t-max=320000  --save_name=-9-0.002- \
#	--model_name=Models_Softmax/CIFAR10_Softmax.pth.tar --gpu=1  --lr=0.002

#python3  adversarial_training_free.py  --max-epoch=1000 --n-repeats=10 --eval-freq=1 --t-max=320000  --save_name=-10-0.005- \
#        --model_name=Models_Softmax/CIFAR10_Softmax.pth.tar --gpu=2  --lr=0.002


#python3  adversarial_training_free.py   --n-repeats=2 --eval-freq=1 --t-max=320000  --save_name=-2-0.00005- \
#        --model_name=Models_Softmax/CIFAR10_Softmax.pth.tar --gpu=3  --lr=0.002 --lr_conprox=0.00005 --weight-conprox=0.0001
