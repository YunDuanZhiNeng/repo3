export CUDA_VISIBLE_DEVICES=0
for outputs_name in outputs  outputs256_18 outputs256_17 outputs256_16 outputs256_15;do # outputs256_17 ; do #outputs256_16 outputs256_15; do # outputs256_1 outputs128; do
        for model in Models_Softmax/frize_Only_PCL_1024_avg_pool.pth.tar; do
	      #for model in Models_Softmax/frize_PLC_avg_pool.pth.tar; do
                for attack in fgsm pgd mim bim; do #fgsm;do
                        for epsilon in 0.03 ; do # 0.02 0.03;do
				for scale in 1; do
                                	    #param="--epsilon=$epsilon --attack=$attack  --file-name=Models_PCL/CIFAR10_PCL.pth.tar"
                                	    param="--epsilon=$epsilon --attack=$attack  --scale=$scale --file-name=$model  --outputs-name=$outputs_name"
                                	    python3 test_frize_robust.py  $param
				done
                        done
                done
        done
done

