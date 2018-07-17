



jobname=s4
num_gpus=4
 
mkdir -p logs/sphere36_bd_bn_sphere_2
now=$(date +"%Y%m%d_%H%M%S")
srun -p TITANXP --job-name=$jobname --gres=gpu:$num_gpus python main.py |tee logs/sphere36_bd_bn_sphere_2/train-$now.log &
