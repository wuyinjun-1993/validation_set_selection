data_dir=$1
res_dir=$2

# echo Running BaseModel training
# CUDA_VISIBLE_DEVICES=0 ./train_basemodel.sh cifar10 imbalanced 0.005 0.1 10031 ${data_dir} ${res_dir} &
# CUDA_VISIBLE_DEVICES=1 ./train_basemodel.sh cifar10 imbalanced 0.01 0.1 10032 ${data_dir} ${res_dir} &
# CUDA_VISIBLE_DEVICES=2 ./train_basemodel.sh cifar10 imbalanced 0.02 0.1 10033 ${data_dir} ${res_dir} &
# CUDA_VISIBLE_DEVICES=3 ./train_basemodel.sh cifar10 imbalanced 0.05 0.1 10034 ${data_dir} ${res_dir} &
# wait
echo Running warmup
CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh cifar10 imbalanced random 1 0.01 0.1 20 10 10030 ${data_dir} ${res_dir} True &
CUDA_VISIBLE_DEVICES=2 ./sequential_label.sh cifar10 imbalanced random 1 0.005 0.1 20 10 10032 ${data_dir} ${res_dir} True &
wait

echo Running all of our methods
CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh cifar10 imbalanced ours1 3 0.01 0.1 20 20 10030 ${data_dir} ${res_dir} False &
CUDA_VISIBLE_DEVICES=1 ./sequential_label.sh cifar10 imbalanced ours2 3 0.01 0.1 20 20 10031 ${data_dir} ${res_dir} False &
CUDA_VISIBLE_DEVICES=2 ./sequential_label.sh cifar10 imbalanced ours1 3 0.005 0.1 20 20 10032 ${data_dir} ${res_dir} False &
CUDA_VISIBLE_DEVICES=3 ./sequential_label.sh cifar10 imbalanced ours2 3 0.005 0.1 20 20 10033 ${data_dir} ${res_dir} False &
wait

echo Running imbalanced 200 experiments
CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh cifar10 imbalanced uncertainty 1 0.005 0.1 20 20 10030 ${data_dir} ${res_dir} False &
CUDA_VISIBLE_DEVICES=1 ./sequential_label.sh cifar10 imbalanced random 1 0.005 0.1 20 20 10031 ${data_dir} ${res_dir} False &
CUDA_VISIBLE_DEVICES=2 ./sequential_label.sh cifar10 imbalanced finetune 1 0.005 0.1 20 20 10032 ${data_dir} ${res_dir} False &
CUDA_VISIBLE_DEVICES=3 ./sequential_label.sh cifar10 imbalanced certainty 1 0.005 0.1 20 20 10033 ${data_dir} ${res_dir} False &
wait

echo Running imbalanced 100 experiments
CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh cifar10 imbalanced uncertainty 1 0.01 0.1 20 20 10031 ${data_dir} ${res_dir} False &
CUDA_VISIBLE_DEVICES=1 ./sequential_label.sh cifar10 imbalanced random 1 0.01 0.1 20 20 10032 ${data_dir} ${res_dir} False &
CUDA_VISIBLE_DEVICES=2 ./sequential_label.sh cifar10 imbalanced finetune 1 0.01 0.1 20 20 10033 ${data_dir} ${res_dir} False &
CUDA_VISIBLE_DEVICES=3 ./sequential_label.sh cifar10 imbalanced certainty 1 0.01 0.1 20 20 10034 ${data_dir} ${res_dir} False &
wait

# 
# echo Running imbalanced 50 experiments
# CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh cifar10 imbalanced uncertainty 1 0.02 0.1 20 100 10031 ${data_dir} ${res_dir} &
# CUDA_VISIBLE_DEVICES=1 ./sequential_label.sh cifar10 imbalanced random 1 0.02 0.1 20 100 10032 ${data_dir} ${res_dir} &
# CUDA_VISIBLE_DEVICES=2 ./sequential_label.sh cifar10 imbalanced finetune 1 0.02 0.1 20 100 10033 ${data_dir} ${res_dir} &
# CUDA_VISIBLE_DEVICES=3 ./sequential_label.sh cifar10 imbalanced certainty 1 0.02 0.1 20 100 10034 ${data_dir} ${res_dir} &
# wait
# 
# echo Running imbalanced 20 experiments
# CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh cifar10 imbalanced uncertainty 1 0.05 0.1 20 100 10031 ${data_dir} ${res_dir} &
# CUDA_VISIBLE_DEVICES=1 ./sequential_label.sh cifar10 imbalanced random 1 0.05 0.1 20 100 10032 ${data_dir} ${res_dir} &
# CUDA_VISIBLE_DEVICES=2 ./sequential_label.sh cifar10 imbalanced finetune 1 0.05 0.1 20 100 10033 ${data_dir} ${res_dir} &
# CUDA_VISIBLE_DEVICES=3 ./sequential_label.sh cifar10 imbalanced certainty 1 0.05 0.1 20 100 10034 ${data_dir} ${res_dir} &
# wait
# 
# echo Running imbalanced 10 experiments
# CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh cifar10 imbalanced uncertainty 1 0.1 0.1 20 100 10031 ${data_dir} ${res_dir} &
# CUDA_VISIBLE_DEVICES=1 ./sequential_label.sh cifar10 imbalanced random 1 0.1 0.1 20 100 10032 ${data_dir} ${res_dir} &
# CUDA_VISIBLE_DEVICES=2 ./sequential_label.sh cifar10 imbalanced finetune 1 0.1 0.1 20 100 10033 ${data_dir} ${res_dir} &
# CUDA_VISIBLE_DEVICES=3 ./sequential_label.sh cifar10 imbalanced certainty 1 0.1 0.1 20 100 10034 ${data_dir} ${res_dir} &
# wait
# 
# CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh cifar10 imbalanced glc 1 0.005 0.1 20 100 10031 ${data_dir} ${res_dir} &
# CUDA_VISIBLE_DEVICES=1 ./sequential_label.sh cifar10 imbalanced glc 1 0.01 0.1 20 100 10032 ${data_dir} ${res_dir} &
# CUDA_VISIBLE_DEVICES=2 ./sequential_label.sh cifar10 imbalanced glc 1 0.02 0.1 20 100 10033 ${data_dir} ${res_dir} &
# CUDA_VISIBLE_DEVICES=3 ./sequential_label.sh cifar10 imbalanced glc 1 0.05 0.1 20 100 10034 ${data_dir} ${res_dir} &
# wait

echo Finished!
