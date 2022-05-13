data_dir=$1
res_dir=$2

echo Running imbalanced 200 experiments
CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh cifar10 imbalanced uncertainty 1 0.005 0.1 40 100 10031 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=1 ./sequential_label.sh cifar10 imbalanced random 1 0.005 0.1 40 100 10032 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=2 ./sequential_label.sh cifar10 imbalanced ours 1 0.005 0.1 40 100 10033 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=3 ./sequential_label.sh cifar10 imbalanced certainty 1 0.005 0.1 40 100 10034 ${data_dir} ${res_dir} &
wait

echo Running imbalanced 100 experiments
CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh cifar10 imbalanced uncertainty 1 0.01 0.1 40 100 10031 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=1 ./sequential_label.sh cifar10 imbalanced random 1 0.01 0.1 40 100 10032 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=2 ./sequential_label.sh cifar10 imbalanced ours 1 0.01 0.1 40 100 10033 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=3 ./sequential_label.sh cifar10 imbalanced certainty 1 0.01 0.1 40 100 10034 ${data_dir} ${res_dir} &
wait

echo Running imbalanced 50 experiments
CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh cifar10 imbalanced uncertainty 1 0.02 0.1 40 100 10031 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=1 ./sequential_label.sh cifar10 imbalanced random 1 0.02 0.1 40 100 10032 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=2 ./sequential_label.sh cifar10 imbalanced ours 1 0.02 0.1 40 100 10033 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=3 ./sequential_label.sh cifar10 imbalanced certainty 1 0.02 0.1 40 100 10034 ${data_dir} ${res_dir} &
wait

echo Running imbalanced 20 experiments
CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh cifar10 imbalanced uncertainty 1 0.05 0.1 40 100 10031 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=1 ./sequential_label.sh cifar10 imbalanced random 1 0.05 0.1 40 100 10032 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=2 ./sequential_label.sh cifar10 imbalanced ours 1 0.05 0.1 40 100 10033 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=3 ./sequential_label.sh cifar10 imbalanced certainty 1 0.05 0.1 40 100 10034 ${data_dir} ${res_dir} &
wait

echo Running imbalanced 10 experiments
CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh cifar10 imbalanced uncertainty 1 0.1 0.1 40 100 10031 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=1 ./sequential_label.sh cifar10 imbalanced random 1 0.1 0.1 40 100 10032 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=2 ./sequential_label.sh cifar10 imbalanced ours 1 0.1 0.1 40 100 10033 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=3 ./sequential_label.sh cifar10 imbalanced certainty 1 0.1 0.1 40 100 10034 ${data_dir} ${res_dir} &
wait

echo Running imbalanced 1 experiments
CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh cifar10 imbalanced uncertainty 1 1 0.1 40 100 10031 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=1 ./sequential_label.sh cifar10 imbalanced random 1 1 0.1 40 100 10032 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=2 ./sequential_label.sh cifar10 imbalanced ours 1 1 0.1 40 100 10033 ${data_dir} ${res_dir} &
CUDA_VISIBLE_DEVICES=3 ./sequential_label.sh cifar10 imbalanced certainty 1 1 0.1 40 100 10034 ${data_dir} ${res_dir} &
wait

echo Finished!
