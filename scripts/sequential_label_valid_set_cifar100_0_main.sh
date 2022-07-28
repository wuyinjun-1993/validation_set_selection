#!/bin/bash
trap "exit" INT



err_label_ratio=${14}

dataset_name=$1
data_dir=$2
save_path_root_dir=$3
output_dir=$4
gpu_ids=$5
#total_valid_ratio=$6
repeat_times=$7
port_num=$8
meta_lr=$9
lr=${10}
batch_size=${11}
test_batch_size=${12}
epochs=${13}
#cached_model_name=${14}

valid_ratio_each_run=$6 #$(( total_valid_ratio / repeat_times ))

save_path_prefix=${save_path_root_dir}/biased_error_${err_label_ratio}_valid_select


total_valid_sample_count=200

export CUDA_VISIBLE_DEVICES=${gpu_ids}
echo CUDA_VISIBLE_DEVICES::${CUDA_VISIBLE_DEVICES}

echo "initial cleaning"

cd ../src/main/


add_valid_in_training_flag="--cluster_method_two --cluster_method_two_plus --not_rescale_features --weight_by_norm  --cosin_dist  --replace --use_model_prov --model_prov_period 20 --total_valid_sample_count ${total_valid_sample_count} --cluster_method_two_sampling --remove_empty_clusters --no_sample_weights_k_means"

lr_decay_flag="--use_pretrained_model --lr_decay"

<<cmd
if (( add_valid_in_training_set == true ))
then
        add_valid_in_training_flag="--cluster_method_two"
fi

lr_decay_flag=""
if (( lr_decay == true  ))
then
	lr_decay_flag="--lr_decay"
fi

echo "add_valid_in_training_flag: ${add_valid_in_training_flag}"
cmd



exe_cmd="python -m torch.distributed.launch \
  --nproc_per_node 1 \
  --master_port ${port_num} \
  main_train.py \
  --nce-t 0.07 \
  --nce-k 200 \
  --data_dir ${data_dir} \
  --dataset ${dataset_name} \
  --valid_count ${valid_ratio_each_run} \
  --meta_lr ${meta_lr} \
  --biased_flip \
  --flip_labels \
  --err_label_ratio ${err_label_ratio} \
  --save_path ${save_path_prefix}_do_train/ \
  --cuda \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epochs 200 \
  --lr_decay \
  --do_train"


output_file_name=${output_dir}/output_${dataset_name}_biased_error_${err_label_ratio}_do_train_0.txt


echo "${exe_cmd} > ${output_file_name}"


#${exe_cmd} > ${output_file_name} 2>&1


exe_cmd="python -m torch.distributed.launch \
  --nproc_per_node 1 \
  --master_port ${port_num} \
  main_train.py \
  --load_dataset \
  --nce-k 200 \
  --data_dir ${data_dir} \
  --dataset ${dataset_name} \
  --valid_count 100 \
  --meta_lr ${meta_lr} \
  --flip_labels \
  --err_label_ratio ${err_label_ratio} \
  --prev_save_path ${save_path_prefix}_do_train/ \
  --save_path ${save_path_root_dir}/rand_error_${err_label_ratio}_warmup/ \
  --cuda \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epochs ${epochs} \
  ${add_valid_in_training_flag} \
  ${lr_decay_flag}"


output_file_name=${output_dir}/output_${dataset_name}_biased_error_${err_label_ratio}_valid_select_seq_select_0_warm_up.txt

echo "${exe_cmd} > ${output_file_name}"

${exe_cmd} > ${output_file_name} 2>&1

cd ../../scripts/

echo $pwd

bash_out_file_name=${save_path_root_dir}/output_rand.txt

port_num=10001

echo "bash sequential_label_valid_set_cifar100_0_rand.sh $1 $2 $3 $4 0 $6 $7 ${port_num} $9 ${10} ${11} ${12} ${13} ${14} > ${bash_out_file_name} 2>&1 &"

bash sequential_label_valid_set_cifar100_0_rand.sh $1 $2 $3 $4 0 $6 $7 ${port_num} $9 ${10} ${11} ${12} ${13} ${14} > ${bash_out_file_name} 2>&1 &


bash_out_file_name=${save_path_root_dir}/output_certain.txt

port_num=10002

echo "bash sequential_label_valid_set_cifar100_0_certain.sh $1 $2 $3 $4 1 $6 $7 ${port_num} $9 ${10} ${11} ${12} ${13} ${14} > ${bash_out_file_name} 2>&1 &"

bash sequential_label_valid_set_cifar100_0_certain.sh $1 $2 $3 $4 1 $6 $7 ${port_num} $9 ${10} ${11} ${12} ${13} ${14} > ${bash_out_file_name} 2>&1 &


bash_out_file_name=${save_path_root_dir}/output_uncertain.txt

port_num=10003

echo "bash sequential_label_valid_set_cifar100_0_uncertain.sh $1 $2 $3 $4 2 $6 $7 ${port_num} $9 ${10} ${11} ${12} ${13} ${14} > ${bash_out_file_name} 2>&1 &"

bash sequential_label_valid_set_cifar100_0_uncertain.sh $1 $2 $3 $4 2 $6 $7 ${port_num} $9 ${10} ${11} ${12} ${13} ${14} > ${bash_out_file_name} 2>&1 &

wait

bash_out_file_name=${save_path_root_dir}/output_method_two.txt

port_num=10004

echo "bash sequential_label_valid_set_cifar100_0.sh $1 $2 $3 $4 0 $6 $7 ${port_num} $9 ${10} ${11} ${12} ${13} ${14} > ${bash_out_file_name} 2>&1 &"

bash sequential_label_valid_set_cifar100_0.sh $1 $2 $3 $4 0 $6 $7 ${port_num} $9 ${10} ${11} ${12} ${13} ${14} > ${bash_out_file_name} 2>&1 &

bash_out_file_name=${save_path_root_dir}/output_method_three.txt

port_num=10005

echo "bash sequential_label_valid_set_cifar100_0_method_three.sh $1 $2 $3 $4 1 $6 $7 ${port_num} $9 ${10} ${11} ${12} ${13} ${14} > ${bash_out_file_name} 2>&1 &"

bash sequential_label_valid_set_cifar100_0_method_three.sh $1 $2 $3 $4 1 $6 $7 ${port_num} $9 ${10} ${11} ${12} ${13} ${14} > ${bash_out_file_name} 2>&1 &

bash_out_file_name=${save_path_root_dir}/output_method_two_2.txt

port_num=10006

echo "bash sequential_label_valid_set_cifar100_0_2.sh $1 $2 $3 $4 2 $6 $7 ${port_num} $9 ${10} ${11} ${12} ${13} ${14} > ${bash_out_file_name} 2>&1 &"

bash sequential_label_valid_set_cifar100_0_2.sh $1 $2 $3 $4 2 $6 $7 ${port_num} $9 ${10} ${11} ${12} ${13} ${14} > ${bash_out_file_name} 2>&1 &

bash_out_file_name=${save_path_root_dir}/output_method_three_2.txt

port_num=10007

echo "bash sequential_label_valid_set_cifar100_0_method_three_2.sh $1 $2 $3 $4 3 $6 $7 ${port_num} $9 ${10} ${11} ${12} ${13} ${14} > ${bash_out_file_name} 2>&1 &"

bash sequential_label_valid_set_cifar100_0_method_three_2.sh $1 $2 $3 $4 3 $6 $7 ${port_num} $9 ${10} ${11} ${12} ${13} ${14} > ${bash_out_file_name} 2>&1 &

wait


