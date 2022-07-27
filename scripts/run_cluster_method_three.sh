#!/bin/bash
trap "exit" INT

echo "clustering method three"


err_label_ratio=${18}
echo "err_label_ratio::${err_label_ratio}"

dataset_name=$1

echo "dataset_name::${dataset_name}"

data_dir=$2

echo "data_dir::${data_dir}"

save_path_root_dir=$3

echo "save_path_root_dir:${save_path_root_dir}"

output_dir=$4

echo "output_dir::${output_dir}"

gpu_ids=$5

echo "gpu_ids::${gpu_ids}"

#total_valid_ratio=$6
repeat_times=$7

echo "repeat_times::${repeat_times}"

port_num=$8

echo "port_num:${port_num}"

meta_lr=$9

echo "meta_lr::${meta_lr}"

lr=${10}

echo "lr::${lr}"

batch_size=${11}

echo "batch_size::${batch_size}"

test_batch_size=${12}

echo "test_batch_size::${test_batch_size}"

epochs=${13}

echo "epochs::${epochs}"



valid_ratio_each_run=$6 #$(( total_valid_ratio / repeat_times ))

echo "valid_ratio_each_run::${valid_ratio_each_run}"

save_path_prefix=${14}

echo "save_path_prefix::${save_path_prefix}"

total_valid_sample_count=${15}

echo "total_valid_sample_count::${total_valid_sample_count}"

lr_decay_flag="${16}"

echo "lr_decay_flag::${lr_decay_flag}"

i=${17}

echo "i::${i}"




export CUDA_VISIBLE_DEVICES=${gpu_ids}
echo CUDA_VISIBLE_DEVICES::${CUDA_VISIBLE_DEVICES}

repeat_nums=20

add_valid_in_training_flag="--cluster_method_three --cosin_dist --replace --use_model_prov --model_prov_period 20 --total_valid_sample_count ${total_valid_sample_count}"


prev_save_path="${save_path_prefix}_do_train_${i}/"

for (( k=1 ;k<repeat_nums ;k++ ))
do


save_path="${save_path_prefix}_valid_select_cluster_method_three_0_${i}_${k}/"

exe_cmd="python -m torch.distributed.launch \
  --nproc_per_node 1 \
  --master_port ${port_num} \
  main_train.py \
  --load_dataset \
  --select_valid_set \
  --nce-k 200 \
  --data_dir ${data_dir} \
  --dataset ${dataset_name} \
  --valid_count ${valid_ratio_each_run} \
  --meta_lr 5 \
  --flip_labels \
  --err_label_ratio ${err_label_ratio} \
  --save_path ${save_path} \
  --prev_save_path ${prev_save_path} \
  --cuda \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epochs ${epochs} \
  ${add_valid_in_training_flag} \
  ${lr_decay_flag}"


output_file_name=${output_dir}/output_${dataset_name}_rand_error_${err_label_ratio}_valid_select_cluster_method_three_0_${i}.txt

echo "${exe_cmd} > ${output_file_name}"

#${exe_cmd} > ${output_file_name} 2>&1 &


prev_save_path=${save_path}

done
