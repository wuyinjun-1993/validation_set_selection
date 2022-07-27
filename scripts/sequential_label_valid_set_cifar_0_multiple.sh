#!/bin/bash
trap "exit" INT



err_label_ratio=0.6

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

save_path_prefix=${save_path_root_dir}/rand_error_${err_label_ratio}

gpu_id_ls=(1 2 3)

port_num_ls=(10001 10002 10005)

total_valid_sample_count=100

export CUDA_VISIBLE_DEVICES=${gpu_id_ls[0]}
echo CUDA_VISIBLE_DEVICES::${CUDA_VISIBLE_DEVICES}

echo "initial cleaning"

cd ../src/main/


#add_valid_in_training_flag="--cluster_method_three --cosin_dist --weight_by_norm --replace --use_model_prov --model_prov_period 20 --total_valid_sample_count ${total_valid_sample_count}"
lr_decay_flag="--use_pretrained_model --lr_decay"




repeat_num=1

for (( i=1 ; i <= repeat_num; i++ ))
do

exe_cmd="python -m torch.distributed.launch \
  --nproc_per_node 1 \
  --master_port ${port_num_ls[0]} \
  main_train.py \
  --nce-t 0.07 \
  --nce-k 200 \
  --data_dir ${data_dir} \
  --dataset ${dataset_name} \
  --valid_count ${valid_ratio_each_run} \
  --meta_lr ${meta_lr} \
  --flip_labels \
  --err_label_ratio ${err_label_ratio} \
  --save_path ${save_path_prefix}_do_train_${i}/ \
  --cuda \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epochs ${epochs} \
  --lr_decay \
  --do_train"


output_file_name=${output_dir}/output_${dataset_name}_rand_error_${err_label_ratio}_do_train_${i}.txt


echo "${exe_cmd} > ${output_file_name}"


#${exe_cmd} > ${output_file_name} 2>&1


echo "random sampling"


add_valid_in_training_flag="--total_valid_sample_count ${total_valid_sample_count}"

exe_cmd="python -m torch.distributed.launch \
  --nproc_per_node 1 \
  --master_port ${port_num} \
  main_train.py \
  --load_dataset \
  --nce-k 200 \
  --data_dir ${data_dir} \
  --dataset ${dataset_name} \
  --valid_count ${valid_ratio_each_run} \
  --meta_lr 5 \
  --flip_labels \
  --err_label_ratio ${err_label_ratio} \
  --save_path ${save_path_prefix}_rand_select_0_${i}/ \
  --prev_save_path ${save_path_prefix}_do_train_${i}/ \
  --cuda \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epochs ${epochs} \
  ${add_valid_in_training_flag} \
  ${lr_decay_flag}"


output_file_name=${output_dir}/output_${dataset_name}_rand_error_${err_label_ratio}_rand_select_0_${i}.txt

echo "${exe_cmd} > ${output_file_name}"

#${exe_cmd} > ${output_file_name} 2>&1 &



echo "clustering method two"

cd ../../scripts/

args="${dataset_name} ${data_dir} ${save_path_root_dir} ${output_dir} ${gpu_id_ls[1]} ${valid_ratio_each_run}  1 ${port_num_ls[1]} ${meta_lr} ${lr} ${batch_size} ${test_batch_size} ${epochs} ${save_path_prefix} ${total_valid_sample_count} ${lr_decay_flag} $i ${err_label_ratio}"

echo "bash run_cluster_method_two.sh ${args} > ${save_path_root_dir}/output_cluster_method_two_${i}.txt"

bash run_cluster_method_two.sh  ${dataset_name} ${data_dir} ${save_path_root_dir} ${output_dir} ${gpu_id_ls[1]} ${valid_ratio_each_run}  1 ${port_num_ls[1]} ${meta_lr} ${lr} ${batch_size} ${test_batch_size} ${epochs} ${save_path_prefix} ${total_valid_sample_count} "${lr_decay_flag}" $i ${err_label_ratio} > ${save_path_root_dir}/output_cluster_method_two_${i}.txt 2>&1 & 


echo "cluster method three"

args="${dataset_name} ${data_dir} ${save_path_root_dir} ${output_dir} ${gpu_id_ls[2]} ${valid_ratio_each_run}  1 ${port_num_ls[2]} ${meta_lr} ${lr} ${batch_size} ${test_batch_size} ${epochs} ${save_path_prefix} ${total_valid_sample_count} ${lr_decay_flag} $i ${err_label_ratio}"


echo "bash run_cluster_method_three.sh ${args} > ${save_path_root_dir}/output_cluster_method_three_${i}.txt 2>&1"


bash run_cluster_method_three.sh ${dataset_name} ${data_dir} ${save_path_root_dir} ${output_dir} ${gpu_id_ls[2]} ${valid_ratio_each_run}  1 ${port_num_ls[2]} ${meta_lr} ${lr} ${batch_size} ${test_batch_size} ${epochs} ${save_path_prefix} ${total_valid_sample_count} "${lr_decay_flag}" $i ${err_label_ratio} > ${save_path_root_dir}/output_cluster_method_three_${i}.txt 2>&1 &


cd ../src/main/

wait

done








