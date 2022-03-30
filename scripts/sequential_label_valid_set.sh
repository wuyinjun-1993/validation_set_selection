#!/bin/bash
trap "exit" INT



err_label_ratio=0.9

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
cached_model_name=${14}

valid_ratio_each_run=$6 #$(( total_valid_ratio / repeat_times ))

save_path_prefix=${save_path_root_dir}/rand_error_${err_label_ratio}_valid_select

export CUDA_VISIBLE_DEVICES=${gpu_ids}
#ilp_learning_rate=(0.00002 0.00005 0.0001 0.0002)
echo CUDA_VISIBLE_DEVICES::${CUDA_VISIBLE_DEVICES}

echo "initial cleaning"

cd ../src/main/


exe_cmd="python -m torch.distributed.launch --nproc_per_node 1 --master_port ${port_num} main_train.py --load_dataset --cached_model_name ${cached_model_name} --nce-t 0.07 --nce-k 200 --data_dir ${data_dir} --dataset ${dataset_name} --valid_ratio ${valid_ratio_each_run} --meta_lr ${meta_lr} --not_save_dataset --flip_labels --err_label_ratio ${err_label_ratio} --save_path ${save_path_prefix}_seq_select_0/ --cuda --lr ${lr} --batch_size ${batch_size} --test_batch_size ${test_batch_size} --epochs ${epochs}"

output_file_name=${output_dir}/output_${dataset_name}_rand_error_${err_label_ratio}_valid_select_seq_select_0.txt

echo "${exe_cmd} > ${output_file_name}"

${exe_cmd} > ${output_file_name} 2>&1 

mkdir ${save_path_prefix}_no_reweighting_seq_select_0/

cp ${save_path_prefix}_seq_select_0/* ${save_path_prefix}_no_reweighting_seq_select_0/



#for k in {1..${repeat_times}}
for (( k=1; k<=repeat_times; k++ ))
do

	exe_cmd="python -m torch.distributed.launch --nproc_per_node 1 --master_port ${port_num} main_train.py --select_valid_set --continue_label --load_cached_weights --cached_sample_weights_name cached_sample_weights --cached_model_name ${cached_model_name} --nce-t 0.07 --nce-k 200 --data_dir ${data_dir} --dataset ${dataset_name} --valid_ratio ${valid_ratio_each_run} --meta_lr ${meta_lr} --not_save_dataset --flip_labels --err_label_ratio ${err_label_ratio} --save_path ${save_path_prefix}_seq_select_$k/ --prev_save_path ${save_path_prefix}_seq_select_$(( k - 1 ))/ --cuda --lr ${lr} --batch_size ${batch_size} --test_batch_size ${test_batch_size} --epochs ${epochs}"

	output_file_name=${output_dir}/output_${dataset_name}_rand_error_${err_label_ratio}_valid_select_seq_select_$k.txt

	echo "${exe_cmd} > ${output_file_name}"
	
	${exe_cmd} > ${output_file_name} 2>&1 
	
	exe_cmd="python -m torch.distributed.launch --nproc_per_node 1 --master_port $(( port_num + 1 )) main_train.py --select_valid_set --continue_label --cached_model_name ${cached_model_name} --nce-t 0.07 --nce-k 200 --data_dir ${data_dir} --dataset ${dataset_name} --valid_ratio ${valid_ratio_each_run} --meta_lr ${meta_lr} --not_save_dataset --flip_labels --err_label_ratio ${err_label_ratio} --save_path ${save_path_prefix}_no_reweighting_seq_select_${k}/ --prev_save_path ${save_path_prefix}_no_reweighting_seq_select_$(( k - 1 ))/ --cuda --lr ${lr} --batch_size ${batch_size} --test_batch_size ${test_batch_size} --epochs ${epochs}"

	output_file_name=${output_dir}/output_${dataset_name}_rand_error_${err_label_ratio}_valid_select_no_reweighting_seq_select_$k.txt

#	echo "${exe_cmd} > ${output_file_name}"

#	${exe_cmd} > ${output_file_name} 2>&1
done



