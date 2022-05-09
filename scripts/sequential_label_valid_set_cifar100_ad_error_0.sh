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
add_valid_in_training_set=${14}
lr_decay=${15}

valid_ratio_each_run=$6 #$(( total_valid_ratio / repeat_times ))

save_path_prefix=${save_path_root_dir}/biased_error_${err_label_ratio}_valid_select


total_valid_sample_count=100

export CUDA_VISIBLE_DEVICES=${gpu_ids}
echo CUDA_VISIBLE_DEVICES::${CUDA_VISIBLE_DEVICES}

echo "initial cleaning"

cd ../src/main/


add_valid_in_training_flag="--cluster_method_three --cosin_dist --weight_by_norm --replace --use_model_prov --model_prov_period 20 --total_valid_sample_count ${total_valid_sample_count}"
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
  --valid_ratio ${valid_ratio_each_run} \
  --meta_lr ${meta_lr} \
  --biased_flip \
  --flip_labels \
  --err_label_ratio ${err_label_ratio} \
  --save_path ${save_path_prefix}_do_train/ \
  --cuda \
  --lr 0.02 \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epochs ${epochs} \
  --do_train"


output_file_name=${output_dir}/output_${dataset_name}_biased_error_${err_label_ratio}_do_train_0.txt


echo "${exe_cmd} > ${output_file_name}"


#${exe_cmd} > ${output_file_name} 2>&1


exe_cmd="python -m torch.distributed.launch \
  --nproc_per_node 1 \
  --master_port ${port_num} \
  main_train.py \
  --load_dataset \
  --select_valid_set \
  --biased_flip \
  --nce-k 200 \
  --data_dir ${data_dir} \
  --dataset ${dataset_name} \
  --valid_ratio ${valid_ratio_each_run} \
  --meta_lr 5 \
  --flip_labels \
  --err_label_ratio ${err_label_ratio} \
  --save_path ${save_path_prefix}_seq_select_0/ \
  --prev_save_path ${save_path_prefix}_do_train/ \
  --cuda \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epochs ${epochs} \
  ${add_valid_in_training_flag} \
  ${lr_decay_flag}"


output_file_name=${output_dir}/output_${dataset_name}_biased_error_${err_label_ratio}_valid_select_seq_select_0.txt

echo "${exe_cmd} > ${output_file_name}"

${exe_cmd} > ${output_file_name} 2>&1 

mkdir ${save_path_prefix}_no_reweighting_seq_select_0/

#cp ${save_path_prefix}_seq_select_0/* ${save_path_prefix}_no_reweighting_seq_select_0/


echo "add_valid_in_training_flag: ${add_valid_in_training_flag}"


#for k in {1..${repeat_times}}
for (( k=1; k<=repeat_times; k++ ))
do

	exe_cmd="python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port ${port_num} \
    main_train.py \
    --load_dataset \
    --biased_flip \
    --select_valid_set \
    --continue_label \
    --load_cached_weights \
    --cached_sample_weights_name cached_sample_weights \
    --nce-t 0.07 \
    --nce-k 200 \
    --data_dir ${data_dir} \
    --dataset ${dataset_name} \
    --valid_ratio ${valid_ratio_each_run} \
    --meta_lr ${meta_lr} \
    --not_save_dataset \
    --flip_labels \
    --err_label_ratio ${err_label_ratio} \
    --save_path ${save_path_prefix}_seq_select_$k/ \
    --prev_save_path ${save_path_prefix}_seq_select_$(( k - 1 ))/ \
    --cuda \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --epochs ${epochs} \
    ${add_valid_in_training_flag} \
	${lr_decay_flag}"

	output_file_name=${output_dir}/output_${dataset_name}_biased_error_${err_label_ratio}_valid_select_seq_select_$k.txt

	echo "${exe_cmd} > ${output_file_name}"
	
	${exe_cmd} > ${output_file_name} 2>&1 
	
done



