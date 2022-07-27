#!/bin/bash
trap "exit" INT

source $1

echo "err_label_ratio::$err_label_ratio"

echo "dataset_name::${dataset_name}"
echo "data_dir::${data_dir}"
echo "save_path_root_dir::${save_path_root_dir}"
echo "output_dir::${output_dir}"
echo "gpu_ids::${gpu_ids}"
#total_valid_ratio=$6
echo "repeat_times::${repeat_times}"
echo "port_num::${port_num}"
echo "meta_lr::${meta_lr}"
echo "lr::${lr}"
echo "batch_size::${batch_size}"
echo "test_batch_size::${test_batch_size}"
echo "epochs::${epochs}"
#cached_model_name=${14}
echo "add_valid_in_training_set::${add_valid_in_training_set}"
echo "lr_decay::${lr_decay}"
echo "warm_up_valid_count::${warm_up_valid_count}"

echo "valid_ratio_each_run::${valid_ratio_each_run}" #$(( total_valid_ratio / repeat_times ))

save_path_prefix=${save_path_root_dir}/rand_error_${err_label_ratio}_valid_select


total_valid_sample_count=200

export CUDA_VISIBLE_DEVICES=${gpu_ids}
echo CUDA_VISIBLE_DEVICES::${CUDA_VISIBLE_DEVICES}

echo "initial cleaning"

cd ../src/main/


add_valid_in_training_flag="--cluster_method_three --not_rescale_features --weight_by_norm  --cosin_dist  --replace --use_model_prov --model_prov_period 20 --total_valid_sample_count ${total_valid_sample_count} --remove_empty_clusters --no_sample_weights_k_means"
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
  --flip_labels \
  --err_label_ratio ${err_label_ratio} \
  --save_path ${save_path_prefix}_do_train/ \
  --cuda \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epochs ${epochs} \
  --do_train \
  --lr_decay
"


output_file_name=${output_dir}/output_${dataset_name}_rand_error_${err_label_ratio}_do_train_0.txt


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
  --valid_count ${warm_up_valid_count} \
  --meta_lr ${meta_lr} \
  --flip_labels \
  --err_label_ratio ${err_label_ratio} \
  --save_path ${save_path_prefix}_seq_select_0_method_three_3/ \
  --prev_save_path ${save_path_root_dir}/rand_error_${err_label_ratio}_warm_up/\
  --cuda \
  --continue_label \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epochs ${epochs} \
  ${add_valid_in_training_flag} \
  ${lr_decay_flag}"


output_file_name=${output_dir}/output_${dataset_name}_rand_error_${err_label_ratio}_valid_select_seq_select_0_method_three_3.txt

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
    --select_valid_set \
    --continue_label \
    --load_cached_weights \
    --cached_sample_weights_name cached_sample_weights \
    --nce-t 0.07 \
    --nce-k 200 \
    --data_dir ${data_dir} \
    --dataset ${dataset_name} \
    --valid_count ${valid_ratio_each_run} \
    --meta_lr ${meta_lr} \
    --not_save_dataset \
    --flip_labels \
    --err_label_ratio ${err_label_ratio} \
    --save_path ${save_path_prefix}_seq_select_${k}_method_three_3/ \
    --prev_save_path ${save_path_prefix}_seq_select_$(( k - 1 ))_method_three_3/ \
    --cuda \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --epochs ${epochs} \
    ${add_valid_in_training_flag} \
	${lr_decay_flag}"

	output_file_name=${output_dir}/output_${dataset_name}_rand_error_${err_label_ratio}_valid_select_seq_select_${k}_method_three_3.txt

	echo "${exe_cmd} > ${output_file_name}"
	
	${exe_cmd} > ${output_file_name} 2>&1 
	
done



