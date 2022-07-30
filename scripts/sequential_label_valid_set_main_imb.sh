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
echo "model_prov_period::${model_prov_period}"
echo "valid_ratio_each_run::${valid_ratio_each_run}" #$(( total_valid_ratio / repeat_times ))
echo "method::${method}"
echo "total_valid_sample_count::${total_valid_sample_count}"

echo "metric::${metric}"
echo "suffix::${suffix}"
echo "imb_factor::${imb_factor}"
echo "use_pretrained_model::$use_pretrained_model"

#total_valid_sample_count=200


mkdir -p ${data_dir}
mkdir -p ${output_dir}
mkdir -p ${save_path_root_dir}

export CUDA_VISIBLE_DEVICES=${gpu_ids}
echo CUDA_VISIBLE_DEVICES::${CUDA_VISIBLE_DEVICES}

echo "initial cleaning"

cd ../src/main/


#--cluster_method_two --weight_by_norm --not_rescale_features  --cosin_dist --replace --use_model_prov --model_prov_period 20 --total_valid_sample_count ${total_valid_sample_count}


if [[ ${method} == "cluster_method_two" ]];
then
	add_valid_in_training_flag="--select_valid_set --cluster_method_two --weight_by_norm --not_rescale_features  --cosin_dist --replace --use_model_prov --model_prov_period ${model_prov_period} --total_valid_sample_count ${total_valid_sample_count}"

elif [[ $method == "cluster_method_three" ]];
then
	
	add_valid_in_training_flag="--select_valid_set --cluster_method_three --weight_by_norm --not_rescale_features  --cosin_dist --replace --use_model_prov --model_prov_period ${model_prov_period} --total_valid_sample_count ${total_valid_sample_count} --no_sample_weights_k_means"

elif [[ $method == "certain" ]];
then

	add_valid_in_training_flag="--total_valid_sample_count ${total_valid_sample_count} --certain_select"

elif [[ $method == "uncertain" ]];
then
	add_valid_in_training_flag="--total_valid_sample_count ${total_valid_sample_count} --uncertain_select"

elif [[ $method == "rand" ]];
then
        add_valid_in_training_flag="--total_valid_sample_count ${total_valid_sample_count}"

elif [[ $method == "ta" ]];
then
        add_valid_in_training_flag="--total_valid_sample_count ${total_valid_sample_count} --ta_vaal_train"


elif [[ $method == 'craige' ]];
then 
	add_valid_in_training_flag="--total_valid_sample_count ${total_valid_sample_count} --craige"

elif [[ $method == 'finetune'  ]]
then 
	add_valid_in_training_flag="--total_valid_sample_count ${total_valid_sample_count} --finetune"

fi



metric_str=""
if [[ ${metric} == "auc" ]];
then
	metric_str="--metric auc"
elif [[ ${metric} == "kappa" ]];
then
	metric_str="--metric kappa"
fi


#add_valid_in_training_flag="--cluster_method_two --weight_by_norm --not_rescale_features  --cosin_dist --replace --use_model_prov --model_prov_period ${model_prov_period} --total_valid_sample_count ${total_valid_sample_count}"

#add_valid_in_training_flag="--cluster_method_three --not_rescale_features --weight_by_norm  --cosin_dist  --replace --use_model_prov --model_prov_period 20 --total_valid_sample_count ${total_valid_sample_count} --remove_empty_clusters --no_sample_weights_k_means"


bias_flip_str="--bias_classes --imb_factor ${imb_factor}"
err_type="imb_factor_${imb_factor}"


lr_decay_flag="--lr_decay"

if "${use_pretrained_model}";
then 
	lr_decay_flag="--use_pretrained_model --lr_decay"
fi


save_path_prefix=${save_path_root_dir}/${err_type}_${err_label_ratio}_valid_select_${method}${suffix}


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
  ${bias_flip_str} \
  --save_path ${save_path_prefix}_do_train/ \
  --cuda \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epochs ${epochs} \
  --do_train \
  ${metric_str} \
  --lr_decay
"


output_file_name=${output_dir}/output_${dataset_name}_${err_type}_${err_label_ratio}_${method}_do_train_0${suffix}.txt


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
  ${bias_flip_str} \
  --save_path ${save_path_prefix}_seq_select_0/ \
  --prev_save_path ${save_path_prefix}_do_train/\
  --cuda \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epochs ${epochs} \
  --total_valid_sample_count ${total_valid_sample_count} \
  ${metric_str} \
  ${lr_decay_flag}"


output_file_name=${output_dir}/output_${dataset_name}_${err_type}_${err_label_ratio}_valid_select_seq_select_0_${method}${suffix}.txt

echo "${exe_cmd} > ${output_file_name}"

#${exe_cmd} > ${output_file_name} 2>&1 

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
    ${bias_flip_str} \
    --save_path ${save_path_prefix}_seq_select_${k}/ \
    --prev_save_path ${save_path_prefix}_seq_select_$(( k - 1 ))/ \
    --cuda \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --epochs ${epochs} \
    ${metric_str} \
    ${add_valid_in_training_flag} \
	${lr_decay_flag}"

	output_file_name=${output_dir}/output_${dataset_name}_${err_type}_${err_label_ratio}_valid_select_seq_select_${k}_${method}${suffix}.txt

	echo "${exe_cmd} > ${output_file_name}"
	
	${exe_cmd} > ${output_file_name} 2>&1 
	
done



