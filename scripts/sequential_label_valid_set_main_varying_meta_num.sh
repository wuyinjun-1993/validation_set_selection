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
echo "bias_flip::${bias_flip}"
echo "method::${method}"
echo "total_valid_sample_count::${total_valid_sample_count}"

echo "metric::${metric}"
echo "suffix::${suffix}"

echo "use_pretrained_model::$use_pretrained_model"

echo "label_aware::${label_aware}"

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
	add_valid_in_training_flag="--select_valid_set --cluster_method_two --weight_by_norm  --cosin_dist  --model_prov_period ${model_prov_period}"

elif [[ $method == "cluster_method_three" ]];
then
	
	add_valid_in_training_flag="--select_valid_set --cluster_method_three --weight_by_norm  --cosin_dist --model_prov_period ${model_prov_period}"


elif [[ $method == "cluster_method_one" ]];
then
        add_valid_in_training_flag="--select_valid_set --cluster_method_two  --cosin_dist --model_prov_period ${model_prov_period}"


elif [[ $method == "certain" ]];
then

	add_valid_in_training_flag="--certain_select"

elif [[ $method == "uncertain" ]];
then
	add_valid_in_training_flag="--uncertain_select"

elif [[ $method == "rand" ]];
then
        add_valid_in_training_flag=""

elif [[ $method == "ta" ]];
then
        add_valid_in_training_flag="--ta_vaal_train"


elif [[ $method == 'craige' ]];
then 
	add_valid_in_training_flag="--craige"

elif [[ $method == 'finetune'  ]]
then 
	add_valid_in_training_flag="--finetune"

fi



metric_str=""
if [[ ${metric} == "auc" ]];
then
	metric_str="--metric auc"
elif [[ ${metric} == "kappa" ]];
then
	metric_str="--metric kappa"
fi


label_aware_flag=""

if ${label_aware};
then
	label_aware_flag="--label_aware"
fi


#add_valid_in_training_flag="--cluster_method_two --weight_by_norm --not_rescale_features  --cosin_dist --replace --use_model_prov --model_prov_period ${model_prov_period} --total_valid_sample_count ${total_valid_sample_count}"

#add_valid_in_training_flag="--cluster_method_three --not_rescale_features --weight_by_norm  --cosin_dist  --replace --use_model_prov --model_prov_period 20 --total_valid_sample_count ${total_valid_sample_count} --remove_empty_clusters --no_sample_weights_k_means"


bias_flip_str=''
err_type='rand_error'

if "${bias_flip}";
then
	bias_flip_str='--biased_flip'
	err_type='bias_error'
fi


flip_label_flag="--flip_labels"

if [[ ${real_noise} = true ]];
then
	echo "use real noise data"

	if [ -f "${data_dir}/CIFAR-N.zip" ];
	then
		echo "file exists!!!"
	else
		echo "download real noise data"
		wget "http://www.yliuu.com/web-cifarN/files/CIFAR-N.zip" -P ${data_dir}
		unzip ${data_dir}/CIFAR-N.zip -d ${data_dir}
	fi
	flip_label_flag="--real_noise"

	bias_flip_str=''
        err_type='real_error'
	err_label_ratio='0'
fi



lr_decay_flag0=''

if "${lr_decay}";
then
        lr_decay_flag0="--lr_decay"
fi



lr_decay_flag=${lr_decay_flag0}

if "${use_pretrained_model}";
then
        lr_decay_flag="--use_pretrained_model ${lr_decay_flag0}"
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
  --data_dir ${data_dir} \
  --dataset ${dataset_name} \
  --valid_count ${valid_ratio_each_run} \
  --meta_lr ${meta_lr} \
  ${flip_label_flag} \
  ${bias_flip_str} \
  --err_label_ratio ${err_label_ratio} \
  --save_path ${save_path_prefix}_do_train/ \
  --cuda \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epochs ${epochs} \
  --do_train \
  ${metric_str} \
  ${lr_decay_flag} \
  ${label_aware_flag}
"


output_file_name=${output_dir}/output_${dataset_name}_${err_type}_${err_label_ratio}_${method}_do_train_0${suffix}.txt


echo "${exe_cmd} > ${output_file_name}"


#${exe_cmd} > ${output_file_name} 2>&1

<<cmd
exe_cmd="python -m torch.distributed.launch \
  --nproc_per_node 1 \
  --master_port ${port_num} \
  main_train.py \
  --load_dataset \
  --data_dir ${data_dir} \
  --dataset ${dataset_name} \
  --valid_count ${warm_up_valid_count} \
  --meta_lr ${meta_lr} \
  ${flip_label_flag} \
  ${bias_flip_str} \
  --err_label_ratio ${err_label_ratio} \
  --save_path ${save_path_prefix}_seq_select_0/ \
  --prev_save_path ${save_path_prefix}_do_train/\
  --cuda \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epochs ${epochs} \
  --total_valid_sample_count ${total_valid_sample_count} \
  ${metric_str} \
  ${lr_decay_flag} \
  ${label_aware_flag}"


output_file_name=${output_dir}/output_${dataset_name}_${err_type}_${err_label_ratio}_valid_select_seq_select_0_${method}${suffix}.txt

echo "${exe_cmd} > ${output_file_name}"

${exe_cmd} > ${output_file_name} 2>&1 

cmd




#for meta_num in "${meta_num_ls[@]}"
#do

echo "meta sample count::${total_valid_sample_count}"

exe_cmd="python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port ${port_num} \
    main_train.py \
    --load_dataset \
    --data_dir ${data_dir} \
    --dataset ${dataset_name} \
    --valid_count ${total_valid_sample_count} \
    --meta_lr ${meta_lr} \
    --not_save_dataset \
    ${flip_label_flag} \
    ${bias_flip_str} \
    --err_label_ratio ${err_label_ratio} \
    --save_path ${save_path_prefix}_seq_select_meta_num_${meta_num}/ \
    --prev_save_path ${save_path_prefix}_do_train/ \
    --cuda \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --epochs ${epochs} \
    ${metric_str} \
    ${add_valid_in_training_flag} \
        ${lr_decay_flag} \
	--total_valid_sample_count ${total_valid_sample_count} \
	${label_aware_flag}"


output_file_name=${output_dir}/output_${dataset_name}_${err_type}_${err_label_ratio}_valid_select_seq_select_0_${method}${suffix}_meta_num_${total_valid_sample_count}.txt

echo "${exe_cmd} > ${output_file_name}"

${exe_cmd} > ${output_file_name} 2>&1


#done

<<comment


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
    --data_dir ${data_dir} \
    --dataset ${dataset_name} \
    --valid_count ${valid_ratio_each_run} \
    --meta_lr ${meta_lr} \
    --not_save_dataset \
    ${flip_label_flag} \
    ${bias_flip_str} \
    --err_label_ratio ${err_label_ratio} \
    --save_path ${save_path_prefix}_seq_select_${k}/ \
    --prev_save_path ${save_path_prefix}_seq_select_$(( k - 1 ))/ \
    --cuda \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --epochs ${epochs} \
    ${label_aware_flag} \
    ${metric_str} \
    ${add_valid_in_training_flag} \
	${lr_decay_flag}"

	output_file_name=${output_dir}/output_${dataset_name}_${err_type}_${err_label_ratio}_valid_select_seq_select_${k}_${method}${suffix}.txt

	echo "${exe_cmd} > ${output_file_name}"
	
	${exe_cmd} > ${output_file_name} 2>&1 
	
done

comment

