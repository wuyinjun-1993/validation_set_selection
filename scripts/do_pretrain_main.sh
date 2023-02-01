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


metric_str=""
if [[ ${metric} == "auc" ]];
then
	metric_str="--metric auc"
elif [[ ${metric} == "kappa" ]];
then
	metric_str="--metric kappa"
fi



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



save_path_prefix=${save_path_root_dir}/${err_type}_${err_label_ratio}_valid_select_rand${suffix}


exe_cmd="python -m torch.distributed.launch \
  --nproc_per_node 1 \
  --master_port ${port_num} \
  main_train.py \
  --data_dir ${data_dir} \
  --dataset ${dataset_name} \
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
  ${lr_decay_flag}
"


output_file_name=${output_dir}/output_${dataset_name}_${err_type}_${err_label_ratio}_rand_do_train_0${suffix}.txt


echo "${exe_cmd} > ${output_file_name}"


${exe_cmd} > ${output_file_name} 2>&1


method_ls=(cluster_method_two cluster_method_three cluster_method_one rand ta finetune craige uncertain certain)

for method in "${method_ls[@]}"
do
	target_path="${save_path_root_dir}/${err_type}_${err_label_ratio}_valid_select_${method}${suffix}_do_train/"
	echo "target_path::${target_path}"
	mkdir -p ${target_path}
	echo "cp -r ${save_path_prefix}_do_train/* ${target_path}"
	cp -r ${save_path_prefix}_do_train/* ${target_path}
done






