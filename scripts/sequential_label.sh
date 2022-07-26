cd ../src/main/

dataset=$1          # mnist, cifar10, or cifar100
noise_type=$2       # uniform, biased, or imbalanced
valid_selection=$3  # ours, uncertainty, or random
num_iterations=$4
err_param=$5        # percent of training data to corrupt
lr=$6
meta_lr=$7
valid_count=$8
port=$9
data_dir=${10}
res_dir=${11}
warmup=${12}

if [ ${noise_type} = "uniform" ];
then
  noise_cmd="--flip_labels --err_label_ratio ${err_param}"
elif [ ${noise_type} = "biased" ];
then
  noise_cmd="--flip_labels --biased_flip --err_label_ratio ${err_param}"
elif [ ${noise_type} = "imbalanced" ];
then
  noise_cmd="--bias_classes --imb_factor ${err_param} --all_layer_grad_no_full_loss"
elif [ ${noise_type} = "imbalanced_uniform" ];
then
  noise_cmd="--flip_labels --err_label_ratio 0.4 --bias_classes --imb_factor ${err_param}"
fi

if [ ${noise_type} = "imbalanced" ] && [ ${valid_selection} != "ours1" ] && [ ${valid_selection} != "ours2" ];
then
  noise_cmd+=" --clustering_by_class"
fi

if [ ${valid_selection} = "ours1" ];
then
  # selection_cmd="--select_valid_set --cluster_method_three --cosin_dist --lr_decay --weight_by_norm --use_model_prov --model_prov_period 10 --replace"
  selection_cmd="--select_valid_set \
    --cluster_method_two \
    --cluster_method_two_plus \
    --cluster_method_two_sampling \
    --cluster_method_two_sample_col_count 1000 \
    --no_sample_weights_k_means \
    --not_rescale_features \
    --weight_by_norm \
    --remove_empty_clusters \
    --use_model_prov \
    --model_prov_period 10 \
    --replace \
    --cosin_dist \
    --lr_decay"
elif [ ${valid_selection} = "ours2" ];
then
  selection_cmd="--select_valid_set \
    --cluster_method_three \
    --cluster_method_three_sampling \
    --cluster_method_three_sample_col_count 1000 \
    --no_sample_weights_k_means \
    --not_rescale_features \
    --weight_by_norm \
    --remove_empty_clusters \
    --use_model_prov \
    --model_prov_period 10 \
    --replace \
    --cosin_dist \
    --lr_decay"
elif [ ${valid_selection} = "uncertainty" ];
then
  selection_cmd="--uncertain_select --lr_decay"
elif [ ${valid_selection} = "certainty" ];
then
  selection_cmd="--certain_select --lr_decay"
elif [ ${valid_selection} = "random" ];
then
  selection_cmd="--lr_decay"
elif [ ${valid_selection} = "finetune" ];
then
  selection_cmd="--finetune --lr_decay"
elif [ ${valid_selection} = "glc" ];
then
  selection_cmd="--glc_train --lr_decay"
elif [ ${valid_selection} = "TA-VAAL" ];
then
  selection_cmd="--ta_vaal_train --lr_decay"
fi

if [ ${warmup} = "True" ];
then
  result_dir=${res_dir}/logs_${dataset}_${noise_type}_${err_param}_lr_0.1_batchsize_128_basemodel/
else
  if [ ${dataset} = "cifar10" ];
  then
    warm_count=10
  else
    warm_count=100
  fi
  result_dir=${res_dir}/logs_${dataset}_random_${noise_type}_${err_param}_lr_${lr}_pretrained_select_${warm_count}_1_1
fi

# Sequentially train the model using a selected "clean" set
for (( k=1; k<=${num_iterations}; k++ ))
do
  prev_result_dir=${result_dir}
  result_dir=${res_dir}/logs_${dataset}_${valid_selection}_${noise_type}_${err_param}_lr_${lr}_pretrained_select_${valid_count}_${k}_${num_iterations}
  echo Iter ${k}, result dir: ${result_dir}, prev dir: ${prev_result_dir}
  mkdir -p ${result_dir}

  exe_cmd="python -m torch.distributed.launch \
  --nproc_per_node 1 \
  --master_port ${port} \
  main_train.py \
  --load_dataset \
  --use_pretrained_model \
  ${selection_cmd} \
  --nce-t 0.07 \
  --nce-k 200 \
  --data_dir ${data_dir} \
  --dataset ${dataset} \
  --valid_count ${valid_count} \
  --total_valid_sample_count ${valid_count} \
  --meta_lr ${meta_lr} \
  ${noise_cmd} \
  --save_path ${result_dir} \
  --prev_save_path ${prev_result_dir} \
  --cuda \
  --lr ${lr} \
  --batch_size 128 \
  --test_batch_size 256 \
  --epochs 100"
   
  output_file_name=${result_dir}/master_log.txt

  if [ ${warmup} != "True" ]
  then
    ${exe_cmd} --continue_label --load_cached_weights --cached_sample_weights_name cached_sample_weights > ${output_file_name} 2>&1
  else
    ${exe_cmd} > ${output_file_name} 2>&1
  fi

done
