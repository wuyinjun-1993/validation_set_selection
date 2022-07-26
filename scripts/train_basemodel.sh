cd ../src/main/

dataset=$1          # mnist, cifar10, or cifar100
noise_type=$2       # uniform, biased, or imbalanced
err_param=$3        # percent of training data to corrupt
lr=$4
port=$5
data_dir=$6
res_dir=$7

if [ ${noise_type} = "uniform" ];
then
  noise_cmd="--flip_labels"
elif [ ${noise_type} = "biased" ];
then
  noise_cmd="--flip_labels --biased_flip --err_label_ratio ${err_param}"
elif [ ${noise_type} = "imbalanced" ];
then
  noise_cmd="--bias_classes --imb_factor ${err_param}"
elif [ ${noise_type} = "imbalanced_uniform" ];
then
  noise_cmd="--flip_labels --err_label_ratio 0.6 --bias_classes --imb_factor ${err_param}"
fi

result_dir=${res_dir}/logs_${dataset}_${noise_type}_${err_param}_lr_${lr}_batchsize_128_basemodel/
echo saving to ${result_dir}
mkdir -p ${result_dir}
python -m torch.distributed.launch \
  --nproc_per_node 1 \
  --master_port ${port} \
  main_train.py \
  --data_dir ${data_dir} \
  --dataset ${dataset} \
  --valid_count 0 \
  ${noise_cmd} \
  --save_path ${result_dir} \
  --cuda \
  --lr ${lr} \
  --batch_size 128 \
  --test_batch_size 256 \
  --epochs 200 \
  --do_train > ${result_dir}/master_log.txt 2>&1
