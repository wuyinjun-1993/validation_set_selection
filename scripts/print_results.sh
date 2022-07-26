dataset=$1
noise_type=$2
res_dir=$3

if [ ${noise_type} = "imbalanced" ]
then
  err_params=(0.005 0.01)
fi

if [ ${dataset} = "cifar10" ]
then
  valid_count=20
elif [ ${dataset} = "cifar100" ]
then
  valid_count=200
fi

echo BaseModel Results
for err_param in "${err_params[@]}"
do
  result_dir=${res_dir}/logs_${dataset}_${noise_type}_${err_param}_lr_0.1_batchsize_128_basemodel/master_log.txt
  line=$( tail -n 1 ${result_dir} )
  echo ${noise_type} ${err_param} BaseModel:
  echo ${line}
done
echo ""

for err_param in "${err_params[@]}"
do
  for method in random uncertainty certainty finetune ours1 ours2
  do
    if [ ${method} = "ours1" ] || [ ${method} = "ours2" ]
    then
      result_dir=${res_dir}/logs_${dataset}_${method}_${noise_type}_${err_param}_lr_0.1_pretrained_select_${valid_count}_3_3/master_log.txt
    else
      result_dir=${res_dir}/logs_${dataset}_${method}_${noise_type}_${err_param}_lr_0.1_pretrained_select_${valid_count}_1_1/master_log.txt
    fi
    line=$( tail -n 1 ${result_dir} )
    echo ${method} ${noise_type} ${err_param}:
    echo ${line}
  done
  echo ""
done
