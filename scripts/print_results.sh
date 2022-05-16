dataset=$1
noise_type=$2
res_dir=$3

if [ ${noise_type} = "imbalanced" ]
then
  err_params=(0.005 0.01 0.02 0.05 0.1 1)
fi

if [ ${dataset} = "cifar10" ]
then
  valid_count=100
elif [ ${dataset} = "cifar100" ]
then
  valid_count=1000
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
  for method in random uncertainty certainty ours1
  do
    result_dir=${res_dir}/logs_${dataset}_${method}_${noise_type}_${err_param}_lr_0.1_pretrained_select_${valid_count}_1_1/master_log.txt
    line=$( tail -n 1 ${result_dir} )
    echo ${method} ${noise_type} ${err_param}:
    echo ${line}
  done
  echo ""
done
