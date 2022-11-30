data_dir=$1
res_dir=$2

echo Running BaseModel training
CUDA_VISIBLE_DEVICES=0 ./train_basemodel.sh toy biased 0.60 0.2 10031 ${data_dir} ${res_dir} &
wait

# echo Running warmup
# CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh toy biased random 1 0.4 0.01 20 6 10030 ${data_dir} ${res_dir} True &
# wait

echo Running different methods for reweighting
CUDA_VISIBLE_DEVICES=0 ./sequential_label.sh toy biased ours1 3 0.60 0.1 40 2 10030 ${data_dir} ${res_dir} False &
CUDA_VISIBLE_DEVICES=1 ./sequential_label.sh toy biased ours2 3 0.60 0.1 40 2 10031 ${data_dir} ${res_dir} False &
CUDA_VISIBLE_DEVICES=2 ./sequential_label.sh toy biased random 3 0.60 0.1 40 2 10032 ${data_dir} ${res_dir} False &
wait

echo Finished!
