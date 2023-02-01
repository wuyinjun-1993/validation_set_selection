err_label_ratio=0.6

dataset_name=toy
data_dir="/data6/wuyinjun/valid_set_selections/synthetic/"
save_path_root_dir="/data6/wuyinjun/valid_set_selections/synthetic/"
output_dir="/data6/wuyinjun/valid_set_selections/synthetic/"
gpu_ids=0
#total_valid_ratio=$6
repeat_times=2
port_num=10330
meta_lr=40
lr=0.1
batch_size=128
test_batch_size=128
epochs=200
#cached_model_name=${14}
add_valid_in_training_set=true
lr_decay=true
warm_up_valid_count=10
model_prov_period=10


valid_ratio_each_run=10 #$(( total_valid_ratio / repeat_times ))
bias_flip=true
method="rand"
total_valid_sample_count=10
use_pretrained_model=true
real_noise=false

