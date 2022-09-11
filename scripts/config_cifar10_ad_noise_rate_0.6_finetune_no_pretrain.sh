err_label_ratio=0.6

dataset_name=cifar10
data_dir="/data6/wuyinjun/valid_set_selections/cifar10_1/"
save_path_root_dir="/data6/wuyinjun/valid_set_selections/cifar10_1/"
output_dir="/data6/wuyinjun/valid_set_selections/cifar10_1/"
gpu_ids=0
#total_valid_ratio=$6
repeat_times=10
port_num=10320
meta_lr=30
lr=0.1
batch_size=128
test_batch_size=128
epochs=150
#cached_model_name=${14}
add_valid_in_training_set=true
lr_decay=true
warm_up_valid_count=10
model_prov_period=20


valid_ratio_each_run=20 #$(( total_valid_ratio / repeat_times ))
bias_flip=true
method="finetune"
total_valid_sample_count=20
use_pretrained_model=true
