err_label_ratio=0.6

dataset_name=imagenet
data_dir="/data6/wuyinjun/valid_set_selections/imagenet/transformed_images/"
save_path_root_dir="/data6/wuyinjun/valid_set_selections/imagenet/"
output_dir="/data6/wuyinjun/valid_set_selections/imagenet/"
gpu_ids=1
#total_valid_ratio=$6
repeat_times=10
port_num=10011
meta_lr=20
lr=0.1
batch_size=32
test_batch_size=32
epochs=20
#cached_model_name=${14}
add_valid_in_training_set=true
lr_decay=true
warm_up_valid_count=10
model_prov_period=20


valid_ratio_each_run=50 #$(( total_valid_ratio / repeat_times ))
bias_flip=true
method="certain"
total_valid_sample_count=50
use_pretrained_model=true
