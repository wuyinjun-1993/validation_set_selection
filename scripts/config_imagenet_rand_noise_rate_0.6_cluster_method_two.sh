err_label_ratio=0.6

dataset_name=imagenet
data_dir="/data4/wuyinjun/valid_set_selections/imagenet/transformed_images/"
save_path_root_dir="/data4/wuyinjun/valid_set_selections/imagenet/"
output_dir="/data4/wuyinjun/valid_set_selections/imagenet/"
gpu_ids=2
#total_valid_ratio=$6
repeat_times=10
port_num=10012
meta_lr=5
lr=0.002
batch_size=32
test_batch_size=32
epochs=30
#cached_model_name=${14}
add_valid_in_training_set=true
lr_decay=true
warm_up_valid_count=10
model_prov_period=20


valid_ratio_each_run=50 #$(( total_valid_ratio / repeat_times ))
bias_flip=false
method="cluster_method_two"
total_valid_sample_count=50
use_pretrained_model=true
