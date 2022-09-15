err_label_ratio=0.6

dataset_name=imagenet
data_dir="/data4/wuyinjun/valid_set_selections/imagenet/transformed_images/"
save_path_root_dir="/data4/wuyinjun/valid_set_selections/imagenet/transformed_images/"
output_dir="/data4/wuyinjun/valid_set_selections/imagenet/transformed_images/"
gpu_ids=1
#total_valid_ratio=$6
repeat_times=2
port_num=10441
meta_lr=20
lr=0.02
batch_size=32
test_batch_size=32
epochs=40
#cached_model_name=${14}
add_valid_in_training_set=true
lr_decay=true
warm_up_valid_count=2
model_prov_period=100


valid_ratio_each_run=100 #$(( total_valid_ratio / repeat_times ))
bias_flip=false
method="rand"
total_valid_sample_count=100
use_pretrained_model=true

