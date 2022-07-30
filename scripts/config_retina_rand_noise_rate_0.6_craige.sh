err_label_ratio=0.6

dataset_name=retina
data_dir="/data1/wuyinjun/valid_set_selections/retina_data/"
save_path_root_dir="/data1/wuyinjun/valid_set_selections/retina_data/"
output_dir="/data1/wuyinjun/valid_set_selections/retina_data/"
gpu_ids=2
#total_valid_ratio=$6
repeat_times=2
port_num=10012
meta_lr=20
lr=0.001
batch_size=128
test_batch_size=128
epochs=30
#cached_model_name=${14}
add_valid_in_training_set=true
lr_decay=true
warm_up_valid_count=10
model_prov_period=20


valid_ratio_each_run=100 #$(( total_valid_ratio / repeat_times ))
bias_flip=false
method="uncertain"
total_valid_sample_count=100
use_pretrained_model=false
