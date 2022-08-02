err_label_ratio=0.6

dataset_name=retina
data_dir="/data1/wuyinjun/valid_set_selections/retina_data/"
save_path_root_dir="/data1/wuyinjun/valid_set_selections/retina_data/"
output_dir="/data1/wuyinjun/valid_set_selections/retina_data/"
gpu_ids=0
#total_valid_ratio=$6
repeat_times=2
port_num=10110
meta_lr=20
lr=0.002
batch_size=128
test_batch_size=128
epochs=100
#cached_model_name=${14}
add_valid_in_training_set=true
lr_decay=true
warm_up_valid_count=2
model_prov_period=20


valid_ratio_each_run=100 #$(( total_valid_ratio / repeat_times ))
bias_flip=false
method="finetune"
total_valid_sample_count=100
use_pretrained_model=true
metric='auc'
