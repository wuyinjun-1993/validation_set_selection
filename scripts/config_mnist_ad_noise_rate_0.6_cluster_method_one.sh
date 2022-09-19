err_label_ratio=0.6

dataset_name=MNIST
data_dir="/data6/wuyinjun/valid_set_selections/mnist_1/"
save_path_root_dir="/data6/wuyinjun/valid_set_selections/mnist_1/"
output_dir="/data6/wuyinjun/valid_set_selections/mnist_1/"
gpu_ids=2
#total_valid_ratio=$6
repeat_times=10
port_num=10362
meta_lr=20
lr=0.1
batch_size=4096
test_batch_size=4096
epochs=500
#cached_model_name=${14}
add_valid_in_training_set=true
lr_decay=false
warm_up_valid_count=10
model_prov_period=40


valid_ratio_each_run=30 #$(( total_valid_ratio / repeat_times ))
bias_flip=true
method="cluster_method_one"
total_valid_sample_count=30
use_pretrained_model=false
