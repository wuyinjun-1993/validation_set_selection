err_label_ratio=0.6

dataset_name=MNIST
data_dir="/data6/wuyinjun/valid_set_selections/mnist_0/"
save_path_root_dir="/data6/wuyinjun/valid_set_selections/mnist_0/"
output_dir="/data6/wuyinjun/valid_set_selections/mnist_0/"
gpu_ids=1
#total_valid_ratio=$6
repeat_times=10
port_num=10041
meta_lr=20
lr=0.1
batch_size=4096
test_batch_size=4096
epochs=500
#cached_model_name=${14}
add_valid_in_training_set=true
lr_decay=false
warm_up_valid_count=100
model_prov_period=20


valid_ratio_each_run=20 #$(( total_valid_ratio / repeat_times ))
bias_flip=false
method="cluster_method_three"
total_valid_sample_count=20
use_pretrained_model=false
