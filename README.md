# How to run the code for validation set selections


Example usage of my code for training a clean model on the fully cleaned version of MNIST dataset (suppose the log directory is /path/to/logs/,  the data directory is /path/to/data/, the gpu is ${gpu_id} and the gpu number is ${num_gpus}). 


To randomly select 0.2% of training samples as validation samples, we can use the following command


```
cd src/main/
CUDA_VISIBLE_DEVICES=${gpu_id} python -m torch.distributed.launch --nproc_per_node ${num_gpus} --master_port 10032 main_train.py --load_dataset --nce-k 200 --data_dir /path/to/data/ --dataset MNIST --valid_ratio 0.002 --meta_lr 50 --flip_labels --err_label_ratio 0.9 --save_path /path/to/logs/ --cuda --lr 0.1 --batch_size 4096 --test_batch_size 256 --epochs 1000
```


in which "--flip_labels" determines whether to pollute the training set and  "--err_label_ratio" determine how many samples are polluted, "--load_dataset" is used to load the existing random airty labels generated by the previous run (if not generate newly dirty labels), which is for controlling the randomness from the generated noisy labels, and "--nce-k" represents the number of hidden neurons in the neural network.
 



In contrast, we can use the following command to select 0.2% validation samples by using our clustering method. Note that this is separated into three phases. In the first phase, we pretrain the model by using all the noisy training samples, which is assumed to have log directory "/path/to/logs0/". Then in the second phase, we use our clustering method to collect the remaining set of samples, which will use some cached information from the previous run with random sampling. So it will use two log directories, one is "/path/to/logs0/" (the log directory of the first run with random sampling) while the other one is "/path/to/logs1/" (the log directory of the current run with our clustering method):

```
cd src/main/
###initial training by using all noisy training samples
## for cifar10 dataset:
CUDA_VISIBLE_DEVICES=${gpu_id} python -m torch.distributed.launch --nproc_per_node ${num_gpus} --master_port 10032 main_train.py --load_dataset --nce-k 200 --data_dir /path/to/data/ --dataset cifar10 --valid_ratio 0.002 --meta_lr 20 --flip_labels --err_label_ratio 0.6 --save_path /path/to/logs0/ --cuda --lr 0.1 --batch_size 128 --test_batch_size 128 --epochs 200 --lr_decay --do_train

## for MNIST dataset:

CUDA_VISIBLE_DEVICES=${gpu_id} python -m torch.distributed.launch --nproc_per_node ${num_gpus} --master_port 10032 main_train.py --load_dataset --nce-k 200 --data_dir /path/to/data/ --dataset MNIST --valid_ratio 0.002 --meta_lr 50 --flip_labels --err_label_ratio 0.9 --save_path /path/to/logs0/ --cuda --lr 0.1 --batch_size 4096 --test_batch_size 128 --epochs 1000 --do_train


###sampling by using our clustering method 

## for MNIST dataset (for the first time)
CUDA_VISIBLE_DEVICES=${gpu_id} python -m torch.distributed.launch --nproc_per_node ${num_gpus} --master_port 10032 main_train.py --select_valid_set  --nce-k 200 --data_dir /path/to/data/ --dataset MNIST --valid_ratio 0.001 --meta_lr 50 --flip_labels --err_label_ratio 0.9 --save_path /path/to/logs1/ --prev_save_path /path/to/logs0/ --cuda --lr 0.1 --batch_size 4096 --test_batch_size 256 --epochs 1000 --cluster_method_two --cosin_dist


## for MNIST dataset (for the second time and onwards)
CUDA_VISIBLE_DEVICES=${gpu_id} python -m torch.distributed.launch --nproc_per_node ${num_gpus} --master_port 10032 main_train.py --select_valid_set --continue_label --load_cached_weights --cached_sample_weights_name cached_sample_weights --cached_model_name MNIST_current.pth --nce-k 200 --data_dir /path/to/data/ --dataset MNIST --valid_ratio 0.001 --meta_lr 50 --flip_labels --err_label_ratio 0.9 --save_path /path/to/logs1/ --prev_save_path /path/to/logs0/ --cuda --lr 0.1 --batch_size 4096 --test_batch_size 256 --epochs 1000 --cluster_method_two --cosin_dist

## for cifar10 dataset (first time)
CUDA_VISIBLE_DEVICES=${gpu_id} python -m torch.distributed.launch --nproc_per_node ${num_gpus} --master_port 10032 main_train.py  --load_dataset --select_valid_set --nce-k 200 --data_dir /path/to/data/ --dataset cifar10 --valid_ratio 0.001 --meta_lr 40 --flip_labels --err_label_ratio 0.6 --save_path /path/to/logs1/ --prev_save_path /path/to/logs0/ --cuda --lr 0.1 --batch_size 128 --test_batch_size 128 --epochs 200 --cluster_method_two --cosin_dist --lr_decay

## for cifar10 dataset (for the second time and onwards)
CUDA_VISIBLE_DEVICES=${gpu_id} python -m torch.distributed.launch --nproc_per_node ${num_gpus} --master_port 10032 main_train.py --continue_label --load_cached_weights --cached_sample_weights_name cached_sample_weights  --load_dataset --select_valid_set --nce-k 200 --data_dir /path/to/data/ --dataset cifar10 --valid_ratio 0.001 --meta_lr 40 --flip_labels --err_label_ratio 0.6 --save_path /path/to/logs2/ --prev_save_path /path/to/logs1/ --cuda --lr 0.1 --batch_size 128 --test_batch_size 128 --epochs 200 --cluster_method_two --cosin_dist --lr_decay

```


in which "--continue_label" represents to load the cached training datasets from the previous run (from which we can further select the remaining 0.1% validation samples), "--load_cached_weights" is a flag to indicate the cached weights on training samples produced by the previous run, "--cached_sample_weights_name" represents the file name of cached weights from the previous run, "--prev_save_path" represents the log directory of the previous run, "--select_valid_set" is a flag to indicate the use of our clustering method, "--lr_decay" is a flag for decaying the learning rate for meta-learning, "--cosin_dist" is a flag for using cosine similarity during k-means clustering, "--cluster_method_two" indicates the cluster method that we are using, which is the default method (so this flag will be deprecated in the next version).






We can also run the combination of the above two commands by using running a script in "./scripts/":
```
cd scripts/

bash sequential_label_valid_set.sh MNIST /path/to/data/ /path/to/parent/logs/ /output/dir/ 2 0.001 2 10032 50 0.1 4096 256  1000 MNIST_current.pth 
```

in which "/path/to/parent/logs/" represents the a folder to contains all the logs from different runs of sample selections (each run will have a separate subfolder to contain logs) and "/output/dir/" represents a folder to store the output information of each run.


