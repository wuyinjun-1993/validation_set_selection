#How to run the code for validation set selections

`
python main_train.py --valid_ratio 0.005 --meta_lr 50 --not_save_dataset --err_label_ratio 0.7 --flip_labels --gpu_id ${gpu_id} --save_path /data6/wuyinjun/valid_set_selections/rand_select_do_meta_train_1/ --data_dir /data6/wuyinjun/valid_set_selections/ --cuda --lr 0.2 --batch_size 4096 --test_batch_size 256 --epochs 400
`
