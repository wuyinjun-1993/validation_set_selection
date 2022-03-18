import logging
import os

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # if args.do_train:
    log_file = os.path.join(args.save_path, 'train.log')
    # else:
    #     log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)