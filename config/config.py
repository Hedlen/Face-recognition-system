class Config(object):
    env = 'default'
    backbone = 'sphere36_bd'
    classify = 'softmax'
    #num_classes = 8631
    num_classes = 10533
    metric = 'sphere'
    tryt = 1
    easy_margin = False
    use_se = True

    display = False  # display vis
    display_tx = True # display tensoboardx
    finetune = False
    val_flag = False
    train_root = '/mnt/lustre/jiangting/Hedlen/github/datasets/CASIA-WebFace-align'
   
    #split train and val 
    train_list = './data/train_label_list.txt'
    val_list = './data/val_label_list.txt'
    # only train set
    train_all_list = '/mnt/lustre/jiangting/Hedlen/github/datasets/CASIA-WebFace-align.txt'

    # mix 
    mix_flag = False
    mix_nums = 2
    mix_train = [["mix1_train_root",
                  "mix1_train_list",
                  "mix1_val_list",
                  "mix1_train_all_list"],
                 ["mix2_train_root",
                  "mix2_train_list",
                  "mix2_val_list",
                  "mix2_train_all_list"]]

    test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    test_list = 'test.txt'

    lfw_root = '/mnt/lustre/jiangting/Hedlen/github/datasets/lfw-deepfunneled-align/'
    lfw_test_list = './data/pairs.txt'
    lfw_test_list_fixe_pairs = './data/lfw_test_pair.txt'
    checkpoints_path = 'checkpoints/'+metric+'/'
    load_model_path = 'models/resnet18.pth'
    test_model_path = 'checkpoints/resnet18_0.pth'
    restore_checkpoints = "None"
    save_interval = 5

    train_batch_size = 512  # batch size
    test_batch_size = 512

    input_shape = (3, 128, 128)

    optimizer = 'mom'
    mom = 0.9
    dropout_Probability = 0.4
    use_gpu = True  # use GPU or not
    #gpu_id = '0, 1'
    gpu_id = '0, 1'
    num_workers = 64  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 30
    lr = 1e-1  # initial learning rate
    #lr_step = 10
    step_size = [16000,24000]
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
