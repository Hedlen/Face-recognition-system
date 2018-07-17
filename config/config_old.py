class Config(object):
    env = 'default'
    backbone = 'sphere36_bd'
    classify = 'softmax'
    num_classes = 8631
    #num_classes = 10574
    metric = 'add_margin'
    tryt = 14
    easy_margin = False
    use_se = True

    display = False
    display_tx = True
    finetune = False

    train_root = '/mnt/lustre/Hypnus/DATA/vggface2/train_sdk_aligned/'
    train_list = '/mnt/lustre/Hypnus/DATA/vggface2/vggface2_lists/trainset_train.txt'
    val_list = '/mnt/lustre/Hypnus/DATA/vggface2/vggface2_lists/trainset_val.txt'

    test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    test_list = 'test.txt'

    lfw_root = '/mnt/lustre/jiangting/Hedlen/github/eveface/data/lfw-deepfunneled_sdk_aligned_crop/'
    lfw_test_list = './data/pairs.txt'
    lfw_test_list_fixe_pairs = './data/lfw_test_pair.txt'
    checkpoints_path = 'checkpoints'
    load_model_path = 'models/resnet18.pth'
    test_model_path = 'checkpoints/resnet18_0.pth'
    restore_checkpoints = "None"
    save_interval = 5

    train_batch_size = 512  # batch size
    test_batch_size = 512

    input_shape = (3, 128, 128)

    optimizer = 'mom'
    mom = 0.9
    dropout_Probability = 0.2
    use_gpu = True  # use GPU or not
    #gpu_id = '0, 1'
    gpu_id = '0, 1'
    num_workers = 64  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 100
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
