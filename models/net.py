import torch
import torch.nn as nn


class sphere20(nn.Module):
    def __init__(self):
        super(sphere20, self).__init__()

        # input = B*3*112*96
        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)  # =>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.relu1_3 = nn.PReLU(64)

        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, 1)  # =>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)  # =>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_5 = nn.PReLU(128)

        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, 1)  # =>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_9 = nn.PReLU(256)

        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, 1)  # =>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512 * 7 * 6, 512)

        # Weight initialization
        # print('Initialization Network Parameter...')
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0), -1)
        x = self.fc5(x)

        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

class sphere20_bd(nn.Module):
    def __init__(self, dropout_Probability=0.2):
        super(sphere20_bd, self).__init__()

        # input = B*3*112*96
        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)  # =>B*64*56*48
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(64)
        self.relu1_3 = nn.PReLU(64)

        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, 1)  # =>B*128*28*24
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(128)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)  # =>B*128*28*24
        self.bn2_4 = nn.BatchNorm2d(128)
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.bn2_5 = nn.BatchNorm2d(128)
        self.relu2_5 = nn.PReLU(128)

        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, 1)  # =>B*256*14*12
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.bn3_4 = nn.BatchNorm2d(256)
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_5 = nn.BatchNorm2d(256)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.bn3_6 = nn.BatchNorm2d(256)
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_7 = nn.BatchNorm2d(256)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.bn3_8 = nn.BatchNorm2d(256)
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_9 = nn.BatchNorm2d(256)
        self.relu3_9 = nn.PReLU(256)

        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, 1)  # =>B*512*7*6
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.PReLU(512)

        
        self.dropout = nn.Dropout(p=dropout_Probability)
        self.fc5 = nn.Linear(512 * 7 * 6, 512)
        #self.bn5 = nn.BatchNorm1d(512)
        # Weight initialization
        # print('Initialization Network Parameter...')
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x = x + self.relu1_3(self.bn1_3(self.conv1_3(self.relu1_2(self.bn1_2(self.conv1_2(x))))))

        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = x + self.relu2_3(self.bn2_3(self.conv2_3(self.relu2_2(self.bn2_2(self.conv2_2(x))))))
        x = x + self.relu2_5(self.bn2_5(self.conv2_5(self.relu2_4(self.bn2_4(self.conv2_4(x))))))

        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = x + self.relu3_3(self.bn3_3(self.conv3_3(self.relu3_2(self.bn3_2(self.conv3_2(x))))))
        x = x + self.relu3_5(self.bn3_5(self.conv3_5(self.relu3_4(self.bn3_4(self.conv3_4(x))))))
        x = x + self.relu3_7(self.bn3_7(self.conv3_7(self.relu3_6(self.bn3_6(self.conv3_6(x))))))
        x = x + self.relu3_9(self.bn3_9(self.conv3_9(self.relu3_8(self.bn3_8(self.conv3_8(x))))))

        x = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        x = x + self.relu4_3(self.bn4_3(self.conv4_3(self.relu4_2(self.bn4_2(self.conv4_2(x))))))

        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc5(x)

        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


class sphere36_bd(nn.Module):
    def __init__(self, dropout_Probability=0.2):
        super(sphere36_bd, self).__init__()

        # input = B*3*112*96
        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)  # =>B*64*56*48
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(64)
        self.relu1_3 = nn.PReLU(64)

        self.conv1_4 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(64)
        self.relu1_4 = nn.PReLU(64)

        self.conv1_5 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn1_5 = nn.BatchNorm2d(64)
        self.relu1_5 = nn.PReLU(64)

        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, 1)  # =>B*128*28*24
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(128)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)  # =>B*128*28*24
        self.bn2_4 = nn.BatchNorm2d(128)
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.bn2_5 = nn.BatchNorm2d(128)
        self.relu2_5 = nn.PReLU(128)

        self.conv2_6 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.bn2_6 = nn.BatchNorm2d(128)
        self.relu2_6 = nn.PReLU(128)

        self.conv2_7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.bn2_7 = nn.BatchNorm2d(128)
        self.relu2_7 = nn.PReLU(128)

        self.conv2_8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.bn2_8 = nn.BatchNorm2d(128)
        self.relu2_8 = nn.PReLU(128)

        self.conv2_9 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.bn2_9 = nn.BatchNorm2d(128)
        self.relu2_9 = nn.PReLU(128)

        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, 1)  # =>B*256*14*12
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.bn3_4 = nn.BatchNorm2d(256)
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_5 = nn.BatchNorm2d(256)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.bn3_6 = nn.BatchNorm2d(256)
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_7 = nn.BatchNorm2d(256)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.bn3_8 = nn.BatchNorm2d(256)
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_9 = nn.BatchNorm2d(256)
        self.relu3_9 = nn.PReLU(256)

        self.conv3_10 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_10 = nn.BatchNorm2d(256)
        self.relu3_10 = nn.PReLU(256)

        self.conv3_11 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_11 = nn.BatchNorm2d(256)
        self.relu3_11 = nn.PReLU(256)

        self.conv3_12 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_12 = nn.BatchNorm2d(256)
        self.relu3_12 = nn.PReLU(256)

        self.conv3_13 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_13 = nn.BatchNorm2d(256)
        self.relu3_13 = nn.PReLU(256)

        self.conv3_14 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_14 = nn.BatchNorm2d(256)
        self.relu3_14 = nn.PReLU(256)

        self.conv3_15 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_15 = nn.BatchNorm2d(256)
        self.relu3_15 = nn.PReLU(256)

        self.conv3_16 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_16 = nn.BatchNorm2d(256)
        self.relu3_16 = nn.PReLU(256)

        self.conv3_17 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3_17 = nn.BatchNorm2d(256)
        self.relu3_17 = nn.PReLU(256)

        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, 1)  # =>B*512*7*6
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.PReLU(512)

        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.bn4_4 = nn.BatchNorm2d(512)
        self.relu4_4 = nn.PReLU(512)

        self.conv4_5 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.bn4_5 = nn.BatchNorm2d(512)
        self.relu4_5 = nn.PReLU(512)
       
        self.dropout = nn.Dropout(p=dropout_Probability)
        self.fc5 = nn.Linear(512 * 7 * 6, 512)
        self.bn5 = nn.BatchNorm1d(512)
        #self.bn5 = nn.BatchNorm1d(512)
        # Weight initialization
        # print('Initialization Network Parameter...')
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # Block one
        x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x = x + self.relu1_3(self.bn1_3(self.conv1_3(self.relu1_2(self.bn1_2(self.conv1_2(x))))))
        x = x + self.relu1_5(self.bn1_5(self.conv1_5(self.relu1_4(self.bn1_4(self.conv1_4(x))))))
        # Block two
        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = x + self.relu2_3(self.bn2_3(self.conv2_3(self.relu2_2(self.bn2_2(self.conv2_2(x))))))
        x = x + self.relu2_5(self.bn2_5(self.conv2_5(self.relu2_4(self.bn2_4(self.conv2_4(x))))))

        x = x + self.relu2_7(self.bn2_7(self.conv2_7(self.relu2_6(self.bn2_6(self.conv2_6(x))))))
        x = x + self.relu2_9(self.bn2_9(self.conv2_9(self.relu2_8(self.bn2_8(self.conv2_8(x))))))
        # Block three
        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = x + self.relu3_3(self.bn3_3(self.conv3_3(self.relu3_2(self.bn3_2(self.conv3_2(x))))))
        x = x + self.relu3_5(self.bn3_5(self.conv3_5(self.relu3_4(self.bn3_4(self.conv3_4(x))))))
        x = x + self.relu3_7(self.bn3_7(self.conv3_7(self.relu3_6(self.bn3_6(self.conv3_6(x))))))
        x = x + self.relu3_9(self.bn3_9(self.conv3_9(self.relu3_8(self.bn3_8(self.conv3_8(x))))))

        x = x + self.relu3_11(self.bn3_11(self.conv3_11(self.relu3_10(self.bn3_10(self.conv3_10(x))))))
        x = x + self.relu3_13(self.bn3_13(self.conv3_13(self.relu3_12(self.bn3_12(self.conv3_12(x))))))
        x = x + self.relu3_15(self.bn3_15(self.conv3_15(self.relu3_14(self.bn3_14(self.conv3_14(x))))))
        x = x + self.relu3_17(self.bn3_17(self.conv3_17(self.relu3_16(self.bn3_16(self.conv3_16(x))))))

        # Block four
        x = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        x = x + self.relu4_3(self.bn4_3(self.conv4_3(self.relu4_2(self.bn4_2(self.conv4_2(x))))))
        x = x + self.relu4_5(self.bn4_5(self.conv4_5(self.relu4_4(self.bn4_4(self.conv4_4(x))))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x) 
        x = self.fc5(x)
        x = self.bn5(x)
        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)
