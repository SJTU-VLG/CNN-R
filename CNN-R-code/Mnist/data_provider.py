import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import struct


class Dataset:
    def __init__(self, train_dir, test_dir):
        self.train_data = self.decode_idx3_ubyte(train_dir+'train-images-idx3-ubyte')
        self.train_label = self.decode_idx1_ubyte(train_dir+'train-labels-idx1-ubyte')
        self.test_data = self.decode_idx3_ubyte(test_dir+'t10k-images-idx3-ubyte')
        self.test_label = self.decode_idx1_ubyte(test_dir+'t10k-labels-idx1-ubyte')
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = self.train_data.shape[0]
        self.test_size = self.test_data.shape[0]
        self.n_classes = 10
        # self.shuffle()

    def shuffle(self):
        train = np.concatenate((self.train_data, self.train_label), axis = 1)
        np.random.shuffle(train)
        self.train_data = train[:,:-1]
        self.train_label = train[:,-1]
        test = np.concatenate((self.test_data, self.test_label), axis = 1)
        np.random.shuffle(test)
        self.test_data = test[:,:-1]
        self.test_label = test[:,-1]

    def next_batch(self, batch_size, phase):
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                data_batch = self.train_data[self.train_ptr:self.train_ptr+batch_size]
                labels_batch = self.train_label[self.train_ptr:self.train_ptr+batch_size]
                self.train_ptr += batch_size
            else:
                new_ptr = (self.train_ptr + batch_size)%self.train_size
                data_batch = np.concatenate((self.train_data[self.train_ptr:], self.train_data[:new_ptr]), axis = 0)
                labels_batch = np.concatenate((self.train_label[self.train_ptr:], self.train_label[:new_ptr]), axis = 0)
                self.train_ptr = new_ptr
        elif phase == 'test':
            if self.test_ptr + batch_size < self.test_size:
                data_batch = self.test_data[self.test_ptr:self.test_ptr+batch_size]
                labels_batch = self.test_label[self.test_ptr:self.test_ptr+batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size)%self.test_size
                data_batch = np.concatenate((self.test_data[self.test_ptr:], self.test_data[:new_ptr]), axis = 0)
                labels_batch = np.concatenate((self.test_label[self.test_ptr:], self.test_label[:new_ptr]), axis = 0)
                self.test_ptr = new_ptr
        else:
            return None, None
        one_hot_labels = labels_batch
        aug_data = np.zeros((batch_size, 32, 32, 3))
        aug_data[:, 2:30, 2:30, 0] = data_batch
        aug_data[:, 2:30, 2:30, 1] = data_batch
        aug_data[:, 2:30, 2:30, 2] = data_batch

        return aug_data/255, one_hot_labels
     
    def decode_idx3_ubyte(self, idx3_ubyte_file):
        """
        解析idx3文件的通用函数
        :param idx3_ubyte_file: idx3文件路径
        :return: 数据集
        """
        # 读取二进制数据
        bin_data = open(idx3_ubyte_file, 'rb').read()

        offset = 0
        fmt_header = '>iiii'
        magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
        print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

        # 解析数据集
        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)
        fmt_image = '>' + str(image_size) + 'B'
        images = np.empty((num_images, num_rows, num_cols))
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print('已解析 %d' % (i + 1) + '张')
            images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
            offset += struct.calcsize(fmt_image)
        return images

    def decode_idx1_ubyte(self, idx1_ubyte_file):
        """
        解析idx1文件的通用函数
        :param idx1_ubyte_file: idx1文件路径
        :return: 数据集
        """
        # 读取二进制数据
        bin_data = open(idx1_ubyte_file, 'rb').read()
        # 解析文件头信息，依次为魔数和标签数
        offset = 0
        fmt_header = '>ii'
        magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
        print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

        # 解析数据集
        offset += struct.calcsize(fmt_header)
        fmt_image = '>B'
        labels = np.empty(num_images)
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print('已解析 %d' % (i + 1) + '张')
            labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
            offset += struct.calcsize(fmt_image)
        labels = np.eye(int(np.max(labels))+1)[labels.astype(int)]
        return labels
