import tensorflow as tf
import numpy as np
import sys

class Model:
    @staticmethod
    def simpleNet(_X, _dropout, n_classes, is_Training):
        # input = tf.expand_dims(_X, -1)
        # block 1
        output = tf.layers.conv2d(_X,
                                 filters=128,
                                 kernel_size=3,
                                 padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 )
        output = tf.contrib.layers.batch_norm(output,
                                 is_training = is_Training,
                                 epsilon=1e-5,
                                 decay=0.9,
                                 scale=True,
                                 center=True,
                                 updates_collections=None)
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, _dropout[0])
        output = tf.layers.conv2d(output,
                                 filters=128,
                                 kernel_size=3,
                                 padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        output = tf.contrib.layers.batch_norm(output,
                                 is_training = is_Training,
                                 epsilon=1e-5,
                                 decay=0.9,
                                 scale=True,
                                 center=True,
                                 updates_collections=None)
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, _dropout[1])
        output = tf.layers.conv2d(output,
                                 filters=128,
                                 kernel_size=3,
                                 padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 )
        output = tf.contrib.layers.batch_norm(output,
                                 is_training = is_Training,
                                 epsilon=1e-5,
                                 decay=0.9,
                                 scale=True,
                                 center=True,
                                 updates_collections=None)
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, _dropout[0])
        output = tf.layers.conv2d(output,
                                 filters=128,
                                 kernel_size=3,
                                 padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        output = tf.contrib.layers.batch_norm(output,
                                 is_training = is_Training,
                                 epsilon=1e-5,
                                 decay=0.9,
                                 scale=True,
                                 center=True,
                                 updates_collections=None)
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, _dropout[1])
        y1 = output
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)
        # block 2
        output = tf.layers.conv2d(output,
                                 filters=256,
                                 kernel_size=3,
                                 padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        output = tf.contrib.layers.batch_norm(output,
                                 is_training = is_Training,
                                 epsilon=1e-5,
                                 decay=0.9,
                                 scale=True,
                                 center=True,
                                 updates_collections=None)
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, _dropout[2])
        output = tf.layers.conv2d(output,
                                 filters=256,
                                 kernel_size=3,
                                 padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        output = tf.contrib.layers.batch_norm(output,
                                 is_training = is_Training,
                                 epsilon=1e-5,
                                 decay=0.9,
                                 scale=True,
                                 center=True,
                                 updates_collections=None)
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, _dropout[3])
        output = tf.layers.conv2d(output,
                                 filters=256,
                                 kernel_size=3,
                                 padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        output = tf.contrib.layers.batch_norm(output,
                                 is_training = is_Training,
                                 epsilon=1e-5,
                                 decay=0.9,
                                 scale=True,
                                 center=True,
                                 updates_collections=None)
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, _dropout[2])
        output = tf.layers.conv2d(output,
                                 filters=256,
                                 kernel_size=3,
                                 padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        output = tf.contrib.layers.batch_norm(output,
                                 is_training = is_Training,
                                 epsilon=1e-5,
                                 decay=0.9,
                                 scale=True,
                                 center=True,
                                 updates_collections=None)
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, _dropout[3])
        y2 = output
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)
        
        # block3
        output = tf.layers.conv2d(output,
                                 filters=512,
                                 kernel_size=3,
                                 padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        output = tf.contrib.layers.batch_norm(output,
                                 is_training = is_Training,
                                 epsilon=1e-5,
                                 decay=0.9,
                                 scale=True,
                                 center=True,
                                 updates_collections=None)
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, _dropout[4])
        output = tf.layers.conv2d(output,
                                 filters=512,
                                 kernel_size=3,
                                 padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        output = tf.contrib.layers.batch_norm(output,
                                 is_training = is_Training,
                                 epsilon=1e-5,
                                 decay=0.9,
                                 scale=True,
                                 center=True,
                                 updates_collections=None)
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, _dropout[5])
        output = tf.layers.conv2d(output,
                                 filters=512,
                                 kernel_size=3,
                                 padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        output = tf.contrib.layers.batch_norm(output,
                                 is_training = is_Training,
                                 epsilon=1e-5,
                                 decay=0.9,
                                 scale=True,
                                 center=True,
                                 updates_collections=None)
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, _dropout[4])
        output = tf.layers.conv2d(output,
                                 filters=512,
                                 kernel_size=3,
                                 padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        output = tf.contrib.layers.batch_norm(output,
                                 is_training = is_Training,
                                 epsilon=1e-5,
                                 decay=0.9,
                                 scale=True,
                                 center=True,
                                 updates_collections=None)
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, _dropout[5])
        y3 = output
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)
        
        output = tf.contrib.layers.flatten(output)

        output = tf.layers.dense(output, units=1024, use_bias=False, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        output = tf.contrib.layers.batch_norm(output, epsilon=1e-5, is_training=is_Training, decay=0.9, scale=True, center=True, updates_collections=None)
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, _dropout[6])

        output = tf.layers.dense(output, units=1024, use_bias=False, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        output = tf.contrib.layers.batch_norm(output, epsilon=1e-5, is_training=is_Training, decay=0.9, scale=True, center=True, updates_collections=None)
        output = tf.nn.relu(output)

        # output = 30*tf.divide(output, tf.norm(output, ord='euclidean'))
        output = tf.nn.dropout(output, _dropout[7])
        
        output = tf.layers.dense(output, units=n_classes, kernel_initializer=tf.contrib.layers.xavier_initializer())

        #using lstm
        y1 = tf.layers.conv2d(y1, filters=128, kernel_size=2, strides=2, padding='VALID')
        y1 = tf.nn.avg_pool(y1, [1, 4, 4, 1], [1, 4, 4, 1], 'VALID')
        y1 = tf.contrib.layers.flatten(y1)
        y2 = tf.layers.conv2d(y2, filters=512, kernel_size=2, strides=2, padding='VALID')
        y2 = tf.nn.avg_pool(y2, [1, 4, 4, 1], [1, 4, 4, 1], 'VALID')
        y2 = tf.contrib.layers.flatten(y2)
        y3 = tf.nn.avg_pool(y3, [1, 4, 4, 1], [1, 4, 4, 1], 'VALID')
        y3 = tf.contrib.layers.flatten(y3)
        y1 = tf.expand_dims(y1, 1)
        y2 = tf.expand_dims(y2, 1)
        y3 = tf.expand_dims(y3, 1)
        x_in = tf.concat([y1,y2,y3], axis=1)
        cell = tf.contrib.rnn.BasicLSTMCell(512, state_is_tuple=True)
        _, state = tf.nn.dynamic_rnn(cell, x_in, dtype=tf.float32)
        y_l = state.h
        y_l = tf.layers.dense(y_l, units=1024, use_bias=False, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        y_l = tf.contrib.layers.batch_norm(y_l, epsilon=1e-5, is_training=is_Training, decay=0.9, scale=True, center=True, updates_collections=None)
        y_l = tf.nn.relu(y_l)
        # y_l = tf.nn.dropout(y_l, _dropout[6])
        y_l = tf.layers.dense(y_l, units=1024, use_bias=False, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        y_l = tf.contrib.layers.batch_norm(y_l, epsilon=1e-5, is_training=is_Training, decay=0.9, scale=True, center=True, updates_collections=None)
        y_l = tf.nn.relu(y_l)
        # y_l = tf.nn.dropout(y_l, _dropout[7])
        y_l = tf.layers.dense(y_l, units=n_classes, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())

        return y_l, output
    
    # @staticmethod
    

    # @staticmethod
    

    @staticmethod
    def residual_net(x, n, n_classes, phase_train, scope='res_net'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            y = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='SAME', use_bias=False, \
                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            y = tf.contrib.layers.batch_norm(y, epsilon=1e-5, is_training=phase_train, decay=0.9, scale=True, center=True, updates_collections=None)
            y = tf.nn.relu(y, name='relu_init')
            y1 = residual_group(y, 64, 64, n, False, phase_train, scope='group_1')
            y2 = residual_group(y1, 64, 64, n, False, phase_train, scope='group_2')
            y3 = residual_group(y2, 64, 128, n, True, phase_train, scope='group_3')
            y4 = residual_group(y3, 128, 128, n, False, phase_train, scope='group_4')
            y5 = residual_group(y4, 128, 256, n, True, phase_train, scope='group_5')
            y6 = residual_group(y5, 256, 256, n, False, phase_train, scope='group_6')
            y = tf.nn.avg_pool(y6, [1, 8, 8, 1], [1, 1, 1, 1], 'VALID', name='avg_pool')
            y = tf.contrib.layers.flatten(y)
            

            #LSTM part
            y1 = tf.layers.conv2d(y1, filters=64, kernel_size=2, strides=2, padding='VALID')
            y1 = tf.nn.avg_pool(y1, [1, 8, 8, 1], [1, 8, 8, 1], 'VALID')
            y1 = tf.contrib.layers.flatten(y1)
            y2 = tf.layers.conv2d(y2, filters=64, kernel_size=2, strides=2, padding='VALID')
            y2 = tf.nn.avg_pool(y2, [1, 8, 8, 1], [1, 8, 8, 1], 'VALID')
            y2 = tf.contrib.layers.flatten(y2)
            y3 = tf.layers.conv2d(y3, filters=256, kernel_size=2, strides=2, padding='VALID')
            y3 = tf.nn.avg_pool(y3, [1, 8, 8, 1], [1, 8, 8, 1], 'VALID')
            y3 = tf.contrib.layers.flatten(y3)
            y4 = tf.layers.conv2d(y4, filters=256, kernel_size=2, strides=2, padding='VALID')
            y4 = tf.nn.avg_pool(y4, [1, 8, 8, 1], [1, 8, 8, 1], 'VALID')
            y4 = tf.contrib.layers.flatten(y4)
            y5 = tf.nn.avg_pool(y5, [1, 8, 8, 1], [1, 8, 8, 1], 'VALID')
            y5 = tf.contrib.layers.flatten(y5)
            y6 = tf.nn.avg_pool(y6, [1, 8, 8, 1], [1, 8, 8, 1], 'VALID')
            y6 = tf.contrib.layers.flatten(y6)
            y1 = tf.expand_dims(y1, 1)
            y2 = tf.expand_dims(y2, 1)
            y3 = tf.expand_dims(y3, 1)
            y4 = tf.expand_dims(y4, 1)
            y5 = tf.expand_dims(y5, 1)
            y6 = tf.expand_dims(y6, 1)
 
            x_in = tf.concat([y1,y2,y3,y4,y5,y6], axis=1)
            # print(x_in.get_shape)
            cell = tf.contrib.rnn.BasicLSTMCell(512, state_is_tuple=True)
            _, state = tf.nn.dynamic_rnn(cell, x_in, dtype=tf.float32)
            y_l = state.h

            y_c = tf.layers.dense(y, units=256, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
            y_c = tf.layers.dense(y_c, units=n_classes, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            # y_l = tf.concat([y, y_l], axis = 1)
            # print(y_l.get_shape())
            y_l = tf.layers.dense(y_l, units=256, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
            y_l = tf.layers.dense(y_l, units=n_classes, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())

        return y_c, y_l

def residual_block(x, n_in, n_out, subsample, phase_train, scope='res_block'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if subsample:
                y = tf.layers.conv2d(x, filters=n_out, kernel_size=3, strides=2, padding='SAME', use_bias=False, \
                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                shortcut = tf.layers.conv2d(x, filters=n_out, kernel_size=3, strides=2, padding='SAME', use_bias=False, \
                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            else:
                y = tf.layers.conv2d(x, filters=n_out, kernel_size=3, strides=1, padding='SAME', use_bias=False, \
                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                shortcut = tf.identity(x, name='shortcut')
            y = tf.contrib.layers.batch_norm(y, epsilon=1e-5, is_training=phase_train, decay=0.9, scale=True, center=True, updates_collections=None )
            y = tf.nn.relu(y, name='relu_1')
            y = tf.layers.conv2d(y, filters=n_out, kernel_size=3, strides=1, padding='SAME', use_bias=True, \
                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            y = tf.contrib.layers.batch_norm(y, epsilon=1e-5, is_training=phase_train, decay=0.9, scale=True, center=True, updates_collections=None )
            y = y + shortcut
            y = tf.nn.relu(y, name='relu_2')
        return y

def residual_group(x, n_in, n_out, n, first_subsample, phase_train, scope='res_group'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            y = residual_block(x, n_in, n_out, first_subsample, phase_train, scope='block_1')
            for i in range(n - 1):
                y = residual_block(y, n_out, n_out, False, phase_train, scope='block_%d' % (i + 2))
        return y