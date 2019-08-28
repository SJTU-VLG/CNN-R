import tensorflow as tf
import numpy as np
import sys

class Model:
    @staticmethod
    def simpleNet(_X, _dropout, n_classes, is_Training):
        # input = tf.expand_dims(_X, -1)
        # block 1
        output = tf.layers.conv2d(_X,
                                 filters=256,
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
        return output
    
    # @staticmethod
    

    # @staticmethod
    

    @staticmethod
    def residual_net(x, n, n_classes, phase_train, scope='res_net'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            y = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='SAME', use_bias=False, \
                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            y = tf.contrib.layers.batch_norm(y, epsilon=1e-5, is_training=phase_train, decay=0.9, scale=True, center=True, updates_collections=None)
            y = tf.nn.relu(y, name='relu_init')
            # att = atten_layer(y, 64, phase_train)
            # y = y*att
            y = residual_group(y, 64, 64, n, False, phase_train, scope='group_1')
            att = atten_layer(y, 32, 64, phase_train)
            y = y*att
            y = residual_group(y, 64, 128, n, True, phase_train, scope='group_2')
            att = atten_layer(y, 16, 128, phase_train)
            y = y*att
            y = residual_group(y, 128, 256, n, True, phase_train, scope='group_3')
            y = tf.layers.conv2d(y, filters=256, kernel_size=1, strides=1, padding='SAME', use_bias=True, \
                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            y = tf.nn.avg_pool(y, [1, 8, 8, 1], [1, 1, 1, 1], 'VALID', name='avg_pool')
            y = tf.contrib.layers.flatten(y)
            # y = 30*tf.divide(y, tf.norm(y, ord='euclidean'))
            feature = y
            y = tf.layers.dense(y, units=256,  kernel_initializer=tf.contrib.layers.xavier_initializer())
            y = tf.layers.dense(y, units=n_classes, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            y_2 = tf.layers.dense(feature, units=256,  kernel_initializer=tf.contrib.layers.xavier_initializer())
            y_2 = tf.layers.dense(y_2, units=100, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
        return feature, att, y, y_2

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

def atten_layer(x, side_len, n_out, phase_train):
    y = tf.layers.conv2d(x, filters=1, kernel_size=3, strides=1, padding='SAME', use_bias=False, \
                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    y = tf.contrib.layers.batch_norm(y, epsilon=1e-5, is_training=phase_train, decay=0.9, scale=True, center=True, updates_collections=None )
    y = tf.nn.tanh(y)
    y = tf.reshape(y, (-1, side_len*side_len))
    y = tf.layers.dense(y, units=side_len*side_len, activation=tf.nn.relu)
    y = tf.layers.dense(y, units=side_len*side_len, activation=tf.nn.relu)
    y = tf.nn.softmax(y)
    y = tf.reshape(y, (-1, side_len, side_len, 1))
    return y
