import tensorflow as tf
import numpy as np
import sys

class Model:
    @staticmethod
    def residual_net(x, n, n_classes, phase_train, scope='res_net'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            y = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding='SAME', use_bias=False, \
                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            y = tf.contrib.layers.batch_norm(y, epsilon=1e-5, is_training=phase_train, decay=0.9, scale=True, center=True, updates_collections=None)
            y = tf.nn.relu(y, name='relu_init')
            y = residual_group(y, 32, 32, n, False, phase_train, scope='group_1')
            y = residual_group(y, 32, 64, n, True, phase_train, scope='group_2')
            y = residual_group(y, 64, 128, n, True, phase_train, scope='group_3')
            y = tf.layers.conv2d(y, filters=256, kernel_size=1, strides=1, padding='SAME', use_bias=True, \
                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            y = tf.nn.avg_pool(y, [1, 8, 8, 1], [1, 1, 1, 1], 'VALID', name='avg_pool')
            y = tf.contrib.layers.flatten(y)
            # y = 30*tf.divide(y, tf.norm(y, ord='euclidean'))
            feature = y
            y = tf.layers.dense(y, units=256,  kernel_initializer=tf.contrib.layers.xavier_initializer())
            y = tf.layers.dense(y, units=n_classes, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
        return feature, y

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