import tensorflow as tf
import sys
from model_attention import Model
import numpy as np
from Mnist.data_provider import Dataset as Data_mnist
from data_providers.cifar import *
from datetime import datetime
import scipy.misc
def main():
    train_dir = './dataset/'
    test_dir = './dataset/'

    train_mnist = './Mnist/dataset/'
    test_mnist = './Mnist/dataset/'

    kinds = 1

    init_lr = 0.01
    training_iters = 80000
    batch_size = 32
    display_step = 100
    test_step = 1000
    num = 10
    load_weight = True
    weight_decay = 0.0001
    n_classes = 10
    keep_rate_train = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.5, 0.5]
    keep_rate_test = [1, 1, 1, 1, 1, 1, 1, 1]
    # parameter
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    Y = tf.placeholder(tf.float32, [None, n_classes])
    Y2 = tf.placeholder(tf.float32, [None, 100])
    keep_var = tf.placeholder(tf.float32, [None])
    # num = tf.placeholder(tf.int32)
    learning_rate = tf.placeholder(tf.float32)
    is_Training = tf.placeholder(tf.bool)
    #prediction result
    feature, att, pred, pred2 = Model.residual_net(X, num, n_classes, is_Training)
    #loss and optimizer
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
    loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred2, labels=Y2))
    # loss = tf.losses.sparse_softmax_cross_entropy(logits = pred, labels = Y)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(loss + 0.0001*l2_loss)  #, use_nesterov=True
    optimizer2 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(loss_2 + 0.0001*l2_loss)  #, use_nesterov=True
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss + 0.001*l2_loss)
    #evaluation 
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    correct_pred_2 = tf.equal(tf.argmax(pred2, 1), tf.argmax(Y2, 1))
    accuracy_2 = tf.reduce_mean(tf.cast(correct_pred_2, tf.float32))
    #Init
    init = tf.global_variables_initializer()
    #load dataset
    # dataset_mnist = Data_mnist(train_mnist, test_mnist)
    dataset = Cifar10AugmentedDataProvider(validation_size=None, one_hot=True, normalization='by_channels', shuffle='every_epoch')

    dataset2 = Cifar100AugmentedDataProvider(validation_size=None, one_hot=True, normalization='by_channels', shuffle='every_epoch')
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        print('Init variable!')
        if load_weight:
            saver.restore(sess, "./Model/model_100_att_3.ckpt")
        else:
            sess.run(init)
        print('start training!')
        step = 1
        lr = init_lr
        while step < training_iters: 
            if step == training_iters/2:
                lr /= 10
            if step == training_iters*3/4:
                lr /= 10
            batch_xs, batch_ys = dataset.train.next_batch(batch_size)
            
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, keep_var: keep_rate_train, learning_rate: lr, is_Training: True})
            
            if step%test_step == 0:
                saver.save(sess, './Model/model_100_att_3.ckpt')
                test_acc = 0.0
                test_count = 0
                for _ in range(dataset.test.size//20):
                    batch_tx, batch_ty = dataset.test.next_batch(20)
                    # batch_mnist_test, label_mnist_test = dataset_mnist.next_batch(20, 'test')
                    # batch_tx[np.where(batch_mnist_test != 0)] = batch_mnist_test[np.where(batch_mnist_test != 0)]
                    acc = sess.run(accuracy, feed_dict={X: batch_tx, Y: batch_ty, keep_var: keep_rate_test, is_Training: False})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print( "{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_acc))
            
            if step%display_step == 0:
                acc = sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, keep_var: keep_rate_test, is_Training: False})
                batch_loss = sess.run(loss, feed_dict={X: batch_xs, Y: batch_ys, keep_var: keep_rate_test, is_Training: False})
                print( "{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}".format(datetime.now(), step, batch_loss, acc))

            # batch_xs100, batch_ys100 = dataset2.train.next_batch(batch_size)
            # sess.run(optimizer2, feed_dict={X: batch_xs100, Y2: batch_ys100, keep_var: keep_rate_train, learning_rate: lr, is_Training: True})
            # if step%display_step == 0:
            #     acc = sess.run(accuracy_2, feed_dict={X: batch_xs100, Y2: batch_ys100, keep_var: keep_rate_test, is_Training: False})
            #     batch_loss = sess.run(loss_2, feed_dict={X: batch_xs100, Y2: batch_ys100, keep_var: keep_rate_test, is_Training: False})
            #     print( "{} Iter {}: Training Loss = {:.4f}, Cifar100 Accuracy = {:.4f}".format(datetime.now(), step, batch_loss, acc))
            # if step%test_step == 0:
            #     saver.save(sess, './Model/model_100_att_3.ckpt')
            #     test_acc = 0.0
            #     test_count = 0
            #     for _ in range(dataset2.test.size//20):
            #         batch_tx, batch_ty = dataset2.test.next_batch(20)
            #         # batch_mnist_test, label_mnist_test = dataset_mnist.next_batch(20, 'test')
            #         # batch_tx[np.where(batch_mnist_test != 0)] = batch_mnist_test[np.where(batch_mnist_test != 0)]
            #         acc = sess.run(accuracy_2, feed_dict={X: batch_tx, Y2: batch_ty, keep_var: keep_rate_test, is_Training: False})
            #         test_acc += acc
            #         test_count += 1
            #     test_acc /= test_count
            #     print( "{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_acc))
            step += 1
            # attention = sess.run(att, feed_dict={X: batch_xs, Y: batch_ys, keep_var: keep_rate_train, learning_rate: lr, is_Training: False})
            # attention = attention[0]
            # scipy.misc.imsave('outfile.jpg', attention[:,:,0]*255)
            # print(batch_xs)
            # scipy.misc.imsave('img.jpg', batch_xs[0]*100)
        saver.save(sess, './Model/model_100_att_3.ckpt')
        print('finish!!!')

def  contrasive_loss(left, right, label):
    margin = 1
    d = tf.reduce_sum(tf.square(left_output - right_output), 1)
    d_sqrt = tf.sqrt(d)

    loss = label * tf.square(tf.maximum(0., margin - d_sqrt)) + (1 - label) * d

    loss = 0.5 * tf.reduce_mean(loss)

if __name__ == '__main__':
    main()