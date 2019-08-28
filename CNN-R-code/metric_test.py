import tensorflow as tf
import sys
from model import Model
from data_providers.cifar import *
from datetime import datetime
import numpy as np
from numpy import *
def main():
    train_dir = './dataset/'
    test_dir = './dataset/'

    init_lr = 0.01
    training_iters = 390
    batch_size = 128
    display_step = 1000
    test_step = 4000
    num = 10
    load_weight = True
    weight_decay = 0.0001
    n_classes = 10
    keep_rate_train = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.5, 0.5]
    keep_rate_test = [1, 1, 1, 1, 1, 1, 1, 1]
    # parameter
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    Y = tf.placeholder(tf.float32, [None, n_classes])
    keep_var = tf.placeholder(tf.float32, [None])
    # num = tf.placeholder(tf.int32)
    learning_rate = tf.placeholder(tf.float32)
    is_Training = tf.placeholder(tf.bool)
    #prediction result
    feature, pred = Model.residual_net(X, num, n_classes, is_Training)
    #loss and optimizer
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
    # loss = tf.losses.sparse_softmax_cross_entropy(logits = pred, labels = Y)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(loss + 0.001*l2_loss)  #, use_nesterov=True
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss + 0.001*l2_loss)
    #evaluation 
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #Init
    init = tf.global_variables_initializer()
    #load dataset
    dataset = Cifar10AugmentedDataProvider(validation_size=None, one_hot=True, normalization='by_channels', shuffle='every_epoch')
    dataset.train.start_new_epoch()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        print('Init variable!')
        if load_weight:
            saver.restore(sess, "./Model/model_1.ckpt")
        else:
            sess.run(init)
        print('start training!')

        features = np.zeros((n_classes, 128), dtype=np.float32)
        count = np.zeros((n_classes), dtype=np.int32)
        step = 1
        lr = init_lr
        while step < training_iters: 
            batch_xs, batch_ys = dataset.train.next_batch(batch_size)
            fea = sess.run(pred, feed_dict={X: batch_xs, Y: batch_ys, keep_var: keep_rate_train, learning_rate: lr, is_Training: False})

            for i in range(batch_ys.shape[0]):
                ind = np.where(batch_ys[i] == 0.91)[0][0]
                features[ind] = features[ind] + fea[i]
                count[ind] += 1
            step += 1
        for i in range(n_classes):
            features[i] /= count[i]
        
        np.save('feature.npy', features)

        features = np.load('feature.npy')

        test_acc = 0
        wrong = 0
        wrong_img = []
        wrong_class = []
        for i in range(10000):
            batch_ts, batch_ty = dataset.test.next_batch(1)
            fea = sess.run(pred, feed_dict={X: batch_ts, Y: batch_ty, keep_var: keep_rate_train, learning_rate: lr, is_Training: False})
            dis_min = 1000000
            ind = -1
            for i in range(n_classes):
                dis = np.sqrt(np.sum(np.square(fea[0] - features[i])))
                # dis = dot(fea[0],features[i])/(linalg.norm(fea[0])*linalg.norm(features[i]))
                if dis < dis_min:
                    dis_min = dis
                    ind = i
            # print(ind, np.where(batch_ty[0] == 0.91)[0][0])
            if ind == np.where(batch_ty[0] == 0.91)[0][0]:
                test_acc += 1
            else:
                wrong_img.append(batch_ts[0])
                wrong_class.append(ind)
        wrong_img = np.array(wrong_img)
        wrong_class = np.array(wrong_class)
        np.save('img.npy', wrong_img)
        np.save('class.npy',wrong_class)
        print(test_acc)
            # if step == training_iters/2:
            #     lr /= 10
            # if step == training_iters*3/4:
            #     lr /= 10
            # batch_xs, batch_ys = dataset.train.next_batch(batch_size)
            # sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, keep_var: keep_rate_train, learning_rate: lr, is_Training: True})
        #     if step%test_step == 0:
        #         test_acc = 0.0
        #         test_count = 0
        #         for _ in range(dataset.test.size//20):
        #             batch_tx, batch_ty = dataset.test.next_batch(20)
        #             acc = sess.run(accuracy, feed_dict={X: batch_tx, Y: batch_ty, keep_var: keep_rate_test, is_Training: False})
        #             test_acc += acc
        #             test_count += 1
        #         test_acc /= test_count
        #         print( "{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_acc))
            
        #     if step%display_step == 0:
        #         acc = sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, keep_var: keep_rate_test, is_Training: False})
        #         batch_loss = sess.run(loss, feed_dict={X: batch_xs, Y: batch_ys, keep_var: keep_rate_test, is_Training: False})
        #         print( "{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}".format(datetime.now(), step, batch_loss, acc))
        #     step += 1
        # saver.save(sess, './Model/model_1.ckpt')
        print('finish!!!')

if __name__ == '__main__':
    main()