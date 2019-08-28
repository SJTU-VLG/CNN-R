import tensorflow as tf
import sys
from model import Model
from data_provider import Dataset
from network import *
from datetime import datetime
def main():
    train_dir = './dataset/'
    test_dir = './dataset/'

    learning_rate = 0.001
    training_iters = 40000
    batch_size = 200
    display_step = 100
    test_step = 400

    n_classes = 10
    keep_rate = 1
    # parameter
    X = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
    Y = tf.placeholder(tf.float32, [None, n_classes])
    keep_var = tf.placeholder(tf.float32)
    #prediction result
    feature, pred = Model.residual_net(X, 10, n_classes, True)
    #loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    #evaluation
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #Init
    init = tf.initialize_all_variables()
    #load dataset
    dataset = Dataset(train_dir, test_dir)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        print('Init variable!')
        sess.run(init)
        print('start training!')
        step = 1
        while step < training_iters:
            batch_xs, batch_ys = dataset.next_batch(batch_size, 'train')
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, keep_var: keep_rate})

            if step%test_step == 0:
                test_acc = 0.0
                test_count = 0
                for _ in range(int(dataset.test_size/batch_size)):
                    batch_tx, batch_ty = dataset.next_batch(batch_size, 'test')
                    acc = sess.run(accuracy, feed_dict={X: batch_tx, Y: batch_ty, keep_var: 1.})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print(sys.stderr, "{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_acc))
            
            if step%display_step == 0:
                acc = sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, keep_var: 1.})
                batch_loss = sess.run(loss, feed_dict={X: batch_xs, Y: batch_ys, keep_var: 1.})
                print(sys.stderr, "{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}".format(datetime.now(), step, batch_loss, acc))

            if step%4000 == 0:
                saver.save(sess, './Model/model_res.ckpt')
            step += 1
        saver.save(sess, './Model/model_res.ckpt')
        print('finish!!!')

if __name__ == '__main__':
    main()