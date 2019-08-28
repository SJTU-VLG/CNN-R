import tensorflow as tf
import sys
from model import Model
from data_providers.cifar import *
from datetime import datetime
def main():
    train_dir = './dataset/'
    test_dir = './dataset/'

    init_lr = 0.001
    disturb_lr = 0.00001
    training_iters = 80000
    batch_size = 16
    display_step = 100
    disturb_step = 20
    test_step = 400
    num = 10
    load_weight = True
    weight_decay = 0.0001
    n_classes = 10
    keep_rate_train = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.5, 0.5]
    keep_rate_test = [1, 1, 1, 1, 1, 1, 1, 1]
    # parameter
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    Y = tf.placeholder(tf.int64, [None])
    # X_M = tf.placeholder(tf.float32, [None, 32, 32, 3])
    CT = tf.placeholder(tf.float32, [None])
    keep_var = tf.placeholder(tf.float32, [None])
    # num = tf.placeholder(tf.int32)
    learning_rate = tf.placeholder(tf.float32)
    is_Training = tf.placeholder(tf.bool)
    #prediction result
    feature, pred = Model.residual_net(X, num, n_classes, is_Training)
    # feature_M, pred_M = Model.residual_net(X_M, num, n_classes, is_Training)
    #loss and optimizer
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    couple_pred = tf.reshape(feature,[120,2,256])
    test_anchor, test_pn = tf.unstack(couple_pred, axis=1)
    contrastive_loss = contrasive_loss(test_anchor, test_pn, CT)
    loss = tf.losses.sparse_softmax_cross_entropy(logits = pred, labels = Y)
    optimizer2 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(loss + 0.001*l2_loss)  #, use_nesterov=True
    optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(contrastive_loss)
    #evaluation 
    correct_pred = tf.equal(tf.argmax(pred, 1), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #Init
    init = tf.global_variables_initializer()
    #load dataset
    dataset = Cifar10AugmentedDataProvider(validation_size=None, one_hot=False, normalization='by_channels', shuffle='every_epoch')
    dataset.train.start_new_epoch()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        print('Init variable!')
        if load_weight:
            saver.restore(sess, "./Model/model_1_c.ckpt")
        else:
            sess.run(init)
        print('start training!')
        step = 1
        lr = init_lr
        while step < training_iters: 
            if step == training_iters/2:
                lr /= 10
                disturb_lr /= 10
            if step == training_iters*3/4:
                lr /= 10
                disturb_lr /= 10
            batch_xs, batch_ys = dataset.train.next_batch(batch_size)
            labels = []
            labels_y = []
            batch_xM = []
            for i in range(0, batch_ys.shape[0]-1):
                for j in range(i+1, batch_ys.shape[0]):
                    batch_xM.append(batch_xs[i])
                    batch_xM.append(batch_xs[j])
                    labels_y.append(batch_ys[i])
                    labels_y.append(batch_ys[j])
                    if batch_ys[i] == batch_ys[j]:
                        labels.append(0)
                    else:
                        labels.append(1)
            batch_xM = np.array(batch_xM, dtype=np.float32)
            batch_ys = np.array(labels_y)

            batch_ys_contra = np.array(labels)
            sess.run(optimizer2, feed_dict={X: batch_xM, Y: batch_ys, CT: batch_ys_contra, keep_var: keep_rate_train, learning_rate: lr, is_Training: True})
            if step % disturb_step == 0:
                sess.run(optimizer1, feed_dict={X: batch_xM, Y: batch_ys, CT: batch_ys_contra, keep_var: keep_rate_train, learning_rate: disturb_lr, is_Training: True})
            if step%test_step == 0:
                saver.save(sess, './Model/model_res_1.ckpt')
                test_acc = 0.0
                test_count = 0
                for _ in range(dataset.test.size//20):
                    batch_tx, batch_ty = dataset.test.next_batch(20)
                    acc = sess.run(accuracy, feed_dict={X: batch_tx, Y: batch_ty, keep_var: keep_rate_test, is_Training: False})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print( "{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_acc))
            
            if step%display_step == 0:
                acc = sess.run(accuracy, feed_dict={X: batch_xM, Y: batch_ys, keep_var: keep_rate_test, is_Training: False})
                batch_loss = sess.run(loss, feed_dict={X: batch_xM, Y: batch_ys, keep_var: keep_rate_test, is_Training: False})
                contra_loss = sess.run(contrastive_loss, feed_dict={X: batch_xM, Y: batch_ys, CT: batch_ys_contra, keep_var: keep_rate_test, is_Training: False})
                # print( "{} Iter {}: Training Loss = {:.4f}".format(datetime.now(), step, batch_loss))
                print( "{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}, Contrastive loss: {:.4f}".format(datetime.now(), step, batch_loss, acc, contra_loss))
            step += 1
        saver.save(sess, './Model/model_res_1.ckpt')
        print('finish!!!')

def  contrasive_loss(left, right, label):
    margin = 3
    d = tf.reduce_sum(tf.square(left - right), 1)
    d_sqrt = tf.sqrt(d)

    loss = label * tf.square(tf.maximum(0., margin - d_sqrt)) + (1 - label) * d

    loss = tf.reduce_mean(loss)

    return loss
if __name__ == '__main__':
    main()