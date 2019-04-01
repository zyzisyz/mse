# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import os
import time
import tensorflow as tf
import numpy as np
import random
from model.model_utils import *
from scipy import stats


def shuffle_set(data, label):
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    label = label[index]
    return data, label


def shuffle_data(data, table):
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    table = table[index]
    return data, table


def MLP_net(input, id, n_hidden, acitvate="elu", keep_prob=1, init_stddev=0.2, istrain=True):
    '''
    # 我自己封装的MPL单层函数
    # 论文中的example是用单层的MPL，激活函数采用tanh，我不知道这里用其他是否有影响
    '''
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.)

    w_str = 'w'+str(id)
    b_str = 'b'+str(id)

    w = tf.get_variable(
        w_str, [input.get_shape()[1], n_hidden], initializer=w_init)
    b = tf.get_variable(b_str, [n_hidden], initializer=b_init)

    output = tf.matmul(input, w) + b

    if acitvate == 'tanh':
        output = tf.nn.tanh(output)
    elif acitvate == 'sigmoid':
        output = tf.nn.sigmoid(output)
    else:
        output = tf.nn.elu(output)
    return output


def loader(path="./data/test_vector.npz"):
    return np.load(path)['vector'], np.load(path)['label']


class VAE(object):
    model_name = "VAE"     # name for checkpoint

    def __init__(self,
                 sess,
                 epoch=20,
                 batch_size=64,
                 z_dim=512,
                 n_hidden=1024,
                 dataset_path='./data/d_train.npz',
                 checkpoint_dir='./experiments/baseline/checkpoint',
                 log_dir='./experiments/baseline/log',
                 keep_prob=1,
                 beta1=0.5,
                 learning_rate=0.00005,
                 b=0.04,
                 mse2_start_e=20,
                 k=0.1
                 ):
        self.k = k
        self.sess = sess
        self.b = b
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.mse2_start_e = mse2_start_e

        # load data
        self.dataset_path = dataset_path
        self.input_data, self.input_utt = loader(dataset_path)

        '''input_data | spk_list'''
        # vector 对应的说话人 int label
        self.spk_list = np.load(
            './data/voxceleb_combined_200000/spk.npz')['spk']
        # 所有说话人 int label
        self.spker = np.load(
            './data/voxceleb_combined_200000/spk.npz')['check']

        '''
        计算spker人数
        self.spk_count_shape = (spk_num, z_dim)
        '''
        spk_count = [0 for _ in range(len(self.spker))]
        for i in self.spker:
            spk_count[int(i)] += 1

        self.spk_count = []
        for i in range(len(spk_count)):
            temp = [spk_count[i] for _ in range(z_dim)]
            self.spk_count.append(temp)

        self.epoch = epoch
        self.batch_size = batch_size

        # get number of batches for a single epoch
        # //表示向下取整
        self.num_batches = len(self.input_data) // self.batch_size

        self.z_dim = z_dim
        self.keep_prob = keep_prob
        self.n_hidden = n_hidden  # MLP的隐含层维度

        self.dnn_input_dim = 512  # 输入的维度为512
        self.dnn_output_dim = 512   # 恢复的也是512

        self.z_dim = z_dim         # dimension of v-vector

        # train
        self.learning_rate = learning_rate
        self.beta1 = beta1

    # Gaussian MLP Encoder
    def MPL_encoder(self, x, n_hidden, n_output, keep_prob):
        with tf.variable_scope("gaussian_MLP_encoder"):

            # initializers
            # 这里是我瞎改的初始化值
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

            # layer-0
            net = MLP_net(input=x, id=0, n_hidden=n_hidden, acitvate='sigmoid',
                          keep_prob=keep_prob)
            # layer-1
            net = MLP_net(input=net, id=1, n_hidden=n_hidden, acitvate='tanh',
                          keep_prob=keep_prob)

            wo = tf.get_variable(
                'wo', [net.get_shape()[1], n_output], initializer=w_init)
            bo = tf.get_variable('bo', [n_output], initializer=b_init)

            gaussian_params = tf.matmul(net, wo) + bo
            mean = gaussian_params

            stddev = tf.constant(
                self.k, shape=[n_output], dtype=tf.float32)
            stddev = self.sess.run(stddev)

        return mean, stddev

    # Bernoulli decoder
    def MLP_decoder(self, z, n_hidden, n_output, keep_prob):
        with tf.variable_scope("bernoulli_MLP_decoder"):
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)
            # layer-0
            net = MLP_net(input=z, id=0, n_hidden=n_hidden, acitvate="tanh",
                          keep_prob=keep_prob)
            # layer-1
            net = MLP_net(input=net, id=3, n_hidden=n_hidden, acitvate='sigmoid',
                          keep_prob=keep_prob)
            # output layer-mean
            wo = tf.get_variable(
                'wo', [net.get_shape()[1], n_output], initializer=w_init)
            bo = tf.get_variable('bo', [n_output], initializer=b_init)
            y = tf.matmul(net, wo) + bo
        return y

    def update_table(self, mean):
        '''
        返回utt_table | 用于计算mse2
        mse2 让相同spker的utt的mean尽可能的接近
        '''

        mean = np.array(mean, dtype=np.float32)
        num_spker = len(self.spker)  # 不同spker 人数
        counter = np.array(self.spk_count, dtype=int)  # 存储每个spker的utt个数

        '''初始化mean值矩阵为0'''
        spk_table = np.zeros(shape=(num_spker, self.z_dim), dtype=np.float32)

        '''加上mean值'''
        for i in range(mean.shape[0]):
            spk_table[self.spk_list[i]] += mean[i]

        '''计算mean的平均值'''

        spk_table = spk_table/counter

        '''算mse2的utt table'''
        utt_table = np.zeros(shape=mean.shape, dtype=np.float32)
        for i in range(utt_table.shape[0]):
            utt_table[i] += spk_table[self.spk_list[i]][0]

        return utt_table

    def build_model(self):
        #######################################################################
        '''网络结构'''
        #  输入inputs的feed
        self.inputs = tf.placeholder(
            tf.float32, [None, self.dnn_input_dim], name='input_vector')
        self.mean_table = tf.placeholder(
            tf.float32, [None, self.z_dim], name='mean_vector')

        # encoding
        self.mu, self.sigma = self.MPL_encoder(
            self.inputs, self.n_hidden, self.z_dim, self.keep_prob)

        # sampling by re-parameterization technique
        z = self.mu + self.sigma * \
            tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

        # decoding
        self.out = self.MLP_decoder(
            z, self.n_hidden, self.dnn_output_dim, self.keep_prob)
        #######################################################################

        '''reconstruct mse'''
        re_mse = tf.reduce_sum(tf.square(self.inputs-self.out), 1)
        self.re_mse = 2*(1-self.b)*tf.reduce_mean(re_mse)

        '''KL散度'''
        KL_divergence = tf.reduce_sum(tf.square(self.mu) + tf.square(
            self.sigma) - tf.log(1e-8 + tf.square(self.sigma)) - 1, [1])
        self.KL_divergence = self.b*0.5*tf.reduce_mean(KL_divergence)

        mse2 = tf.reduce_sum(
            tf.square(self.mu-self.mean_table), 1)
        self.mse2 = tf.reduce_mean(mse2)

        '''total loss'''
        self.loss = self.re_mse+self.KL_divergence+self.mse2

        """ Summary """
        re_mse_sum = tf.summary.scalar("re_mse", self.re_mse)
        mse2_sum = tf.summary.scalar("mse2", self.mse2)
        kl_sum = tf.summary.scalar("kl", self.KL_divergence)
        loss_sum = tf.summary.scalar("loss", self.loss)

        # final summary operations
        self.merged_summary_op = tf.summary.merge_all()

    def train(self):

        self.build_model()

        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(
            self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load_ckp(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        for epoch in range(start_epoch, self.epoch):
            mean = self.sess.run(self.mu, feed_dict={
                self.inputs: self.input_data})
            table = self.update_table(mean)

            input_data, table = shuffle_data(self.input_data, table)
            
            for idx in range(start_batch_id, self.num_batches):
                # get batch data
                batch_input_data = input_data[idx *
                                                   self.batch_size:(idx+1)*self.batch_size]
                batch_input_table = table[idx *
                                          self.batch_size:(idx+1)*self.batch_size]

                """ Training """
                # t_vars = tf.trainable_variables()
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    # self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                    #        .minimize(self.loss, var_list=t_vars)
                    self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                            .minimize(self.loss)
            

                # update autoencoder
                _, summary_str, loss, remse, kl_loss, mse_2 = self.sess.run([self.optim, self.merged_summary_op, self.loss, self.re_mse, self.KL_divergence, self.mse2],
                                                                            feed_dict={self.inputs: batch_input_data, self.mean_table: batch_input_table})
                
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] loss: %.8f, re_mse: %.8f, kl: %.8f, mse2: %.8f,"
                      % (epoch, idx, self.num_batches, loss, remse, kl_loss, mse_2))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save_ckp(self.checkpoint_dir, counter)

        # save model for final step
        self.save_ckp(self.checkpoint_dir, counter)

    def predict(self, input_vector):

        # initialize all variables
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(
            self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits

        could_load, checkpoint_counter = self.load_ckp(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        predict_mu, predict_sigma = self.sess.run([self.mu, self.sigma],
                                                  feed_dict={self.inputs: input_vector})

        return predict_mu

    def visualize_results(self, epoch):
        pass

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_path,
            self.batch_size, self.z_dim)

    def save_ckp(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(
            checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(
            checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load_ckp(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(
            checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
