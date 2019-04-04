import numpy as np
from model.vae import *
from model.model_utils import *
import tensorflow as tf
import os

tf.app.flags.DEFINE_integer('epoch', 20, 'epoch num')

tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')

tf.app.flags.DEFINE_integer('n_hidden', 1024, 'dim of hidden')

tf.app.flags.DEFINE_integer('z_dim', 512, 'dim of z')

tf.app.flags.DEFINE_float('keep_prob', 0.9, 'dropout keep_prob')

tf.app.flags.DEFINE_float('learn_rate', 0.00001, 'learn rate')

tf.app.flags.DEFINE_float('beta1', 0.5, 'beta1 for AdamOptimizer')

tf.app.flags.DEFINE_float('b', 0.04, 'b')

tf.app.flags.DEFINE_float('k', 1., 'k')

tf.app.flags.DEFINE_string('dataset_path', './data/d.npz',
                           'd/x vector data path')

# store flag
params = tf.app.flags.FLAGS


experiment_dir = '/experiments/'+'z'+str(params.z_dim)+'_h' + str(params.n_hidden)

experiment_dir = os.path.dirname(os.path.abspath(__file__))+experiment_dir

checkpoint_dir = experiment_dir+'/checkpoint'

log_dir = experiment_dir+'/train_log'


print('model/checkpoint/logs will save in {}.'.format(experiment_dir))

with tf.Session() as sess:
    test = VAE(
        sess=sess,
        epoch=params.epoch,
        batch_size=params.batch_size,
        z_dim=params.z_dim,
        dataset_path=params.dataset_path,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        n_hidden=params.n_hidden,
        keep_prob=params.keep_prob,
        learning_rate=params.learn_rate,
        b=params.b,
        mse2_start_e=params.mse2_start_e,
    )
    print('model/checkpoint/logs will save in {}.'.format(experiment_dir))

    test.train()
    print('model / checkpoint / logs will save in {}.'.format(experiment_dir))

    paths = ["./data/voxceleb_combined_200000/xvector",
             "./data/sitw_dev/enroll/xvector",
             "./data/sitw_dev/test/xvector",
             "./data/sitw_eval/enroll/xvector",
             "./data/sitw_eval/test/xvector"
             ]

    for path in paths:
        if os.path.exists(path+'.ark') == True:
            os.remove(path+'.ark')
            print('delete {}.ark'.format(path))

    for path in paths:
        vector, labels = loader(path+'.npz')
        predict_mu = test.predict(vector)
        print(path)
        print(predict_mu.shape)
        get_skew_and_kurt(predict_mu)
        with open(path+'.ark', 'w') as f:
            for i in range(predict_mu.shape[0]):
                f.write(str(labels[i]))
                f.write('  [ ')
                for j in predict_mu[i]:
                    f.write(str(j))
                    f.write(' ')
                f.write(']')
                f.write('\n')
        print('{}.ark is done!'.format(path))

    print('\nall done!')
    print('model/checkpoint/logs will save in {}.'.format(experiment_dir))
