import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm, trange

eps = 0.00001

class SupervisedNet:
    def __init__(self, sess, seq_len=20, lr=0.0001, gamma=0.8, horizon=20):
        '''
        Supervised sequence->rate classifier.
        '''
        self.sess = sess
        self.gamma = gamma
        self.horizon = horizon
        self.seq_len = seq_len
        self.lr = lr
        self.build_placeholders()
        self.build_network()
        self.make_train_ops()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter('results', self.sess.graph)

    def build_placeholders(self):
        self.sequences = tf.placeholder(tf.float32, shape=[None, self.seq_len * (self.ktop + self.kmem), 4], name='sequences')
        self.rewards = tf.placeholder(tf.float32, shape=[None])

    def build_network(self):
        self.conv1 = tf.layers.conv1d(inputs=self.sequences, filters=80,
                                      kernel_size=7, activation=tf.nn.relu)
        self.conv2 = tf.layers.conv1d(inputs=self.conv1, filters=80,
                                      kernel_size=7, activation=tf.nn.relu)
        self.conv3 = tf.layers.conv1d(inputs=self.conv2, filters=80,
                                      kernel_size=1, activation=tf.nn.relu)
        self.merged = tf.layers.dense(inputs=tf.concat([tf.layers.flatten(self.conv3),
                                                         self.on_target_labels], axis=1),
                                         units=512, activation=tf.nn.relu)
        self.dense1_critic = tf.layers.dense(inputs=self.merged, units=512,
                                             activation=tf.nn.relu)
        self.dense2_critic = tf.layers.dense(self.dense1_critic, units=1)

        self.prediction = tf.squeeze(self.dense2_critic, axis=1)

    def make_train_ops(self):
        self.critic_loss = tf.reduce_mean(tf.squared_difference(self.rewards, self.prediction))
        self.train_critic = tf.train.AdamOptimizer(self.lr).minimize(self.critic_loss)


    def train(data, batch):
        tdata = random.sample(data, batch)
        self.sess.run(self.train_critic, feed_dict={
                self.sequences = [i[0] for i in tdata],
                self.rewards = [i[1] for i in tdata]
            })
        labels = predict([i[0] for i in data])
        path = [i[1] for i in sorted(zip(labels, data))][:self.horizon]
        top3 = list(sorted(data, key=lambda x: -x[1]))[:3]
        for i, (s, r) in enumerate(path):
            if vec_dna(s) in [vec_dna(a) for a, b in top3]:
                return i
        
    def evaluate(data, batch):
        labels = predict([i[0] for i in data])
        path = [i[1] for i in sorted(zip(labels, data))][:self.horizon]
        top3 = list(sorted(data, key=lambda x: -x[1]))[:3]
        for i, (s, r) in enumerate(path):
            if vec_dna(s) in [vec_dna(a) for a, b in top3]:
                return i

    def predict(seqs):
        return self.sess.run(self.prediction, feed_dict={self.sequences=seqs})

    
    
def dna_vec(s):
    '''
    Convert DNA to one-hot vector.
    '''
    mask = np.array(['ATCG'.index(i) for i in s.upper()])
    arr = np.zeros([len(s), 4])
    arr[np.arange(len(s)), mask] = 1
    return arr


def vec_dna(v):
    '''
    Convert DNA vector back to string.
    '''
    return ''.join(['ATCG'[np.argmax(i)] for i in v])
