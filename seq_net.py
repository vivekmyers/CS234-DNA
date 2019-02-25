import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm, trange

eps = 0.00001

class SeqNet:
    def __init__(self, seq_len=20, ktop=5, lr=0.0001, gamma=0.8, horizon=20):
        '''
        ktop is size of memory buffer, lr is learning rate, and horizon is number of iterations
        per episode. Gamma and seq_len are self explanatory.
        '''
        self.gamma = gamma
        self.seq_len = seq_len
        self.ktop = ktop
        self.lr = lr
        self.horizon = horizon
        self.sess = tf.Session()
        self.build_placeholders()
        self.build_network()
        self.make_train_ops()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter('results', self.sess.graph)

    def build_placeholders(self):
        self.actions = tf.placeholder(tf.float32, shape=[None, self.seq_len, 4], name='actions')
        self.sequences = tf.placeholder(tf.float32, shape=[None, self.seq_len * self.ktop, 4], name='sequences')
        self.returns = tf.placeholder(tf.float32, shape=[None], name='returns')
        self.on_target_labels = tf.placeholder(tf.float32, shape=[None, self.ktop], name='on_target_labels')
        self.static_advantages = tf.placeholder(tf.float32, shape=[None], name='static_advantages')

    def build_network(self):
        self.conv1 = tf.layers.conv1d(inputs=self.sequences, filters=80,
                                      kernel_size=7, activation=tf.nn.relu)
        self.conv2 = tf.layers.conv1d(inputs=self.conv1, filters=80,
                                      kernel_size=7, activation=tf.nn.relu)
        self.conv3 = tf.layers.conv1d(inputs=self.conv2, filters=80,
                                      kernel_size=1, activation=tf.nn.relu)
        self.merged = tf.concat([tf.layers.flatten(self.conv3),
                                 self.on_target_labels], axis=1)
        self.dense1_actor = tf.layers.dense(inputs=self.merged, units=512,
                                            activation=tf.nn.relu)
        self.dense2_actor = tf.layers.dense(inputs=self.dense1_actor, units=self.seq_len * 4)
        self.dense1_critic = tf.layers.dense(inputs=self.merged, units=512,
                                             activation=tf.nn.relu)
        self.dense2_critic = tf.layers.dense(self.dense1_critic, units=1)

        self.output_seq = tf.reshape(self.dense2_actor, [-1, self.seq_len, 4])
        self.sample_output_seq = tf.reshape(tf.one_hot(
            tf.reshape(
                tf.multinomial(tf.reshape(self.output_seq, [-1, 4]), 1),
                shape=[-1, self.seq_len]),
            depth=4), [-1, self.seq_len, 4])
        self.logprob = tf.reduce_sum(
            -tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.actions, logits=self.output_seq), axis=1)
        self.baseline = tf.squeeze(self.dense2_critic, axis=1)
        self.advantage = self.returns - self.baseline

    def make_train_ops(self):
        self.actor_loss = tf.reduce_mean(-self.logprob * self.static_advantages)
        self.train_actor = tf.train.AdamOptimizer(self.lr).minimize(self.actor_loss)
        self.critic_loss = tf.reduce_mean(tf.squared_difference(self.returns, self.baseline))
        self.train_critic = tf.train.AdamOptimizer(self.lr).minimize(self.critic_loss)

    def get_advantages(self, seqs, r_ktops, actions, returns):
        return self.sess.run(self.advantage, feed_dict={
            self.actions: list(actions),
            self.sequences: list(seqs),
            self.returns: list(returns),
            self.on_target_labels: list(r_ktops),
        })

    def improve(self, seqs, r_ktop, actions, rewards):
        '''
        Policy gradient update in both actor and critic
        '''
        returns = [sum((self.gamma ** (i - j)) * rewards[i]
                       for i in range(j, len(rewards)))
                   for j, _ in enumerate(rewards)]
        adv = self.get_advantages(seqs, r_ktop, actions, returns)
        adv = np.array(adv)
        adv = (adv - np.mean(adv)) / (np.std(adv) + eps)
        cl, _ = self.sess.run([self.critic_loss, self.train_critic], feed_dict={
            self.actions: actions,
            self.sequences: seqs,
            self.returns: returns,
            self.on_target_labels: r_ktop,
        })
        al, _ = self.sess.run([self.actor_loss, self.train_actor], feed_dict={
            self.static_advantages: adv,
            self.actions: actions,
            self.sequences: seqs,
            self.returns: returns,
            self.on_target_labels: r_ktop,
        })

    def run(self, seqs, rewards):
        return np.squeeze(self.sess.run(self.sample_output_seq, feed_dict={
            self.sequences: np.array([seqs]),
            self.on_target_labels: np.array([rewards]),
        }))
    
    def flat(self, state):
        return np.reshape(state, [state.shape[0] * state.shape[1], 4])

    def path(self, samples):
        '''
        Get states, actions, and rewards of trajectory in given sample space. Rewards are scaled down
        based on L1 distance to valid sequence.
        '''
        samples = samples[:]
        state = random.sample(samples, self.ktop)
        for x in state:
            for i, v in enumerate(samples):
                if (v[0] == x[0]).all():
                    del samples[i]
                    break
        visited = state[:]
        path = []
        for i in range(self.horizon):
            if not samples:
                break
            seqs, rewards = [np.array(i) for i in zip(*state)]
            action = self.run(self.flat(seqs), rewards)
            d = lambda x: sum((x[0][i] != action[i]).any() for i, _ in enumerate(x[0]))
            new_seq, reward = min(samples, key=d)
            for i, v in enumerate(samples):
                if (v[0] == new_seq).all():
                    del samples[i]
                    break
            visited.append((new_seq, reward))
            path.append((state, action, reward * (1 - d((new_seq, reward)) / self.seq_len)))
            state = sorted(visited, key=lambda x: -x[1])[:self.ktop]
        return zip(*path)

    def train(self, samples, iterations):
        '''
        Trains on samples (list of one-hot dna strand, on-target rate tuples) and returns avg reward
        '''
        results = []
        for i in range(iterations):
            states, actions, rewards = self.path(samples)
            for k, i, j in list(zip(states, actions, rewards)):
                results.append(j)
            seqs = np.array([[v[0] for v in x] for x in states])
            labels = np.array([[v[1] for v in x] for x in states])
            self.improve(np.reshape(seqs, [seqs.shape[0], 
                                           seqs.shape[1] * seqs.shape[2], 4]), 
                                             labels, actions, rewards)
        return np.mean(np.array(results))
            
    def evaluate(self, samples, iterations):
        '''
        Runs on samples and returns avg reward
        '''
        rewards = []
        for i in range(iterations):
            s, a, r = net.path(samples)
            for k, i, j in list(zip(s, a, r)):
                rewards.append(j)
        return np.mean(np.array(rewards))
    
    
def dna_vec(s):
    '''
    Convert DNA to one-hot vector.
    '''
    mask = np.array(['ATCG'.index(i) for i in s])
    arr = np.zeros([len(s), 4])
    arr[np.arange(len(s)), mask] = 1
    return arr

def vec_dna(v):
    '''
    Convert DNA vector back to string.
    '''
    return ''.join(['ATCG'[np.argmax(i)] for i in v])


