import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm, trange

eps = 0.00001

class SeqNet:
    def __init__(self, sess, seq_len=20, ktop=5, lr=0.0001, gamma=0.8, horizon=20):
        '''
        ktop is size of memory buffer, lr is learning rate, and horizon is number of iterations
        per episode. Gamma and seq_len are self explanatory.
        '''
        self.sess = sess
        self.gamma = gamma
        self.seq_len = seq_len
        self.ktop = ktop
        self.lr = lr
        self.horizon = horizon
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

    def improve(self, seqs, r_ktop, actions, returns):
        '''
        Policy gradient update in both actor and critic
        '''
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
    
    def multi_run(self, seqs_list, rewards_list):
        return self.sess.run(self.sample_output_seq, feed_dict={
            self.sequences: np.array(seqs_list),
            self.on_target_labels: np.array(rewards_list),
        })
    
    def multi_path(self, samples, n=100):
        '''
        Same as path but runs multiple concurrently.
        '''
        samples_set = [samples[:] for _ in range(n)]
        state_set = [[samples_set[i][random.randrange(len(samples_set[i]))] for _ in range(self.ktop)] for i in range(n)]
        visited_set = [state[:] for state in state_set]
        path_set = [[] for _ in state_set]
        for i in range(self.horizon):
            if not any(i for i in samples_set):
                break
            seqs_set = []
            rewards_set = []
            for state in state_set:
                seqs, rewards = [np.array(i) for i in zip(*state)]
                seqs_set.append(seqs)
                rewards_set.append(rewards)
            action_set = self.multi_run([self.flat(x) for x in seqs_set], rewards_set)
            new_seq_set = []
            reward_set = []
            for k, sample in enumerate(samples_set):
                d = lambda x: np.linalg.norm((x[0] - action_set[k]).flatten(), ord=1) // 2
                idx = min(range(len(sample)), key=lambda x: d(sample[x]))
                new_seq, reward = sample[idx]
                new_seq_set.append(new_seq)
                reward_set.append(reward)
                sample[idx] = (new_seq, 0)

            for i, visited in enumerate(visited_set):
                visited.append((new_seq_set[i], reward_set[i]))
            for i, path in enumerate(path_set):
                path.append((state_set[i], action_set[i], reward_set[i] *\
                             (1 - d((new_seq_set[i], reward_set[i])) / self.seq_len)))
            for i, _ in enumerate(state_set):
                state_set[i] = visited_set[i][-self.ktop:]#, key=lambda x: -x[1])[:self.ktop]
        return [zip(*path) for path in path_set]


    def path(self, samples):
        '''
        Get states, actions, and rewards of trajectory in given sample space. Rewards are scaled down
        based on L1 distance to valid sequence.
        '''
        return self.multi_path(samples, 1)[0]

    
    def train(self, samples, batch):
        '''
        Trains on samples (list of one-hot dna strand, on-target rate tuples) and returns avg reward.
        Generates batch trajectories concurrently.
        '''
        results = []
        seqs_set = []
        labels_set = []
        actions_set = []
        rewards_set = []
        for path in self.multi_path(samples, batch):
            states, actions, rewards = path
            for k, i, j in list(zip(states, actions, rewards)):
                results.append(j)
            seqs = np.array([[v[0] for v in x] for x in states])
            labels = np.array([[v[1] for v in x] for x in states])
            seqs = np.reshape(seqs, [seqs.shape[0], 
                                           seqs.shape[1] * seqs.shape[2], 4]);
            for seq in seqs:
                seqs_set.append(seq)
            for label in labels:
                labels_set.append(label)
            for action in actions:
                actions_set.append(action)
            returns = [sum((self.gamma ** (i - j)) * rewards[i]
                       for i in range(j, len(rewards)))
                           for j, _ in enumerate(rewards)]
            for reward in returns:
                rewards_set.append(reward)
        self.improve(seqs_set, labels_set, actions_set, rewards_set)
        return np.mean(np.array(results))
    
    def single_train(self, samples, iterations):
        '''
        Like above, but repeatedly trains on single trajectory
        '''
        results = []
        for i in range(iterations):
            states, actions, rewards = self.path(samples)
            for k, i, j in list(zip(states, actions, rewards)):
                results.append(j)
            seqs = np.array([[v[0] for v in x] for x in states])
            labels = np.array([[v[1] for v in x] for x in states])
            returns = [sum((self.gamma ** (i - j)) * rewards[i]
                       for i in range(j, len(rewards)))
                           for j, _ in enumerate(rewards)]
            self.improve(np.reshape(seqs, [seqs.shape[0], 
                                           seqs.shape[1] * seqs.shape[2], 4]), 
                                             labels, actions, returns)
        return np.mean(np.array(results))
            
    def evaluate(self, samples, batch):
        '''
        Runs on samples and returns avg reward
        '''
        results = []
        seqs_set = []
        labels_set = []
        actions_set = []
        rewards_set = []
        for path in self.multi_path(samples, batch):
            states, actions, rewards = path
            for k, i, j in list(zip(states, actions, rewards)):
                results.append(j)
            seqs = np.array([[v[0] for v in x] for x in states])
            labels = np.array([[v[1] for v in x] for x in states])
            seqs = np.reshape(seqs, [seqs.shape[0], 
                                           seqs.shape[1] * seqs.shape[2], 4]);
            for seq in seqs:
                seqs_set.append(seq)
            for label in labels:
                labels_set.append(label)
            for action in actions:
                actions_set.append(action)
            returns = [sum((self.gamma ** (i - j)) * rewards[i]
                       for i in range(j, len(rewards)))
                           for j, _ in enumerate(rewards)]
            for reward in returns:
                rewards_set.append(reward)
        return np.mean(np.array(results))
    
    
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
