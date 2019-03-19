import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm, trange


class WolpertingerNet:
    def __init__(self, sess, seq_len=20, ktop=3, kmem=2, lr=0.0001, gamma=0.8, horizon=20, knn=5, decay=2048, replay=128, penalty=1, topn=0, reinforce=True):
        '''
        ktop is size of memory buffer, lr is learning rate, and horizon is number of iterations
        per episode. Gamma and seq_len are self explanatory.
        '''
        self.sess = sess
        self.reinforce = reinforce
        self.topn = topn
        self.penalty = penalty
        self.decay = decay
        self.knn = knn
        self.replay = replay
        self.buffer = []
        self.gamma = gamma
        self.seq_len = seq_len
        self.ktop = ktop
        self.kmem = kmem
        self.itr = tf.Variable(0, dtype=tf.float32)
        self.eps = 1 / (1 + (self.itr / self.decay))
        self.itr_update_op = tf.assign_add(self.itr, 1)
        self.lr = lr
        self.horizon = horizon
        self.build_placeholders()
        self.build_network()
        self.make_train_ops()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter('results', self.sess.graph)

    def build_placeholders(self):
        self.actions = tf.placeholder(tf.float32, shape=[None, self.seq_len, 4], name='actions')
        self.sequences = tf.placeholder(tf.float32, shape=[None, self.seq_len * (self.ktop + self.kmem), 4], name='sequences')
        self.returns = tf.placeholder(tf.float32, shape=[None], name='returns')
        self.on_target_labels = tf.placeholder(tf.float32, shape=[None, (self.ktop + self.kmem)], name='rates')
        self.q_computed = tf.placeholder(tf.float32, shape=[None], name='q_computed')

    def build_network(self):
        self.conv1 = tf.layers.conv1d(inputs=self.sequences, filters=80,
                                      kernel_size=7, activation=tf.nn.relu, name='conv1')
        self.conv2 = tf.layers.conv1d(inputs=self.conv1, filters=80,
                                      kernel_size=7, activation=tf.nn.relu, name='conv2')
        self.conv3 = tf.layers.conv1d(inputs=self.conv2, filters=80,
                                      kernel_size=1, activation=tf.nn.relu, name='conv3')
        self.merged = tf.layers.dense(inputs=tf.concat([tf.layers.flatten(self.conv3),
                                                         self.on_target_labels], axis=1, name='merged'),
                                              units=512, activation=tf.nn.relu, name='m_dense')
        self.dense1_actor = tf.layers.dense(inputs=self.merged, units=512,
                                            activation=tf.nn.relu, name='a_dense1')
        self.dense2_actor = tf.layers.dense(inputs=self.dense1_actor, units=self.seq_len * 4, name='a_dense2')
        
        self.critic_conv1 = tf.layers.conv1d(inputs=self.actions, filters=80,
                                      kernel_size=7, activation=tf.nn.relu, name='c_conv1')
        self.critic_conv2 = tf.layers.conv1d(inputs=self.critic_conv1, filters=80,
                                      kernel_size=7, activation=tf.nn.relu, name='c_conv2')
        self.critic_conv3 = tf.layers.conv1d(inputs=self.critic_conv2, filters=80,
                                      kernel_size=1, activation=tf.nn.relu, name='c_conv3')
        self.dense1_critic = tf.layers.dense(inputs=tf.concat([self.merged, 
                                               tf.layers.flatten(self.critic_conv3)], axis=1), units=512,
                                               activation=tf.nn.relu, name='c_dense1')
        self.dense2_critic = tf.layers.dense(self.dense1_critic, units=1, name='c_dense2')

        self.output_seq = tf.reshape(self.dense2_actor, [-1, self.seq_len, 4], name='proto_action')
        self.sample_output_seq = tf.reshape(tf.one_hot(
            tf.reshape(
                tf.multinomial(tf.reshape(self.output_seq, [-1, 4]), 1),
                shape=[-1, self.seq_len]),
            depth=4), [-1, self.seq_len, 4], name = 'sampled_proto')
        self.logprob = tf.reduce_sum(
            -tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.actions, logits=self.output_seq, name='cross_entropy'), axis=1, name='logprob')
        self.q_pred = tf.squeeze(self.dense2_critic, axis=1, name='q_pred')

    def make_train_ops(self):
        self.actor_loss = tf.reduce_mean(-self.logprob * self.q_computed, name='actor_loss')
        self.train_actor = tf.train.AdamOptimizer(self.lr).minimize(self.actor_loss, name='actor_train_op')
        self.critic_loss = tf.reduce_mean(tf.squared_difference(self.returns, self.q_pred, name='mse'), name='critic_loss')
        self.train_critic = tf.train.AdamOptimizer(self.lr).minimize(self.critic_loss, name='critic_train_op')


    def improve_actor(self, seqs, r_ktop, actions, new_seqs, returns):
        '''
        Policy gradient update in actor
        '''
        q_computed = self.sess.run(self.q_pred, feed_dict={
            self.actions: actions,
            self.sequences: seqs,
            self.on_target_labels: r_ktop,
        })
        q_computed = np.array(returns) if self.reinforce else np.array(q_computed)
        q_computed = (q_computed - q_computed.mean()) / (1 + q_computed.std())
        actor_loss, _ = self.sess.run([self.actor_loss, self.train_actor], feed_dict={
            self.actions: actions,
            self.sequences: seqs,
            self.on_target_labels: r_ktop,
            self.q_computed: q_computed
        })
        return actor_loss
    
    def improve_critic(self, seqs, r_ktop, actions, new_seqs, returns):
        '''
        Refit Q function
        '''
        critic_loss, q_computed, _ = self.sess.run([self.critic_loss, self.q_pred, self.train_critic], feed_dict={
            self.actions: new_seqs,
            self.sequences: seqs,
            self.returns: returns,
            self.on_target_labels: r_ktop,
        })
        return critic_loss
        
    def run(self, seqs, rewards):
        return np.squeeze(self.sess.run(self.sample_output_seq, feed_dict={
            self.sequences: np.array([seqs]),
            self.on_target_labels: np.array([rewards]),
        }))
    
    def take_max(self, seqs, rewards, actions):
        '''
        Argmax Q over actions given prior state (seqs, reward).
        '''
        vals = self.sess.run(self.q_pred, feed_dict={
            self.sequences: [seqs for a in actions],
            self.on_target_labels: [rewards for a in actions],
            self.actions: actions,
        })
        if random.random() < self.sess.run(self.eps):
            return random.choice(range(len(vals)))
        else:
            return np.argmax(vals)
        
    def getQ(self, seqs, rewards, action):
        vals = self.sess.run(self.q_pred, feed_dict={
            self.sequences: [seqs],
            self.on_target_labels: [rewards],
            self.actions: [action],
        })
        return vals[0]
    
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
        topn_set = []
        for sample in samples_set:
            topn_set.append([vec_dna(a) for a, b in sorted(sample, key=lambda x: -x[1])[:self.topn]])
        state_set = [[samples_set[i][random.randrange(len(samples_set[i]))] for _ in range(self.ktop + self.kmem)] for i in range(n)]
        visited_set = [state[:] for state in state_set]
        for i, _ in enumerate(state_set):
            state_set[i] = list(sorted(visited_set[i], key=lambda x: -x[1])[:self.ktop]) + visited_set[i][-self.kmem:]
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
                idx = list(sorted(range(len(sample)), key=lambda x: d(sample[x])))[:self.knn]
                proposed_samples = np.array(sample)[idx]
                act_idx = self.take_max(self.flat(seqs_set[k]), rewards_set[k], [i[0] for i in proposed_samples])
                new_seq, reward = sample[idx[act_idx]]
                new_seq_set.append(new_seq)
                reward_set.append(reward)
                sample[idx[act_idx]] = (new_seq, reward * self.penalty)

            for i, visited in enumerate(visited_set):
                visited.append((new_seq_set[i], reward_set[i]))
            for i, path in enumerate(path_set):
                r = reward_set[i]
                seq = vec_dna(new_seq_set[i])
                if self.topn and seq not in topn_set[i]:
                    r = 0
                path.append((state_set[i], action_set[i], new_seq_set[i], r))
            for i, _ in enumerate(state_set):
                state_set[i] = list(sorted(visited_set[i], key=lambda x: -x[1])[:self.ktop]) + visited_set[i][-self.kmem:]
        return [zip(*path) for path in path_set]


    def path(self, samples):
        '''
        Get states, actions, and rewards of trajectory in given sample space. Rewards are scaled down
        based on L1 distance to valid sequence.
        '''
        return self.multi_path(samples, 1)[0]

    
    def train(self, samples, batch, replay):
        '''
        Trains on samples (list of one-hot dna strand, on-target rate tuples) and returns avg reward.
        Generates batch trajectories concurrently.
        '''
        results = []
        seqs_set = []
        labels_set = []
        actions_set = []
        rewards_set = []
        new_seqs_set = []
        for path in self.multi_path(samples, batch):
            states, actions, new_seqs, rewards = path
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
            for new_seq in new_seqs:
                new_seqs_set.append(new_seq)
            returns = [sum((self.gamma ** (i - j)) * rewards[i]
                       for i in range(j, len(rewards)))
                           for j, _ in enumerate(rewards)]
            for reward in returns:
                rewards_set.append(reward)
        self.buffer.append((seqs_set, labels_set, actions_set, new_seqs_set, rewards_set))
        self.buffer = self.buffer[:self.replay]
        self.sess.run(self.itr_update_op)
        critic_loss = []
        for data in random.sample(self.buffer, min([replay, len(self.buffer)])):
            c_loss = self.improve_critic(*data)
            critic_loss.append(c_loss)
        actor_loss = [self.improve_actor(seqs_set, labels_set, actions_set, new_seqs_set, rewards_set)]
        self.last_loss = (np.array(actor_loss).mean(), np.array(critic_loss).mean())
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
            states, actions, new_seqs, rewards = path
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
    mask = np.array(['ATCG'.index(i) for i in s.upper()])
    arr = np.zeros([len(s), 4])
    arr[np.arange(len(s)), mask] = 1
    return arr


def vec_dna(v):
    '''
    Convert DNA vector back to string.
    '''
    return ''.join(['ATCG'[np.argmax(i)] for i in v])
