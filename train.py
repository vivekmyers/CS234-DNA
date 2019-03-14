from tqdm import trange
import tensorflow as tf
import random
import sys
import os
import pickle
from seq_net import *

sess = tf.Session()
net = SeqNet(sess, seq_len=32)

itr = int(sys.argv[1])

# Get data

data = {}
for f in os.listdir('data/genes'):
    data[f[:-2]] = pickle.load(open('data/genes/' + f, 'rb'), encoding='latin1')

labels = [x for x in data]
testing = ['LCE6A','RCOR3','KLHL20','GPR137B','USH2A','SMYD3',
           'CNIH3','CFHR2','WDYHV1','OPN3','TNN',
           'CAMK1G','LENEP','CYB5R1','EPHX1']
training = [x for x in labels if x not in testing]

val = []
tran = []
times = []

# Load old data

val = pickle.load(open('results/validation.p', 'rb'))
tran = pickle.load(open('results/training.p', 'rb'))
times = pickle.load(open('results/times.p', 'rb'))

# Load weights

saver = tf.train.Saver()
saver.restore(net.sess, 'results/model.ckpt')

# Get search time

def get_time_top3():
    test_data = data[random.choice(testing)]

    s, a, r = net.path([(dna_vec(a), b) for a, b in test_data])
    top3 = sorted(test_data, key=lambda x: x[1])[-3:]

    for i, (state, action, reward) in enumerate(zip(s, a, r)):
        best_seen = state[-1]# max(state, key=lambda x: x[1])
        if vec_dna(best_seen[0]) in [x[0] for x in top3]:
            return i
    return 20



# Train for itr iterations

for i in trange(itr):
    gene = random.choice(training)
    samples = [(dna_vec(a), b) for a, b in data[gene]]
    #%lprun -f SeqNet.multi_path net.train(samples, 10)
    tran.append(net.train(samples, 50))
    gene = random.choice(testing)
    samples = [(dna_vec(a), b) for a, b in data[gene]]
    times.append(get_time_top3())
    if i % 2 + 1:
        val.append(net.evaluate(samples, 10))
    else:
        val.append(val[-1])

    if i % 10 and i < itr - 1:
        continue

    # Save training data

    pickle.dump(val, open('results/validation.p', 'wb'))
    pickle.dump(tran, open('results/training.p', 'wb'))
    pickle.dump(times, open('results/times.p', 'wb'))

    # Save weights

    saver = tf.train.Saver()
    saver.save(net.sess, 'results/model.ckpt')
