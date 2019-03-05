from tqdm import trange
import tensorflow as tf
import random
import sys
import os
import pickle
from seq_net import *

sess = tf.Session()
net = SeqNet(sess)

itr = int(sys.argv[1])

# Get data

data = {}
for f in os.listdir('data'):
    data[f[:-2]] = pickle.load(open('data/' + f, 'rb'), encoding='latin1')
    
labels = [x for x in data]
testing = ['NF2']
training = [x for x in labels if x != 'NF2']

val = []
tran = []

# Load old data

val = pickle.load(open('results/validation.p', 'rb'))
tran = pickle.load(open('results/training.p', 'rb'))

# Load weights

saver = tf.train.Saver()
saver.restore(net.sess, 'results/model.ckpt')

# Train for itr iterations

for i in trange(itr):
    gene = random.choice(training)
    samples = [(dna_vec(a), b) for a, b in data[gene]]
    tran.append(net.train(samples, 50))
    gene = random.choice(testing)
    samples = [(dna_vec(a), b) for a, b in data[gene]]
    if i % 2 + 1:
        val.append(net.evaluate(samples, 10))
    else:
        val.append(val[-1])

    if i % 10 and i < itr - 1:
        continue

    # Save training data

    pickle.dump(val, open('results/validation.p', 'wb'))
    pickle.dump(tran, open('results/training.p', 'wb'))

    # Save weights

    saver = tf.train.Saver()
    saver.save(net.sess, 'results/model.ckpt')
