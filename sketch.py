
# coding: utf-8

# In[ ]:

from time import time

import tensorflow as tf

from wavenet.utils import make_batch, mu_law_bins
from wavenet.models import Model, Generator

from IPython.display import Audio

import numpy as np

inputs, targets = make_batch('assets/voice.wav')
num_time_samples = inputs.shape[1]
num_channels = 1
gpu_fraction = 1.0

model = Model(num_time_samples=num_time_samples,
              num_channels=num_channels,
              gpu_fraction=gpu_fraction)

Audio(inputs.reshape(inputs.shape[1]), rate=44100)


# In[ ]:

tic = time()
model.train(inputs, targets)
toc = time()

# save the variable to disk
save_path = model.saver.save(model.sess, "/tmp/model.ckpt")
print("Model saved in file: %s" % save_path)

print('Training took {} seconds.'.format(toc-tic))


# In[ ]:

# Restore variables from disk.
model.saver.restore(model.sess, "/tmp/model.ckpt")

generator = Generator(model)

# Get first sample of input
input_ = inputs[:, 0:1, 0]

tic = time()
predictions = generator.run(input_, 32000)
toc = time()
print('Generating took {} seconds.'.format(toc-tic))


# In[ ]:

generator = Generator(model)

# Get first sample of input
input_ = inputs[:, 0:1, 0]

check_train = model.sess.run(model.outputs, feed_dict={model.inputs:inputs, model.targets:targets})
#check_test = generator.model.sess.run(generator.outputs, feed_dict={generator.inputs:input_})


# In[ ]:

## generate with longer sequence given

# Restore variables from disk.
model.saver.restore(model.sess, "/tmp/model.ckpt")

generator = Generator(model)

# flush the states
for t in range(10000):
    check_test = generator.model.sess.run([generator.out_ops], feed_dict={generator.inputs:inputs[:, t:t+1, 0]})

# start generating
tic = time()
predictions = generator.run(inputs[:, t+1:t+2, 0], 16000)
toc = time()
print('Generating took {} seconds.'.format(toc-tic))

# output_dist = generator.model.sess.run(generator.out_ops, feed_dict={generator.inputs: inputs[:, t+1:t+2, 0]})[0][0, :]


# In[ ]:

Audio(predictions, rate=44100)


# In[ ]:

inputs = np.random.choice(generator.bins, 1, p=list(output_dist)/sum(list(output_dist)))[None]
print(inputs)


# In[ ]:

print(generator.bins[2][None, None])

