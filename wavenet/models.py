import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from wavenet.layers import _causal_linear, _output_linear, conv1d, dilated_conv1d, dilated_conv1d_nopad, post_processing, dilated_generation, post_processing_generation
from wavenet.utils import mu_law_bins

class Model(object):

    def __init__(self, 
                 num_input_samples,
                 num_output_samples,
                 num_channels = 1,
                 num_classes = 256, 
                 num_blocks = 1, 
                 num_layers = 14, 
                 num_post_layers = 2, 
                 num_hidden = 128,
                 num_gated = 128, 
                 num_skip = 128,
                 gpu_fraction = 0.5):
        
        self.num_input_samples = num_input_samples
        self.num_output_samples = num_output_samples
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_post_layers = num_post_layers
        self.num_hidden = num_hidden
        self.num_gated = num_gated
        self.skip = num_skip
        self.gpu_fraction = gpu_fraction
        
        inputs = tf.placeholder(tf.float32, shape=(None, num_input_samples, num_channels))
        targets = tf.placeholder(tf.int32, shape=(None, num_output_samples))
        
        h = inputs
        hs = []
        skips = []
        tests = []
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2 ** i
                name = 'b{}-l{}'.format(b, i)
                h, skip, test = dilated_conv1d_nopad(h, num_gated, num_hidden, num_skip, num_output_samples, rate=rate, name=name)
                tf.summary.histogram(name + 'h', h)
                hs.append(h)
                skips.append(skip)
                tests.append(test)

        outputs = post_processing(skips, num_post_layers, num_classes)
        tf.summary.histogram('outputs', outputs)
        
        costs = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, targets)
        cost = tf.reduce_mean(costs)
        tf.summary.scalar('cost', cost)
        
        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        
        self.inputs = inputs
        self.targets = targets
        self.outputs = outputs
        self.hs = hs
        self.skips = skips
        self.tests = tests
        self.costs = costs
        self.cost = cost
        self.train_step = train_step
        self.merged = merged
        self.train_writer = train_writer
        self.saver = tf.train.Saver()
        self.sess = sess

    def _train(self, inputs, targets):
        feed_dict = {self.inputs: inputs,
         self.targets: targets}
        cost, summary, _ = self.sess.run([self.cost, self.merged, self.train_step], feed_dict=feed_dict)
        return (cost, summary)

    def train(self, inputs, targets):
        losses = []
        terminal = False
        i = 0
        while not terminal:
            i += 1
            cost, summary = self._train(inputs, targets)
            self.train_writer.add_summary(summary, i)
            if cost < 0.1:
                terminal = True
            losses.append(cost)
            if i % 50 == 0:
                plt.plot(losses)
                plt.show()


class Generator(object):

    def __init__(self, model, batch_size = 1, input_size = 1):
        self.model = model
        _, self.bins = mu_law_bins(self.model.num_classes)
        inputs = tf.placeholder(tf.float32, [batch_size, input_size], name='inputs')
        print ('Make Generator.\n')
        count = 0
        h = inputs
        init_ops = []
        push_ops = []
        skips = []
        hs = []
        tests = []
        for b in range(self.model.num_blocks):
            for i in range(self.model.num_layers):
                rate = 2 ** i
                name = 'b{}-l{}'.format(b, i)
                if count == 0:
                    state_size = 1
                else:
                    state_size = self.model.num_hidden
                q = tf.FIFOQueue(rate, dtypes=tf.float32, shapes=(batch_size, state_size))
                init = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))
                state_ = q.dequeue()
                push = q.enqueue([h])
                init_ops.append(init)
                push_ops.append(push)
                h, skip, test = dilated_generation(h, state_, name=name)
                skips.append(skip)
                hs.append(h)
                tests.append(test)
                count += 1

        outputs = post_processing_generation(skips, self.model.num_post_layers, self.model.num_classes)
        outputs_softmax = tf.nn.softmax(outputs)
        out_ops = [outputs_softmax]
        out_ops.extend(push_ops)
        self.inputs = inputs
        self.init_ops = init_ops
        self.out_ops = out_ops
        self.push_ops = push_ops
        self.outputs = outputs
        self.outputs_softmax = outputs_softmax
        self.hs = hs
        self.batch_size = batch_size
        self.tests = tests
        self.model.sess.run(self.init_ops)

    def run(self, input, num_samples):
        predictions = []
        for step in range(num_samples):
            feed_dict = {self.inputs: input}
            output_dist = self.model.sess.run(self.out_ops, feed_dict=feed_dict)[0]
            value = np.argmax(output_dist[0, :])
            input = np.array(self.bins[value])[(None, None)]
            predictions.append(input)
            if step % 1000 == 0:
                predictions_ = np.concatenate(predictions, axis=1)
                plt.plot(predictions_[0, :], label='pred')
                plt.legend()
                plt.xlabel('samples from start')
                plt.ylabel('signal')
                plt.show()

        predictions_ = np.concatenate(predictions, axis=1)
        return predictions_
