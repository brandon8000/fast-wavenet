import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from layers import (_causal_linear, _output_linear, conv1d,
                    dilated_conv1d, post_processing, dilated_generation, post_processing_generation)
from utils import mu_law_bins


class Model(object):
    def __init__(self,
                 num_time_samples,
                 num_channels=1,
                 num_classes=256,
                 num_blocks=1,
                 num_layers=14,
                 num_post_layers=2,
                 num_hidden=128,
                 num_gated=128,
                 num_skip=128,
                 gpu_fraction=0.5):
        
        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_post_layers = num_post_layers
        self.num_hidden = num_hidden
        self.num_gated = num_gated
        self.skip = num_skip
        self.gpu_fraction = gpu_fraction
        
        inputs = tf.placeholder(tf.float32,
                                shape=(None, num_time_samples, num_channels))
        targets = tf.placeholder(tf.int32, shape=(None, num_time_samples))

        h = inputs
        hs = []
        skips = []
        tests = [];
        
        # dilated conv
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                h, skip, test = dilated_conv1d(h, num_gated, num_hidden, num_skip, rate=rate, name=name)
                # write summary to tensorboard
                tf.histogram_summary(name+'h', h)
                hs.append(h)
                skips.append(skip)
                tests.append(test)
        
        # post processing
        outputs = post_processing(skips, num_post_layers, num_classes)
        # write summary to tensorboard
        tf.histogram_summary('outputs', outputs)

        costs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            outputs, targets)
        cost = tf.reduce_mean(costs)
        
        # write summary to tensorboard
        tf.scalar_summary('cost', cost)
        
        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.initialize_all_variables())
        
        # Merge all the summaries and write them
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter('logs/train', sess.graph)

        self.inputs = inputs
        self.targets = targets
        self.outputs = outputs
        self.hs = hs
        self.tests = tests
        self.costs = costs
        self.cost = cost
        self.train_step = train_step
        self.merged = merged
        self.train_writer = train_writer
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()
        self.sess = sess

    def _train(self, inputs, targets):
        feed_dict = {self.inputs: inputs, self.targets: targets}
        
        # test = self.sess.run(self.test, feed_dict=feed_dict)
        
        cost, summary, _ = self.sess.run(
            [self.cost, self.merged, self.train_step],
            feed_dict=feed_dict)
        
        return cost, summary

    def train(self, inputs, targets):
        losses = []
        terminal = False
        i = 0
        while not terminal:
            i += 1
            cost, summary = self._train(inputs, targets)
            self.train_writer.add_summary(summary, i)
            if cost < 1e-1:
                terminal = True
            losses.append(cost)
            if i % 50 == 0:
                plt.plot(losses)
                plt.show()
                


class Generator(object):
    def __init__(self, model, batch_size=1, input_size=1):
        self.model = model
        # self.bins = np.linspace(-1, 1, self.model.num_classes)
        _, self.bins = mu_law_bins(self.model.num_classes)

        inputs = tf.placeholder(tf.float32, [batch_size, input_size],
                                name='inputs')

        print('Make Generator.')

        count = 0
        h = inputs

        init_ops = []
        push_ops = []
        skips = []
        hs = []
        tests = []
        for b in range(self.model.num_blocks):
            for i in range(self.model.num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                if count == 0:
                    state_size = 1
                else:
                    state_size = self.model.num_hidden
                    
                q = tf.FIFOQueue(rate,
                                 dtypes=tf.float32,
                                 shapes=(batch_size, state_size))
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
        
        # Initialize queues.
        self.model.sess.run(self.init_ops)

    def run(self, input, num_samples):
        predictions = []
        for step in range(num_samples):

            feed_dict = {self.inputs: input}
            
            # draw the max
            # output_dist = self.model.sess.run(self.out_ops, feed_dict=feed_dict)[0] # ignore push ops
            # value = np.argmax(output_dist[0, :])
            # input = np.array(self.bins[value])[None, None]
            
            # draw from the distribution
            output_dist = self.model.sess.run(self.out_ops, feed_dict=feed_dict)[0][0, :] # ignore push ops
            #print('output distribution:{}'.format(output_dist))
            #plt.plot(output_dist)
            #plt.show()
            input = np.random.choice(self.bins, self.batch_size, p=list(output_dist)/sum(list(output_dist)))[None]
                        
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
