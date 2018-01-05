import numpy as np
import tensorflow as tf
from tools import api
import os

# models
class models:

    def __init__(self):
        # common
        self.batch_size = 1
        self.checkpoint = 5000

        # lstm
        self.lstm_units = 30
        self.lstm_input_max_len = 5
        self.lstm_input_size = 4
        self.lstm_target_size = self.lstm_input_size
        self.lstm_output_size = 2 * self.lstm_units
        self.lstm_num_layers = 5
        self.lstm_num_iter = 50
        self.lstm_num_report = 500

        pass

    def save_ckpt(self):
        pass

    # return (inps, targs) [batch_size, input_max_len, inp_size]
    # generator
    def get_data(self, num_iter, length):
        # TODO:
        # test = np.random.randint(0, 20, (self.batch_size, self.lstm_input_max_len, self.lstm_input_size))
        # return test, test
        return api.gen_data([1, 2, 3, 4], length, num_iter)

    # runs = [opt, loss, preds, merge]
    def train_loop(self, inps, targs, runs, num_iter, num_report):
        # TODO:
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        feed_dict = dict()

        train_writer = tf.summary.FileWriter(os.getcwd() + '/train', sess.graph)
        for i, (x, y) in enumerate(self.get_data(num_iter, self.lstm_input_max_len)):
            # ignore date now
            x = np.delete(x, [0], axis=1)
            y = np.delete(y, [0], axis=1)
            # TODO: reshape... batch = 1
            x = np.reshape(x, (1, ) + x.shape)
            y = np.reshape(y, (1, ) + y.shape)

            feed_dict[inps], feed_dict[targs] = x, y
            res = sess.run(runs, feed_dict=feed_dict) # tuple
            # add summary
            train_writer.add_summary(res[3], i)
            if i != 0 and i % num_report == 0:
                print('Opt: {}'.format(res[0]))
                print('Loss: {}'.format(res[1]))
                print('Pred: {}'.format(res[2]))
                print('Inp : {}'.format(feed_dict[inps]))
                print('Targ: {}'.format(feed_dict[targs]))
            if i != 0 and i % self.checkpoint == 0:
                print('Saving checkpoint...')
                self.save_ckpt()

    def var_summary(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    # try with continuous data (not 'inc', 'dec', 'balanced', etc)
    def stack_bidir_lstm_model(self):

        # lstm_input_size = [# of transformations (rdiff, etc.)]
        tf.reset_default_graph()

        f_cell = [tf.nn.rnn_cell.BasicLSTMCell(self.lstm_units, state_is_tuple=True) for i in range(self.lstm_num_layers)]
        b_cell = [tf.nn.rnn_cell.BasicLSTMCell(self.lstm_units, state_is_tuple=True) for i in range(self.lstm_num_layers)]

        inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.lstm_input_max_len, self.lstm_input_size))
        targets = tf.placeholder(tf.float32, shape=(self.batch_size, self.lstm_input_max_len, self.lstm_target_size))

        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            f_cell,
            b_cell,
            inputs,
            dtype=tf.float32
        )
        stddev = np.sqrt(2.0 / (self.batch_size * self.lstm_output_size * self.lstm_target_size))
        with tf.name_scope('weights'):
            w = tf.Variable(tf.truncated_normal(shape=[self.batch_size, self.lstm_output_size, self.lstm_target_size], stddev=stddev))
            self.var_summary(w)

        stddev2 = np.sqrt(2.0 / (self.lstm_input_max_len * self.lstm_target_size))
        with tf.name_scope('bias'):
            b = tf.Variable(tf.truncated_normal(shape=[self.lstm_input_max_len, self.lstm_target_size], stddev=stddev2))
            self.var_summary(b)

        with tf.name_scope('preds'):
            preds = tf.matmul(outputs, w) + b # [batch_size, input_max_len, target_size]
            self.var_summary(preds)

        loss = tf.reduce_mean(tf.square(preds - targets))
        tf.summary.scalar('loss', loss)

        # apply gradient clipping here? "This is the correct way to perform gradient clipping" lol
        opt = tf.train.AdamOptimizer(1e-2).minimize(loss)

        merge = tf.summary.merge_all()

        self.train_loop(inputs, targets, [opt, loss, preds, merge], self.lstm_num_iter, self.lstm_num_report)

    # TODO: cnn & seq2seq (with attention & news sentiment)

x = models()
x.stack_bidir_lstm_model()
