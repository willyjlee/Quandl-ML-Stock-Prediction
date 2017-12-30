import numpy as np
import tensorflow as tf

# models
class models:

    def __init__(self):
        # common
        self.batch_size = 1
        self.checkpoint = 5000

        # lstm
        self.lstm_units = 30
        self.lstm_input_max_len = 2
        self.lstm_input_size = 2
        self.lstm_target_size = self.lstm_input_size
        self.lstm_output_size = 2 * self.lstm_units
        self.lstm_num_layers = 5
        self.lstm_num_iter = 20000
        self.lstm_num_report = 500

        pass

    def save_ckpt(self):
        pass

    # return (inps, targs) [batch_size, input_max_len, inp_size]
    def generate_data(self):
        # TODO:
        #return np.array([1]), np.array([2])
        test = np.random.randint(0, 20, (self.batch_size, self.lstm_input_max_len, self.lstm_input_size))
        return test, test

    # runs = [opt, loss, preds]
    def train_loop(self, inps, targs, runs, num_iter, num_report):
        # TODO:
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        feed_dict = dict()

        for i in range(num_iter):
            feed_dict[inps], feed_dict[targs] = self.generate_data()
            res = sess.run(runs, feed_dict=feed_dict) # tuple
            if i != 0 and i % num_report == 0:
                print('Opt: {}'.format(res[0]))
                print('Loss: {}'.format(res[1]))
                print('Pred: {}'.format(res[2]))
                print('Inp : {}'.format(feed_dict[inps]))
                print('Targ: {}'.format(feed_dict[targs]))
            if i != 0 and i % self.checkpoint == 0:
                print('Saving checkpoint...')
                self.save_ckpt()

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
        w = tf.Variable(tf.truncated_normal(shape=[self.batch_size, self.lstm_output_size, self.lstm_target_size], stddev=stddev))
        stddev2 = np.sqrt(2.0 / (self.lstm_input_max_len * self.lstm_target_size))
        b = tf.Variable(tf.truncated_normal(shape=[self.lstm_input_max_len, self.lstm_target_size], stddev=stddev2))

        preds = tf.matmul(outputs, w) + b # [batch_size, input_max_len, target_size]
        loss = tf.reduce_mean(tf.square(preds - targets))

        # apply gradient clipping here? "This is the correct way to perform gradient clipping" lol
        opt = tf.train.AdamOptimizer(1e-3).minimize(loss)

        self.train_loop(inputs, targets, [opt, loss, preds], self.lstm_num_iter, self.lstm_num_report)

    # TODO: cnn & seq2seq (with attention & news sentiment)

x = models()
x.stack_bidir_lstm_model()