import numpy as np
import tensorflow as tf

# models
class models:

    def __init__(self):
        # common
        self.batch_size = 10

        # lstm
        self.lstm_units = 30
        self.lstm_input_max_len = 8
        self.lstm_input_size = 10
        self.lstm_output_size = 2 * self.lstm_units
        self.lstm_num_layers = 5

        pass

    def generate_data(self):
        # TODO:
        pass

    def train_loop(self):
        # TODO:
        pass

    # try with continuous data (not 'inc', 'dec', 'balanced', etc)
    def stack_bidir_lstm_model(self):

        # lstm_input_size = [# of transformations (rdiff, etc.)]
        tf.reset_default_graph()

        f_cell = [tf.nn.rnn_cell.BasicLSTMCell(self.lstm_units, state_is_tuple=True) for i in range(self.lstm_num_layers)]
        b_cell = [tf.nn.rnn_cell.BasicLSTMCell(self.lstm_units, state_is_tuple=True) for i in range(self.lstm_num_layers)]

        inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.lstm_input_max_len, self.lstm_input_size))
        targets = tf.placeholder(tf.float32, shape=(self.batch_size, self.lstm_input_max_len))

        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            f_cell,
            b_cell,
            inputs,
            dtype=tf.float32
        )
        stddev = np.sqrt(2.0 / (self.batch_size * self.lstm_output_size))
        w = tf.Variable(tf.truncated_normal(shape=[self.batch_size, self.lstm_output_size, 1], stddev=stddev))
        stddev2 = np.sqrt(2.0 / (self.lstm_input_max_len))
        b = tf.Variable(tf.truncated_normal(shape=[self.lstm_input_max_len, 1], stddev=stddev2))

        preds = tf.matmul(outputs, w) + b # [batch_size, input_max_len, 1]
        preds = tf.reshape(preds, shape=[self.batch_size, self.lstm_input_max_len])
        loss = tf.reduce_mean(tf.square(preds - targets))

        # apply gradient clipping here? "This is the correct way to perform gradient clipping" lol
        opt = tf.train.AdamOptimizer(1e-3).minimize(loss)

        self.train_loop()

    # TODO: cnn & seq2seq (with attention & news sentiment)

x = models()
x.stack_bidir_lstm_model()