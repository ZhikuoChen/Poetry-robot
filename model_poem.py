import tensorflow as tf
class lstm(object):
    def __init__(self,vocab_size,lstm_size, num_layers,batch_size,learning_rate):
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.lstm_size = lstm_size  #128
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.build_graph()
    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.LSTM_layer_op()
        self.loss_op()
    def add_placeholders(self):
        self.input_data = tf.placeholder(tf.int32, [None, None])
        self.output_targets = tf.placeholder(tf.int32, [None, None])
    def lookup_layer_op(self):
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [self.vocab_size, self.lstm_size], -1.0, 1.0))  
        #inputs的shape为[batch_size,max_length,lstm_size]
        self.inputs = tf.nn.embedding_lookup(embedding, self.input_data)

    def LSTM_layer_op(self):
        """
        :param lstm_inputs: [batch_size, max_seq_len, embedding_size] 
        :return: [batch_size, max_seq_len, lstm_size] 
        """
        cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_size, state_is_tuple=True)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        #outputs的shape为[batch_size, max_length, lstm_size] = [64, ?, 128]
        outputs, self.last_state = tf.nn.dynamic_rnn(cell, self.inputs,
                                                initial_state=self.initial_state)
        outputs = tf.reshape(outputs, [-1, self.lstm_size])
        weights = tf.Variable(tf.truncated_normal([self.lstm_size, self.vocab_size]))
        bias = tf.Variable(tf.zeros(shape=[self.vocab_size]))
        self.logits = tf.nn.bias_add(tf.matmul(outputs, weights), bias=bias)
        # [batch_size*max_length, vocab_size+1]
      
    def loss_op(self):
          # output_data must be one-hot encode
          labels = tf.one_hot(tf.reshape(self.output_targets, [-1]), depth=self.vocab_size)
          # should be [batch_size*max_length*vocab_size]

          loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=self.logits)
          # loss shape should be [batch_size*max_length, vocab_size]
          self.total_loss = tf.reduce_mean(loss)
          self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
          self.prediction = tf.nn.softmax(self.logits)
