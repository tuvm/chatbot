import tensorflow as tf
import numpy as np
import sys


class Seq2Seq(object):
    """
        Build Seq2seq class
    """
    def __init__(self, xseq_len, yseq_len,
            xvocab_size, yvocab_size,
            emb_dim, num_layers, ckpt_path,
            lr=0.0001,
            epochs=100000, model_name='seq2seq_model'):

        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.ckpt_path = ckpt_path
        self.epochs = epochs
        self.model_name = model_name

        # build thy graph
        #  attach any part of the graph that needs to be exposed, to the self
        def __graph__():

            # placeholders
            tf.reset_default_graph()
            #  encoder inputs : list of indices of length xseq_len
            self.enc_ip = [ tf.placeholder(shape=[None,],
                            dtype=tf.int64,
                            name='ei_{}'.format(t)) for t in range(xseq_len) ]

            #  labels that represent the real outputs
            self.labels = [ tf.placeholder(shape=[None,],
                            dtype=tf.int64,
                            name='ei_{}'.format(t)) for t in range(yseq_len) ]

            #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
            self.dec_ip = [ tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO') ] + self.labels[:-1]


            # Basic LSTM cell wrapped in Dropout Wrapper
            self.keep_prob = tf.placeholder(tf.float32)

            # type
            self.type_model = tf.placeholder(tf.int32)

            #using attention
            self.using_attention = tf.placeholder(tf.int32)

            # define the basic cell
            basic_cell = tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.BasicLSTMCell(emb_dim, state_is_tuple=True),
                    output_keep_prob=self.keep_prob)
            # emb_dim = num_units
            # stack cells together : n layered model
            stacked_lstm = tf.contrib.rnn.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True)


            # for parameter sharing between training model
            #  and testing model
            with tf.variable_scope('decoder') as scope:
                #  build the seq2seq model
                #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions
                if self.type_model == 1: #train
                    feed_previous = False
                else:
                    feed_previous = True

                if self.using_attention == 1: # using attention
                    self.decode_outputs, self.decode_states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                        self.enc_ip,
                        self.dec_ip,
                        stacked_lstm,
                        xvocab_size,
                        yvocab_size,
                        emb_dim,
                        feed_previous=feed_previous
                    )
                else:
                    self.decode_outputs, self.decode_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                        self.enc_ip,
                        self.dec_ip,
                        stacked_lstm,
                        xvocab_size,
                        yvocab_size,
                        emb_dim,
                        feed_previous=feed_previous
                    )
            # build loss
            loss_weights = [ tf.ones_like(label, dtype=tf.float32) for label in self.labels ]
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(
                self.decode_outputs,
                self.labels,
                loss_weights,
                yvocab_size
            )
            tf.summary.scalar("loss", self.loss)
            # train op to minimize the loss
            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        __graph__()

    # get the feed dictionary
    def get_feed(self, X, Y, keep_prob, type, using_attention):
        feed_dict = { self.enc_ip[t]: X[t] for t in range(self.xseq_len) }
        feed_dict.update({ self.labels[t]: Y[t] for t in range(self.yseq_len) })
        feed_dict[self.keep_prob] = keep_prob # dropout prob
        feed_dict[self.type_model] = type
        feed_dict[self.using_attention] = using_attention
        return feed_dict

    def train_batch(self, sess, train_batch_gen, step, train_writer, using_attention=1):
        batchX, batchY = train_batch_gen.next() # get next batches
        feed_dict = self.get_feed(batchX, batchY, keep_prob=0.5, type=1,
        using_attention=using_attention) # build feed
        self.summary, _, loss_v = sess.run([self.merge, self.train_op, \
            self.loss], feed_dict)
        train_writer.add_summary(self.summary, step)
        return loss_v

    def eval_step(self, sess, eval_batch_gen, step, eval_writer, using_attention):
        batchX, batchY = eval_batch_gen.next() # get next batches
        feed_dict = self.get_feed(batchX, batchY, keep_prob=1., type=2,
            using_attention=using_attention) # build feed
        self.summary, loss_v, dec_op_v = sess.run([self.merge, self.loss, \
            self.decode_outputs], feed_dict)
        eval_writer.add_summary(self.summary, step)
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        return loss_v, dec_op_v, batchX, batchY

    def eval_batches(self, sess, eval_batch_gen, num_batches, step, eval_writer):
        losses = []
        for i in range(num_batches):
            write_tensorboard = (i==1)
            loss_v, dec_op_v, batchX, batchY = self.eval_step(sess, \
                eval_batch_gen, step, eval_writer)
            losses.append(loss_v)
        return np.mean(losses)

    def train(self, train_set, valid_set, sess=None, using_attention=1):
        saver = tf.train.Saver()
        if not sess: # if no session is given
            sess = tf.Session() # create a session
            sess.run(tf.global_variables_initializer()) # init all variables

        sys.stdout.write('\n<log> Training started </log>\n')
        # run M epochs
        train_writer = tf.summary.FileWriter('logs/128_attention/train', sess.graph)
        eval_writer = tf.summary.FileWriter('logs/128_attention/eval')
        self.merge = tf.summary.merge_all()
        for i in range(self.epochs):
            try:
                print 'Train step: %i' % i
                self.train_batch(sess, train_set, i, train_writer)
                if i % 50 == 0:
                    if i:
                        saver.save(sess, self.ckpt_path + self.model_name + \
                            '.ckpt', global_step=i) # save model to disk
                    # evaluate to get validation loss
                    val_loss = self.eval_batches(sess, valid_set, 1, i, eval_writer)
                    print('\nModel saved to disk at iteration #{}'.format(i))
                    print('val   loss : {0:.6f}'.format(val_loss))
                    sys.stdout.flush()
            except KeyboardInterrupt: # this will most definitely happen, so handle it
                print('Interrupted by user at iteration {}'.format(i))
                self.session = sess
                return sess

    def restore_last_session(self):
        saver = tf.train.Saver()
        sess = tf.Session()  # create a session
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path) # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path) # restore session
        else:
            sess.run(tf.global_variables_initializer())
        return sess

    # prediction
    def predict(self, sess, X, Y, using_attention=1):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_prob] = 1.
        feed_dict[self.type_model] = 3
        feed_dict[self.using_attention] = using_attention
        dec_op_v = sess.run(self.decode_outputs, feed_dict)
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        return np.argmax(dec_op_v, axis=2) # return the index of item with highest probability
