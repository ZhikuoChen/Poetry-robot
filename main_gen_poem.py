'''
测试命令:python main_gen_poem.py --mode gen
'''
import os,time,math,sys
import numpy as np
import tensorflow as tf
from model_poem import lstm
from process_poems import process_poems, generate_batch
tf.app.flags.DEFINE_string('--mode', 'gen', 'train/gen')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_integer('lstm_size', 128, 'lstm size.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'num layers.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
tf.app.flags.DEFINE_string('checkpoint_dir','.\checkpoints', 'checkpoints save path.')
tf.app.flags.DEFINE_string('file_path','data/poems.txt', 'file name of poems.')
tf.app.flags.DEFINE_string('model_prefix', 'ner.ckp', 'model save prefix.')
tf.app.flags.DEFINE_integer('epochs', 40, 'train how many epochs.')
FLAGS = tf.app.flags.FLAGS
start_token = 'S'
end_token = 'E'
def time_transform(s):
    ret = ''
    if s >= 60 * 60:
        h = math.floor(s / (60 * 60))
        ret += '{}h'.format(h)
        s -= h * 60 * 60
    if s >= 60:
        m = math.floor(s / 60)
        ret += '{}m'.format(m)
        s -= m * 60
    if s >= 1:
        s = math.floor(s)
        ret += '{}s'.format(s)
    return ret
def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample >= len(vocabs):
       sample = len(vocabs) - 1
    return vocabs[sample]
 
def pretty_print_poem(poem):
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')
if not os.path.exists(FLAGS.checkpoint_dir):
   os.makedirs(FLAGS.checkpoint_dir)
#word_int_map为一个字典，键为字符，值为该字符对应的编号
#poems_vector为二维列表，以每首诗中的每个字符的编号构成的列表为元素
#vocabularies#为所有字符组成的列表
poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)
print('共有%d个字符' %len(vocabularies))
#batches_inputs为三维列表，第一维为num_batches存储所有的批数,第二维为batch_size
#，第三维为该batch中的最大长度
#batches_outputs为三维列表，第一维为num_batches存储所有的批数,第二维为batch_size
#，第三维为该batch中的最大长度
batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, 
                                             poems_vector, word_to_int)
#共541首诗
train_inputs=list(zip(batches_inputs,batches_outputs))  #将输入和输出合并 
if FLAGS.mode=='train':
   model = lstm(vocab_size=len(vocabularies), lstm_size=FLAGS.lstm_size, 
                   num_layers=FLAGS.num_layers, batch_size=FLAGS.batch_size, 
                   learning_rate=FLAGS.learning_rate)
   with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        start_epoch=1
        #若训练过程中，半途结束则接着训练
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        # 开始训练
        print('start training...')
        metrics = '  '.join(['\r[{}]','{:.1f}%','{:s}','Epoch:{:d}','{}/{}','loss={:.3f}','{}/{}'])
        bars_max = 20
        for epoch in range(start_epoch,FLAGS.epochs+1):
            num_batches = len(poems_vector) // FLAGS.batch_size
            np.random.shuffle(train_inputs)    #混杂数据
            batch_trained=0
            time_start = time.time()
            for batch in range(num_batches):
                batch_trained+=FLAGS.batch_size
                feed_dict={model.input_data: train_inputs[batch][0], 
                           model.output_targets: train_inputs[batch][1]}
                _,loss = sess.run([model.train_op,model.total_loss],feed_dict=feed_dict)
                time_now = time.time()
                time_spend = time_now - time_start  #已花费时间
                time_estimate = time_spend / (batch_trained / (num_batches*FLAGS.batch_size))  #预估总计花费时间
                #求每轮训练已完成的百分比
                percent = min(100, batch_trained /  (num_batches*FLAGS.batch_size)) * 100
                bars = math.floor(percent / 100 * bars_max) 
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                sys.stdout.write(
                          metrics.format('=' * bars + '-' * (bars_max - bars),
                          percent,timestamp,epoch,batch_trained, num_batches*FLAGS.batch_size,loss,
                          time_transform(time_spend), time_transform(time_estimate)))
                sys.stdout.flush()    #刷新
#                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#                print('[%s]: Epoch: %d batch: %d training loss: %.6f' % (timestamp,epoch, batch, loss))
            if epoch % 5 == 0:
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.model_prefix), global_step=epoch)
else:
    print('write tang poem...')
    model = lstm(vocab_size=len(vocabularies), lstm_size=FLAGS.lstm_size, 
                   num_layers=FLAGS.num_layers, batch_size=1, 
                   learning_rate=FLAGS.learning_rate)
    with tf.Session() as sess:
         sess.run(tf.global_variables_initializer())
         saver = tf.train.Saver(tf.global_variables())
         ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
         if ckpt: 
            saver.restore(sess, ckpt)
            print("Reading model parameters from %s" % ckpt)
         #'S'
         start = np.array([list(map(word_to_int.get, start_token))])
         feed_dict={model.input_data: start}
         #注意这里不能写model.train_op，因为没有给model.output_targets赋值，无法求解。
         [predict, last_state] = sess.run([model.prediction,
                                               model.last_state],feed_dict=feed_dict)
         begin_word = "春"
         if begin_word:
            word = begin_word
         else:
            word = to_word(predict, vocabularies)
         poem = ''
         while word != end_token:
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = word_to_int[word]
            feed_dict={model.input_data: x, model.initial_state: last_state}
            [predict, last_state] = sess.run([model.prediction,
                                     model.last_state],feed_dict=feed_dict)
            word = to_word(predict, vocabularies)
         pretty_print_poem(poem)
         




