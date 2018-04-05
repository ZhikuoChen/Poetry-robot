# Poetry-robot
作诗机器人部分:遍历每首诗，每每首诗的开头加入起始标志位'S'和终止标志位'E'。然后再次遍历每首诗，
按照每个汉字出现次数的多少从上到下编号，创建字典，键为汉字，值为该汉字对应的编号，最后将填充符'pad'加入
字典中。再次遍历每首诗，将每首诗的所有字符都转化成字符对应的数字。在每个epoch开始后，都进行数据混杂，
然后每次从训练集中选择batch_size个数据，求出这batch_size个数据最长诗的长度，其他的句子也都用pad对应
的编号填充到最大长度。batches_outputs为batches_inputs右移一位并加上到最后再加上batches_inputs的最后
一位得到。先将batches_inputs输入到embedding层，而每个字对应的词向量是随机产生的
得到inputs[batch_size,max_length,lstm_size],然后再经过两层lstm得到outputs形状为
[batch_size, max_length, lstm_size]，reshpae为[batch_size*max_length, lstm_size]，经过
projection层得到logits,形状为[batch_size*max_length, vocab_size]，然后将其与目标
batches_outputs进行softmax操作，得到loss，使用Adam优化器进行优化。
