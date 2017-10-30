import collections
import numpy as np

start_token = 'S'
end_token = 'E'
def process_poems(file_name):
    # 诗集
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:#title是诗的标题
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                #如果一首诗的长度很短或很长，则跳过
                if len(content) < 5 or len(content) > 79:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass
    # 按一首诗的字数从短到长排序排序
    # poems为每首诗为元素组成的列表
    poems = sorted(poems, key=lambda l: len(line))
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    # 这里根据包含了每个字对应的频率
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)  #为所有字符组成的列表
    # 取前多少个常用字
    words = words[:len(words)] + (' ',) #添加一个空格字符
    # 每个字映射为一个数字ID
    word_int_map = dict(zip(words, range(len(words))))
    #{'，': 0, '。': 1, 'S': 2, 'E': 3, '不': 4, '人': 5, '山': 6, '风': 7, 
    #'日': 8, '无': 9, '一': 10, '云': 11, '花': 12, '春': 13, '来': 14, 
    #'何': 15, '水': 16, '月': 17, '上': 18, '有': 19, '时': 20, '中': 21, 
    #'天': 22, '年': 23, '归': 24, '秋': 25, '相': 26, '知': 27, '长': 28}
    #word_int_map为一个字典，键为字符，值为该字符对应的编号
    poems_vector = [list(map(lambda word: 
            word_int_map.get(word, len(words)), poem)) for poem in poems]
    #poems_vector为二维列表，以每首诗中的每个字符的编号构成的列表为元素
    return poems_vector, word_int_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    # 每次取64首诗进行训练
    num_batches = len(poems_vec) // batch_size  #总共的batchs
    x_batches = []
    y_batches = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        batches = poems_vec[start_index:end_index]
        # 找到这个batch的所有poem中最长的poem的长度
        length = max(map(len, batches))
        # 填充一个这么大小的空batch，空的地方放空格对应的index标号
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row in range(batch_size):
            # 每一行就是一首诗，在原本的长度上把诗还原上去
            x_data[row, :len(batches[row])] = batches[row]
        y_data = np.copy(x_data)
        #y的数据从开始到倒数第二个元素为x_data[:, 1:],最后一个元素为x_data的最后一个
        #元素，为'空格'的编号或者end_token对应的编号
        y_data[:, :-1] = x_data[:, 1:] 
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches
