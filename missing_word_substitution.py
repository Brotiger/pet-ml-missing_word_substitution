import numpy as np
import pandas as pd
import sys, random, math
import os
from collections import Counter

np.random.seed(1)
random.seed(1)

# Удаляет символы из строки
def clean_line(line, symbols):
    for symbol in symbols:
        line = line.replace(symbol, "")
    return line.replace("  ", " ")

# Разбивает строку массив, удаляет пустые элементы
def split_tokens(line):
    tokens = list(set(line.split(" ")))
    return tokens

def get_wordcnt(tokens):
    wordcnt = Counter()
    for sent in tokens:
        for word in sent:
            wordcnt[word] -= 1
    return wordcnt

# Возвращает 10 слов для которых Евклидово расстояние меньше всего
def similar(target = 'beautiful'):
    target_index = word2index[target]
    scores = Counter()
    
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - (weights_0_1[target_index])
        squared_difference = raw_difference * raw_difference
        # Суммируем различия по всем весам так как иначе не сможем сравнить и выбрать 10 слов у которых разница в весах наименьшая
        # Можно обойтись без вычисления корня, разница не поменяется, просто числа будут в 2 раза больше
        # Минус добавляется так как нам нужны наиболее близкие друг к другу слова, а чем больше расстояние между их весами тем дальше они друг от друга
        # Все выше сказанное так же подходим под формулу рассчета Евклидова расстояния
        scores[word] = -math.sqrt(sum(squared_difference))
    return scores.most_common(10)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

number_of_lines = 1000
stop_symbols = ["<br", "/>", ".", ",", "?", "!"]

raw_reviews = []

data = pd.read_csv(os.environ['MWS_DATASET_PATH'])
raw_reviews = data['review'].head(number_of_lines)

for i, raw_review in enumerate(raw_reviews):
    raw_review = raw_review.lower()
    raw_reviews[i] = clean_line(raw_review, stop_symbols)

# Набор строк состоящих из массива слов
tokens = list(map(split_tokens, raw_reviews))
wordcnt = get_wordcnt(tokens)
vocab = list(set(map(lambda x:x[0], wordcnt.most_common())))

# Формируем асоциативный массив, слово - его позация в словаре
word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i

# Так как ниже вместо dot мы используем sum для вычисления layer_1
# Eсли бы мы хотели использовать dot input_dataset было бы необходимо преобразовать к виду [0,1,0,1,0,0...],
# размер массива должен был бы равняться размеру словаря
concatenated = list()
input_dataset = list()
for sent in tokens:
    sent_indices = list()
    for word in sent:
        try:
            sent_indices.append(word2index[word])
            concatenated.append(word2index[word])
        except:
            ""
    input_dataset.append(list(set(sent_indices)))
concatenated = np.array(concatenated)

random.shuffle(input_dataset)

alpha = 0.05
iterations = 2

hidden_size = 50
window = 2
negative = 5

weights_0_1 = (np.random.rand(len(vocab), hidden_size) -0.5) * 0.2
weights_1_2 = np.random.rand(len(vocab), hidden_size) * 0

# Устанавливаем правильны вариант подстановки
layer_2_target = np.zeros(negative + 1)
layer_2_target[0] = 1

for rev_i, review in enumerate(input_dataset * iterations):
    for target_i in range(len(review)):
        # Берем слово из отзыва и конкатенируем с случайными 5 словами, 
        # прогнозируем только для подмножества так как прогнозировать для всего набора слов слишком дорого
        target_samples = [review[target_i]] + list(concatenated[(np.random.rand(negative) * len(concatenated)).astype('int').tolist()])
        
        # Берем кусок отзыва и вырезаем из него слова
        left_context = review[max(0, target_i - window): target_i]
        right_context = review[target_i + 1:min(len(review), target_i + window)]

        # Высчитываем среднее значение весов по стобцам
        layer_1 = np.mean(weights_0_1[left_context + right_context], axis=0)
        layer_2 = sigmoid(layer_1.dot(weights_1_2[target_samples].T))
        
        layer_2_delta = layer_2 - layer_2_target
        layer_1_delta = layer_2_delta.dot(weights_1_2[target_samples])
        
        weights_0_1[left_context + right_context] -= layer_1_delta * alpha
        weights_1_2[target_samples] -= np.outer(layer_2_delta, layer_1) * alpha
    
    # В процессе обучения выводим наиболее приближенные слова к слову terrible
    if (rev_i % 250 == 0 or rev_i == (number_of_lines * iterations) - 1):
        sys.stdout.write('\nIter: ' + str(rev_i) + ' Progress:' + str(rev_i/float(len(input_dataset) * iterations)) + " " + str(similar('terrible')))