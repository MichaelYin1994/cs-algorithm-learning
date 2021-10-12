# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 23:14:22 2019

@author: XPS13
"""


import csv
import itertools
import operator
import pickle
import numpy as np
import nltk
import sys
import gc
from datetime import datetime
from numba import jit
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
np.random.seed(2019)
###############################################################################
###############################################################################
class LoadSave(object):
    def __init__(self, fileName=None):
        self._fileName = fileName
    
    def save_data(self, data):
        assert self._fileName != None, "Invaild file path !"
        self.__save_data(data)
    
    def load_data(self):
        assert self._fileName != None, "Invaild file path !"
        return self.__load_data()
        
    def __save_data(self, data):
        print("--------------Start saving--------------")
        print("Save data to path {}.".format(self._fileName))
        f = open(self._fileName, "wb")
        pickle.dump(data, f)
        f.close()
        print("--------------Saving successed !--------------\n")
        
    def __load_data(self):
        assert self._fileName != None, "Invaild file path !"
        print("--------------Start loading--------------")
        print("Load from path {}.".format(self._fileName))
        f = open(self._fileName, 'rb')
        data = pickle.load(f)
        f.close()
        print("--------------loading successed !--------------\n")
        return data

def create_training_data():
    vocabulary_size = 8000
    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"
    
    ###########################################################################
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading CSV file...")
    with open('reddit-comments-2015-08.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, skipinitialspace=True)
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))
    
    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    
    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    
    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))
    
    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
    
    print("\nExample sentence: '%s'" % sentences[0])
    print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])
    
    ###########################################################################
    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    
    # Print an training data example
    x_example, y_example = X_train[17], y_train[17]
    print("x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example))
    print("\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example))

    ls = LoadSave("Train.pkl")
    ls.save_data([X_train, y_train, index_to_word, word_to_index, vocab])


###############################################################################
###############################################################################
class RecurrentNeuralNetwork():
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
    
    # 输入：单语句
    @jit
    def forward_propagation(self, x):
        # 单条序列，计算长度
        T = len(x)
        
        # 对于单条序列，计算其length + 1的隐状态，每一时间步的输出也保存下来
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((T, self.word_dim))

        # 对于每一时间步而言...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = self.softmax(self.V.dot(s[t]))
        return [o, s]
    
    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)
    
    def softmax(self, x):
        xt = np.exp(x - np.max(x))
        return xt / np.sum(xt)
    
    # 输入：语料库
    def calculate_total_loss(self, X_train, y_train):
        # X_train: 编码过的语料库(One-hot encoding), List like
        # y_train: X_train对应的输出
        
        L = 0
        # 对于每一条句子而言...
        for i in np.arange(len(y_train)):
            # 计算每一条句子的o与s
            o, s = self.forward_propagation(X_train[i])
            # o[行，列(Predicted label)]
            correct_word_predictions = o[np.arange(len(y_train[i])), y_train[i]]
            
            # 计算实际的label(+1)，与预测的输出o[行，列(Predicted label)]的交叉熵
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L
    
    # 输入：语料库
    def calculate_loss(self, X_train, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(X_train, y)/N
    
    # 输入：单语句
    def back_propagation_through_time(self, X, y):
        T = len(y)
        # 前向传播计算o与隐状态s
        o, s = self.forward_propagation(X)
        
        # 累积每一个变量的梯度值
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1
        
        # 对于每一个输出进行反向传播...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T) # 计算矩阵的外积
            
            # 初始化delta的计算
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # 通过时间进行反向传播(最多传播self.bptt_truncate步)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])              
                dLdU[:, X[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]
    
    # 输入：单语句
    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # 通过反向传播计算梯度
        bptt_gradients = self.back_propagation_through_time(x, y)
        
        # 列出需要检查梯度的变量的名称
        model_parameters = ['U', 'V', 'W']
        
        # 对于每一个变量进行梯度校验
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            # 获取pname对应的变量的实际parameter
            parameter = operator.attrgetter(pname)(self)
            print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
            
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    print("+h Loss: %f" % gradplus)
                    print("-h Loss: %f" % gradminus)
                    print("Estimated_gradient: %f" % estimated_gradient)
                    print("Backpropagation gradient: %f" % backprop_gradient)
                    print("Relative Error: %f" % relative_error)
                    return 
                it.iternext()
            print("Gradient check for parameter %s passed." % (pname))
    
    # 输入：单语句
    def stochastic_gradient_descent(self, x, y, learning_rate):
        # 计算梯度
        dLdU, dLdV, dLdW = self.back_propagation_through_time(x, y)
        
        # 根据学习率与梯度更新参数
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
    
    # Outer SGD Loop
    # - model: The RNN model instance
    # - X_train: The training data set
    # - y_train: The training data labels
    # - learning_rate: Initial learning rate for SGD
    # - nepoch: Number of times to iterate through the complete dataset
    # - evaluate_loss_after: Evaluate the loss after this many epochs
    @jit
    def fit(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        # 保存计算的loss信息，方便后续的绘图
        losses = []
        num_examples_seen = 0
        
        # 对于每一个epoch进行迭代
        for epoch in range(nepoch):
            # 每evaluate_loss_after步打印一次梯度
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                # 自适应的学习率
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5  
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            # 对于每一个样本...
            for i in range(len(y_train)):
                # 单独更新一次梯度
                self.stochastic_gradient_descent(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1
        return losses
###############################################################################
###############################################################################
if __name__ == "__main__":
    ls = LoadSave("Train.pkl")
    data = ls.load_data()
    X_train, y_train, index_to_word, word_to_index, vocab = data[0], data[1], data[2], data[3], data[4]
    del data
    gc.collect()
    
    vocabulary_size = 8000
    rnn = RecurrentNeuralNetwork(vocabulary_size)
#    o, s = rnn.forward_propagation(X_train[21])
#    predictions = rnn.predict(X_train[21])
#    print("Expected Loss for random predictions: %f" % np.log(len(index_to_word)))
#    print("Actual loss: %f" % rnn.calculate_loss(X_train[:1000], y_train[:1000]))
    losses = rnn.fit(X_train[:500], y_train[:500], learning_rate=0.02, nepoch=10, evaluate_loss_after=2)