'''
基于numpy实现rnn的前向传播和反向传播。
'''
import numpy as np

class Rnn:
    def __init__(self, hidden_layers: int, hidden_status_size: int, input_size: int, output_size: int):
        '''
        初始化
        :param hidden_layers: 隐藏层数
        :param input_size: 输入大小
        :param output_size: 输出大小
        '''
        self.U = np.random.uniform(-np.sqrt(1/hidden_status_size), np.sqrt(1/input_size), (hidden_status_size, input_size))
        self.V = np.random.uniform(-np.sqrt(1/output_size), np.sqrt(1/hidden_status_size), (output_size, hidden_status_size))
        self.W = np.random.uniform(-np.sqrt(1/hidden_status_size), np.sqrt(1/hidden_status_size), (hidden_status_size, hidden_status_size))
        self.b = np.zeros(input_size, 1)
        self.c = np.zeros(output_size, 1)

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_status_size = hidden_status_size


    def softmax(self, x):
        p = np.exp(x - np.max(x))
        return p/np.sum(p)

    def forward(self, x):
        '''
        前向反馈
        :param x: 输入，大小必须为(input_size, hidden_layers)
        :return: hidden_stauts, output
        '''
        if np.size(x) != (self.input_size, self.hidden_layers):
            print("intput size is error, expect to be:{} get:{}".format((self.input_size, self.hidden_layers), np.size(x)))
            exit(-1)
        hidden_status = np.array((self.hidden_status_size, self.hidden_layers))
        output = np.array((self.output_size, self.hidden_layers))
        for t in range(self.hidden_layers):
            hidden_status[t] = np.tanh(np.dot(self.U, x[t]) + np.dot(self.W, self.hidden_status[t-1]) + self.b)
            output[t] = self.softmax(np.dot(self.V, hidden_status[t]) + self.c)
        return hidden_status, output

    def backforward(self):
        pass



