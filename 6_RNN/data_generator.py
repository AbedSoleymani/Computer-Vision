import numpy as np
import matplotlib.pyplot as plt

class Data_generator:
    def __init__(self, seq_length=20):
        self.seq_length = seq_length

    def generate(self):
        time_steps = np.linspace(0, np.pi, self.seq_length + 1)
        data = np.sin(time_steps)
        data.resize((self.seq_length + 1, 1))

        return data, time_steps

    @staticmethod
    def display(input, prediction, time_steps, title):
        plt.plot(time_steps[:len(input)], input, 'r.', label='input, x')
        plt.plot(time_steps[:len(prediction)], prediction, 'b.', label='prediction')
        plt.legend(loc='best')
        plt.title(title)
        plt.show()