import numpy
import scipy
from scipy.special import expit

# Определение класса нейронной сети
class neuralNetwork:
    # Инициализировать нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Задать количество узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Матрицы весовых коэффициентов связей wih и who
        # Весовые коэффициенты связей между узлом i и узлом j
        # следующего слоя обозначены как w_i_j:
        # w11 w21
        # w12 w22 и т.д.
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # Коэффициент обучения
        self.lr = learningrate

        # Использование сигмойды в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)

        pass
    # Тренировка нейронной сети
    def train(self, inputs_list, targets_list):
        # Преобразование списка входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # Рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # Рассчитать входящие сигналы для выходноо слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # Ошибки выходного слоя =
        # (целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        # Ошибки скрытого слоя - это ошибки outputs_errors,
        # и рекомбинированнные на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Обновить весовые коэффициенты для связей между
        # скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # Обновить весовые коэффициенты для связей между
        # выходным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass
    # Опорос нейронной сети
    def query(self, inputs_list):
        # Преобразовать список входных значений
        # в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        # Рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # Рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

print(n.query([1.0, 0.5, -1.5]))