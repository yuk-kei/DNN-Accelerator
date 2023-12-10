import numpy as np
import struct
from time import perf_counter

NUM_TESTS = 10000

# Static allocation of network parameters and their outputs
images = np.zeros((NUM_TESTS, 28, 28), dtype=np.uint8)
labels = np.zeros(NUM_TESTS, dtype=np.uint8)

image = np.zeros((1, 32, 32))
conv1_weights = np.zeros((6, 1, 5, 5))
conv1_bias = np.zeros(6)
conv1_output = np.zeros((6, 28, 28))

pool2_output = np.zeros((6, 14, 14))

conv3_weights = np.zeros((16, 6, 5, 5))
conv3_bias = np.zeros(16)
conv3_output = np.zeros((16, 10, 10))

pool4_output = np.zeros((16, 5, 5))

conv5_weights = np.zeros((120, 16, 5, 5))
conv5_bias = np.zeros(120)
conv5_output = np.zeros((120, 1, 1))

fc6_weights = np.zeros((10, 120, 1, 1))
fc6_bias = np.zeros(10)
fc6_output = np.zeros(10)


def relu(input_value):
    return np.maximum(0, input_value)


def convolution(input_array, weights, bias, output):
    for co in range(weights.shape[0]):
        for h in range(output.shape[1]):
            for w in range(output.shape[2]):
                sum = 0
                for i in range(h, h + 5):
                    for j in range(w, w + 5):
                        if 0 <= i < input_array.shape[1] and 0 <= j < input_array.shape[2]:
                            sum += weights[co][0][i-h][j-w] * input_array[0][i][j]
                output[co][h][w] = relu(sum + bias[co])


def max_pooling(input_array, output):
    for c in range(output.shape[0]):
        for h in range(output.shape[1]):
            for w in range(output.shape[2]):
                max_value = -np.inf
                for i in range(h*2, h*2 + 2):
                    for j in range(w*2, w*2 + 2):
                        if 0 <= i < input_array.shape[1] and 0 <= j < input_array.shape[2]:
                            max_value = max(max_value, input_array[c][i][j])
                output[c][h][w] = max_value


def fully_connected(input_array, weights, bias, output):
    for n in range(weights.shape[0]):
        sum = 0
        for c in range(weights.shape[1]):
            sum += weights[n][c][0][0] * input_array[c][0][0]
        output[n] = relu(sum + bias[n])
