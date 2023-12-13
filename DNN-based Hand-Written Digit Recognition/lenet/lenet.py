import numpy as np
import struct
import time
from datetime import datetime

# Constants
NUM_TESTS = 10000

# Network parameters and outputs initialization
images = np.zeros((NUM_TESTS, 28, 28), dtype=np.uint8)
labels = np.zeros(NUM_TESTS, dtype=np.uint8)
image = np.zeros((1, 32, 32), dtype=float)
conv1_weights = np.zeros((6, 1, 5, 5), dtype=float)
conv1_bias = np.zeros(6, dtype=float)
conv1_output = np.zeros((6, 28, 28), dtype=float)
pool2_output = np.zeros((6, 14, 14), dtype=float)
conv3_weights = np.zeros((16, 6, 5, 5), dtype=float)
conv3_bias = np.zeros(16, dtype=float)
conv3_output = np.zeros((16, 10, 10), dtype=float)
pool4_output = np.zeros((16, 5, 5), dtype=float)
conv5_weights = np.zeros((120, 16, 5, 5), dtype=float)
conv5_bias = np.zeros(120, dtype=float)
conv5_output = np.zeros((120, 1, 1), dtype=float)
fc6_weights = np.zeros((10, 120, 1, 1), dtype=float)
fc6_bias = np.zeros(10, dtype=float)
fc6_output = np.zeros(10, dtype=float)

def relu(input_array):
    return np.maximum(input_array, 0)

def convolution(input, weights, bias, output_shape):
    output = np.zeros(output_shape)
    for co in range(output_shape[0]):
        for h in range(output_shape[1]):
            for w in range(output_shape[2]):
                sum = 0
                for i in range(h, h + 5):
                    for j in range(w, w + 5):
                        for ci in range(input.shape[0]):
                            sum += weights[co][ci][i - h][j - w] * input[ci][i][j]
                output[co][h][w] = sum + bias[co]
    return output

def max_pooling(input, output_shape):
    output = np.zeros(output_shape)
    for c in range(output_shape[0]):
        for h in range(output_shape[1]):
            for w in range(output_shape[2]):
                max_value = float('-inf')
                for i in range(h * 2, h * 2 + 2):
                    for j in range(w * 2, w * 2 + 2):
                        max_value = max(max_value, input[c][i][j])
                output[c][h][w] = max_value
    return output

def parse_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, 28, 28)
    return images

def parse_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels
  
def parse_parameters(filename):
    global conv1_weights, conv1_bias, conv3_weights, conv3_bias
    global conv5_weights, conv5_bias, fc6_weights, fc6_bias

    with open(filename, 'rb') as f:
        conv1_weights = np.fromfile(f, dtype=np.float32, count=6*1*5*5).reshape((6, 1, 5, 5))
        conv1_bias = np.fromfile(f, dtype=np.float32, count=6)
        conv3_weights = np.fromfile(f, dtype=np.float32, count=16*6*5*5).reshape((16, 6, 5, 5))
        conv3_bias = np.fromfile(f, dtype=np.float32, count=16)
        conv5_weights = np.fromfile(f, dtype=np.float32, count=120*16*5*5).reshape((120, 16, 5, 5))
        conv5_bias = np.fromfile(f, dtype=np.float32, count=120)
        fc6_weights = np.fromfile(f, dtype=np.float32, count=10*120*1*1).reshape((10, 120, 1, 1))
        fc6_bias = np.fromfile(f, dtype=np.float32, count=10)

        
def get_image(images, idx):
    img = images[idx]
    img_padded = np.pad(img, ((2, 2), (2, 2)), mode='constant', constant_values=-1)
    return img_padded / 255.0 * 2.0 - 1.0

def main():
    print("Starting LeNet")

    images = parse_mnist_images("images.bin")
    labels = parse_mnist_labels("labels.bin")
    parse_parameters("params.bin")

    num_correct = 0
    start_time = datetime.now()


    for k in range(NUM_TESTS):
        img = get_image(images, k)
        img = img.reshape((1, 32, 32))

        # Forward pass through the network
        conv1 = convolution(img, conv1_weights, conv1_bias, (6, 28, 28))
        relu1 = relu(conv1)
        pool2 = max_pooling(relu1, (6, 14, 14))
        conv3 = convolution(pool2, conv3_weights, conv3_bias, (16, 10, 10))
        relu3 = relu(conv3)
        pool4 = max_pooling(relu3, (16, 5, 5))
        conv5 = convolution(pool4, conv5_weights, conv5_bias, (120, 1, 1))
        relu5 = relu(conv5)
        fc6 = np.dot(fc6_weights.reshape(10, 120), relu5.reshape(120)) + fc6_bias

        # Index of the largest output is the result
        result = np.argmax(fc6)

        if result == labels[k]:
            num_correct += 1

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Accuracy = {num_correct / NUM_TESTS * 100}%")
    print(f"Total time: {duration} seconds")

if __name__ == "__main__":
    main()
