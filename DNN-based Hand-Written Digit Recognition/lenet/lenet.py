import numpy as np
import struct
import time
from datetime import datetime

# Constants
NUM_TESTS = 10000


# Network parameters and outputs initialization
# images = np.zeros((NUM_TESTS, 28, 28), dtype=np.uint8)
# labels = np.zeros(NUM_TESTS, dtype=np.uint8)
# image = np.zeros((1, 32, 32), dtype=float)
# conv1_weights = np.zeros((6, 1, 5, 5), dtype=float)
# conv1_bias = np.zeros(6, dtype=float)
# conv1_output = np.zeros((6, 28, 28), dtype=float)
# pool2_output = np.zeros((6, 14, 14), dtype=float)
# conv3_weights = np.zeros((16, 6, 5, 5), dtype=float)
# conv3_bias = np.zeros(16, dtype=float)
# conv3_output = np.zeros((16, 10, 10), dtype=float)
# pool4_output = np.zeros((16, 5, 5), dtype=float)
# conv5_weights = np.zeros((120, 16, 5, 5), dtype=float)
# conv5_bias = np.zeros(120, dtype=float)
# conv5_output = np.zeros((120, 1, 1), dtype=float)
# fc6_weights = np.zeros((10, 120, 1, 1), dtype=float)
# fc6_bias = np.zeros(10, dtype=float)
# fc6_output = np.zeros(10, dtype=float)


def relu(input_array):
    """
    relu activation function
    :param input_array:
    :return: relu(input_array)
    """
    return np.maximum(input_array, 0)


def convolution1(input, weights, bias, output_shape):
    """
    :param input: (1, 32, 32)
    :param weights: (6, 1, 5, 5)
    :param bias: (6, )
    :param output_shape: (6, 28, 28)
    :return output: (6, 28, 28)
    """
    output = np.zeros(output_shape, dtype=np.float32)
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


def relu_conv(input, output_shape):
    """
    relu activation function for convolution layer
    :param input:
    :param output_shape:
    :return: output
    """
    output = np.zeros(output_shape, dtype=np.float32)

    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            for k in range(output_shape[2]):
                output[i][j][k] = relu(input[i][j][k])

    return output


def max_pooling(input, output_shape):
    """
    max pooling layer
    :param input:
    :param output_shape:
    :return: output
    """
    output = np.zeros(output_shape, dtype=np.float32)
    for c in range(output_shape[0]):
        for h in range(output_shape[1]):
            for w in range(output_shape[2]):
                max_value = float('-inf')
                for i in range(h * 2, h * 2 + 2):
                    for j in range(w * 2, w * 2 + 2):
                        max_value = max(max_value, input[c][i][j])
                output[c][h][w] = max_value
    return output


def convolution3(input, weights, bias, output_shape):
    """
    :param input:  (6, 14, 14)
    :param weights: (16, 6, 5, 5)
    :param bias: (16, )
    :param output_shape: (16, 10, 10)
    :return output: (16, 10, 10)
    """
    output = np.zeros(output_shape, dtype=np.float32)

    for co in range(output_shape[0]):
        for h in range(output_shape[1]):
            for w in range(output_shape[2]):
                sum_val = 0.0
                for i in range(h, h + 5):
                    for j in range(w, w + 5):
                        for ci in range(6):
                            sum_val += weights[co][ci][i - h][j - w] * input[ci][i][j]
                output[co][h][w] = sum_val + bias[co]

    return output


def convolution5(input, weights, bias, output_shape):
    """
    :param input: (16, 5, 5)
    :param weights: (120, 16, 5, 5)
    :param bias: (120, )
    :param output_shape: (120, 1, 1)
    """
    output = np.zeros(output_shape, dtype=np.float32)

    for co in range(output_shape[0]):
        sum_val = 0.0
        for i in range(5):
            for j in range(5):
                for ci in range(16):
                    sum_val += weights[co][ci][i][j] * input[ci][i][j]
        output[co][0][0] = sum_val + bias[co]

    return output


def fully_connected_6(input, weights, bias, output_shape):
    """
    :param input: (120, 1, 1)
    :param weights: (10, 120, 1, 1)
    :param bias: (10, )
    :param output_shape: (10, )
    :return output: (10, )
    """
    output = np.zeros(output_shape, dtype=np.float32)

    for n in range(output_shape[0]):
        output[n] = 0
        for c in range(120):
            output[n] += weights[n][c][0][0] * input[c][0][0]
        output[n] += bias[n]

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
    with open(filename, 'rb') as f:
        conv1_weights = np.fromfile(f, dtype=np.float32, count=6 * 1 * 5 * 5).reshape((6, 1, 5, 5))
        conv1_bias = np.fromfile(f, dtype=np.float32, count=6)
        conv3_weights = np.fromfile(f, dtype=np.float32, count=16 * 6 * 5 * 5).reshape((16, 6, 5, 5))
        conv3_bias = np.fromfile(f, dtype=np.float32, count=16)
        conv5_weights = np.fromfile(f, dtype=np.float32, count=120 * 16 * 5 * 5).reshape((120, 16, 5, 5))
        conv5_bias = np.fromfile(f, dtype=np.float32, count=120)
        fc6_weights = np.fromfile(f, dtype=np.float32, count=10 * 120 * 1 * 1).reshape((10, 120, 1, 1))
        fc6_bias = np.fromfile(f, dtype=np.float32, count=10)

    return conv1_weights, conv1_bias, conv3_weights, conv3_bias, conv5_weights, conv5_bias, fc6_weights, fc6_bias


def get_image(images, idx):
    img = images[idx]
    img_padded = np.pad(img, ((2, 2), (2, 2)), mode='constant', constant_values=-1)
    return img_padded / 255.0 * 2.0 - 1.0


def main():
    print("Starting LeNet")

    # Initialize network parameters
    print("Parsing MNIST images")
    images = parse_mnist_images("images.bin")

    print("Parsing MNIST labels")
    labels = parse_mnist_labels("labels.bin")

    print("Parsing network parameters")

    (conv1_weights,
     conv1_bias,
     conv3_weights,
     conv3_bias,
     conv5_weights,
     conv5_bias,
     fc6_weights,
     fc6_bias) = parse_parameters("params.bin")

    num_correct = 0
    start_time = datetime.now()

    for k in range(NUM_TESTS):
        print(f"new image index{k}")
        img = get_image(images, k)
        img = img.reshape((1, 32, 32))

        # Forward pass through the network
        conv1 = convolution1(img, conv1_weights, conv1_bias, (6, 28, 28))
        relu1 = relu(conv1)
        pool2 = max_pooling(relu1, (6, 14, 14))
        conv3 = convolution3(pool2, conv3_weights, conv3_bias, (16, 10, 10))
        relu3 = relu(conv3)
        pool4 = max_pooling(relu3, (16, 5, 5))
        conv5 = convolution5(pool4, conv5_weights, conv5_bias, (120, 1, 1))
        relu5 = relu(conv5)
        fc6 = fully_connected_6(relu5, fc6_weights, fc6_bias, (10, ))

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
