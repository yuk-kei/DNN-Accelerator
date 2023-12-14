import numpy as np


# Start function definitions of different layers
def relu(input):
    return np.maximum(0, input)


# Convolution Layer 1
def convolution1(input, weights, bias):
    output = np.zeros((6, 28, 28), dtype=np.float32)

    for co in range(6):
        for h in range(28):
            for w in range(28):
                sum_val = 0.0
                for i in range(h, h + 5):
                    for j in range(w, w + 5):
                        sum_val += weights[co][0][i - h][j - w] * input[0][i][j]
                output[co][h][w] = sum_val + bias[co]

    return output


# ReLU Layer 1
def relu1(input):
    output = np.zeros((6, 28, 28), dtype=np.float32)

    for i in range(6):
        for j in range(28):
            for k in range(28):
                output[i][j][k] = relu(input[i][j][k])

    return output


# Pooling Layer 2
def max_pooling2(input):
    output = np.zeros((6, 14, 14), dtype=np.float32)

    for c in range(6):
        for h in range(14):
            for w in range(14):
                max_value = -float("inf")
                for i in range(h * 2, h * 2 + 2):
                    for j in range(w * 2, w * 2 + 2):
                        max_value = np.maximum(max_value, input[c][i][j])
                output[c][h][w] = max_value

    return output


# ReLU Layer 2
def relu2(input):
    output = np.zeros((6, 14, 14), dtype=np.float32)

    for i in range(6):
        for j in range(14):
            for k in range(14):
                output[i][j][k] = relu(input[i][j][k])

    return output


# Convolution Layer 3
def convolution3(input, weights, bias):
    output = np.zeros((16, 10, 10), dtype=np.float32)

    for co in range(16):
        for h in range(10):
            for w in range(10):
                sum_val = 0.0
                for i in range(h, h + 5):
                    for j in range(w, w + 5):
                        for ci in range(6):
                            sum_val += weights[co][ci][i - h][j - w] * input[ci][i][j]
                output[co][h][w] = sum_val + bias[co]

    return output


# ReLU Layer 3
def relu3(input):
    output = np.zeros((16, 10, 10), dtype=np.float32)

    for i in range(16):
        for j in range(10):
            for k in range(10):
                output[i][j][k] = relu(input[i][j][k])

    return output


# Pooling Layer 4
def max_pooling4(input):
    output = np.zeros((16, 5, 5), dtype=np.float32)

    for c in range(16):
        for h in range(5):
            for w in range(5):
                max_value = -float("inf")
                for i in range(h * 2, h * 2 + 2):
                    for j in range(w * 2, w * 2 + 2):
                        max_value = np.maximum(max_value, input[c][i][j])
                output[c][h][w] = max_value

    return output


# ReLU Layer 4
def relu4(input):
    output = np.zeros((16, 5, 5), dtype=np.float32)

    for i in range(16):
        for j in range(5):
            for k in range(5):
                output[i][j][k] = relu(input[i][j][k])

    return output


# Convolution Layer 5
def convolution5(input, weights, bias):
    output = np.zeros((120, 1, 1), dtype=np.float32)

    for co in range(120):
        sum_val = 0.0
        for i in range(5):
            for j in range(5):
                for ci in range(16):
                    sum_val += weights[co][ci][i][j] * input[ci][i][j]
        output[co][0][0] = sum_val + bias[co]

    return output


# ReLU Layer 5
def relu5(input):
    output = np.zeros((120, 1, 1), dtype=np.float32)

    for i in range(120):
        output[i][0][0] = relu(input[i][0][0])

    return output


# Fully connected Layer 6
def fc6(input, weights, bias):
    output = np.zeros(10, dtype=np.float32)

    for n in range(10):
        output[n] = 0
        for c in range(120):
            output[n] += weights[n][c][0][0] * input[c][0][0]
        output[n] += bias[n]

    return output


def get_image(images, idx):
    image = np.zeros((1, 32, 32), dtype=np.float32)

    for i in range(32):
        for j in range(32):
            if i < 2 or i > 29 or j < 2 or j > 29:
                image[0, i, j] = -1.0
            else:
                image[0, i, j] = images[idx * 28 * 28 + (i - 2) * 28 + j - 2] / 255.0 * 2.0 - 1.0
    return image


def parse_mnist_images(filename):
    try:
        with open(filename, 'rb') as rf:
            print("Opened MNIST images data file")

            # Read the header (4 unsigned integers)
            header = np.fromfile(rf, dtype=np.uint32, count=4)
            if len(header) != 4:
                print("ERROR: Failed to read the header from images file")
            print("Read header from file")

            # Read images data (unsigned chars)
            images = np.fromfile(rf, dtype=np.uint8, count=NUM_TESTS * 28 * 28)
            images = images.reshape((NUM_TESTS * 28 * 28))
            print("Read images from file")

            return images

    except FileNotFoundError:
        print("ERROR when opening MNIST images data file!")
        return None


def parse_mnist_labels(filename):
    try:
        # Read labels data
        with open(filename, 'rb') as rf_labels:
            print("Opened MNIST labels data file")

            # Read the header (2 unsigned integers)
            header_labels = np.fromfile(rf_labels, dtype=np.uint32, count=2)
            # Check if the header was read successfully
            if len(header_labels) != 2:
                print("Can't read header from file")
            print("Read header from file")

            # Read labels data (unsigned chars)
            labels = np.fromfile(rf_labels, dtype=np.uint8, count=NUM_TESTS)
            print("Read labels from file")

            # Verify if the expected number of bytes for the labels data was read
            if labels.size != NUM_TESTS:
                print("Can't read labels from file")
                return None

        return labels

    except FileNotFoundError:
        print("ERROR when opening MNIST label data file!")
        return None


def parse_parameters(filename):
    try:
        # Open the parameter file
        with open(filename, 'rb') as rf:
            print("Opened parameter file")

            # Read conv1_weights (150 float values)
            conv1_weights = np.fromfile(rf, dtype=np.float32, count=150)
            if len(conv1_weights) != 150:
                print("Can't read conv1_weights from file")
                return None, None, None, None, None, None, None, None
            print("Read conv1_weights from file")

            # Read conv1_bias (6 float values)
            conv1_bias = np.fromfile(rf, dtype=np.float32, count=6)
            if len(conv1_bias) != 6:
                print("Can't read conv1_bias from file")
                return None, None, None, None, None, None, None, None
            print("Read conv1_bias from file")

            # Read conv3_weights (2400 float values)
            conv3_weights = np.fromfile(rf, dtype=np.float32, count=2400)
            if len(conv3_weights) != 2400:
                print("Can't read conv3_weights from file")
                return None, None, None, None, None, None, None, None
            print("Read conv3_weights from file")

            # Read conv3_bias (16 float values)
            conv3_bias = np.fromfile(rf, dtype=np.float32, count=16)
            if len(conv3_bias) != 16:
                print("Can't read conv3_bias from file")
                return None, None, None, None, None, None, None, None
            print("Read conv3_bias from file")

            # Read conv5_weights (48000 float values)
            conv5_weights = np.fromfile(rf, dtype=np.float32, count=48000)
            if len(conv5_weights) != 48000:
                print("Can't read conv5_weights from file")
                return None, None, None, None, None, None, None, None
            print("Read conv5_weights from file")

            # Read conv5_bias (120 float values)
            conv5_bias = np.fromfile(rf, dtype=np.float32, count=120)
            if len(conv5_bias) != 120:
                print("Can't read conv5_bias from file")
                return None, None, None, None, None, None, None, None
            print("Read conv5_bias from file")

            # Read fc6_weights (1200 float values)
            fc6_weights = np.fromfile(rf, dtype=np.float32, count=1200)
            if len(fc6_weights) != 1200:
                print("Can't read fc6_weights from file")
                return None, None, None, None, None, None, None, None
            print("Read fc6_weights from file")

            # Read fc6_bias (10 float values)
            fc6_bias = np.fromfile(rf, dtype=np.float32, count=10)
            if len(fc6_bias) != 10:
                print("Can't read fc6_bias from file")
                return None, None, None, None, None, None, None, None
            print("Read fc6_bias from file")

        return conv1_weights, conv1_bias, conv3_weights, conv3_bias, conv5_weights, conv5_bias, fc6_weights, fc6_bias

    except FileNotFoundError:
        print("ERROR when opening parameter file!")
        return None, None, None, None, None, None, None, None


if __name__ == "__main__":
    NUM_TESTS = 10000  # Define NUM_TESTS value
    print("Starting LeNet")

    print("Parsing MNIST images")
    images = parse_mnist_images("images.bin")

    print("Parsing MNIST labels")
    labels = parse_mnist_labels("labels.bin")

    print("Parsing parameters")
    conv1_weights, conv1_bias, conv3_weights, conv3_bias, conv5_weights, conv5_bias, fc6_weights, fc6_bias = parse_parameters(
        "params.bin")

    conv1_weights = conv1_weights.reshape((6, 1, 5, 5))
    conv3_weights = conv3_weights.reshape((16, 6, 5, 5))
    conv5_weights = conv5_weights.reshape((120, 16, 5, 5))
    fc6_weights = fc6_weights.reshape((10, 120, 1, 1))

    # Perform inference loop on each test image
    print("Running inference")
    num_correct = 0

    for k in range(NUM_TESTS):
        print(f"Save the people | count {k}")
        image = get_image(images, k)
        # Perform inference steps
        conv1_output = convolution1(image, conv1_weights, conv1_bias)
        relu1_output = relu1(conv1_output)

        pool2_output = max_pooling2(relu1_output)
        relu2_output = relu2(pool2_output)

        conv3_output = convolution3(relu2_output, conv3_weights, conv3_bias)
        relu3_output = relu3(conv3_output)

        pool4_output = max_pooling4(relu3_output)
        relu4_output = relu4(pool4_output)

        conv5_output = convolution5(relu4_output, conv5_weights, conv5_bias)
        relu5_output = relu5(conv5_output)

        fc6_input = relu5_output.reshape((120, 1, 1))
        fc6_output = fc6(fc6_input, fc6_weights, fc6_bias)

        # Calculate prediction based on final output
        prediction = np.argmax(fc6_output)

        # Compare prediction with ground truth label and update accuracy count
        if prediction == labels[k]:
            num_correct += 1

        if k % 10 == 0:
            print(f"Test Image: {k}", end=" ")
            if prediction == labels[k]:
                print()
            else:
                print("(WRONG)")

    # Calculate accuracy
    accuracy = num_correct / NUM_TESTS * 100
    print(f"\nAccuracy = {accuracy:.2f}%")