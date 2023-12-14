// Originally written for UIUC ECE 527 Fall 2017
// University of Illinois, Urbana-Champaign
// Author - Sitao Huang, Ashutosh Dhar
// Revised Fall 2020 by Mang Yu, UIUC ECE
// Revised Fall 2023 by Sitao Huang, for UCI EECS 298
// Demo code of LeNet Convolutional Neural Network
// *****************************************
// Before running the code make that the following binary data files are under
// the same folder as the executable file.
// 1. images.bin: 10,000 test images
// 2. labels.bin: class labels for each test image
// 3. params.bin: weights of the LeNet CNN
// This code demos inference on the MNIST dataset with a LeNet CNN
// Provided network parameters have been training already and should give an 
// accuracy of 98.39% when running on all the 10,000 images. 

#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <vector>
#include <array>
#include <chrono>

// Max number of test samples in LeNet is 10,000
// You can reduce this for testing/debugging
// Final evaluation must use all 10,000 test samples
#define NUM_TESTS 10000

// Uncomment the following line to disable prints during testing and measure
// the execution time for running tests
// #define BENCHMARK

using namespace std;
using namespace std::chrono;

//Static allocation of test images
unsigned char images[NUM_TESTS*28*28];
unsigned char labels[NUM_TESTS];

//Static allocation of network parameters and their outputs
float image[1][32][32] = {0};
float conv1_weights[6][1][5][5] = {0};
float conv1_bias[6] = {0};
float conv1_output[6][28][28] = {0};

float pool2_output[6][14][14] = {0};

float conv3_weights[16][6][5][5] = {0};
float conv3_bias[16] = {0};
float conv3_output[16][10][10] = {0};

float pool4_output[16][5][5] = {0};

float conv5_weights[120][16][5][5] = {0};
float conv5_bias[120] = {0};
float conv5_output[120][1][1] = {0};

float fc6_weights[10][120][1][1] = {0};
float fc6_bias[10] = {0};
float fc6_output[10] = {0};



// ************************************************//
// Start declaration of layer computation functions//
// ************************************************//

// Start function definitions of different layers
inline float relu(float input) {
    return (input > 0)? input:0;
}

// Convolution Layer 1
void convolution1(float input[1][32][32], float weights[6][1][5][5], 
                  float bias[6], float output[6][28][28]) {
    for(int co = 0; co < 6; co++)
        for(int h = 0; h < 28; h++)
            for(int w = 0; w < 28; w++)
            {
                float sum = 0;
                for(int i = h, m = 0; i < (h + 5); i++, m++)
                {
                    for(int j = w, n = 0; j < (w + 5); j++, n++)
                        sum += weights[co][0][m][n] * input[0][i][j];
                }
                output[co][h][w] = sum + bias[co];
            }
}

// ReLU Layer 1
void relu1(float input[6][28][28], float output[6][28][28]) {
    for(int i = 0; i < 6; i++)
        for(int j = 0; j < 28; j++)
            for(int k = 0; k < 28; k++)
                output[i][j][k] = relu(input[i][j][k]);
}

// Pooling Layer 2
void max_pooling2(float input[6][28][28],float output[6][14][14]) {
    for(int c = 0;c < 6; c++)
        for(int h = 0; h < 14; h++)
            for(int w = 0; w < 14; w++)
            {
                float max_value=-1000000000000.0;
                for(int i = h*2; i < h*2+2; i++)
                {
                    for(int j = w*2;j < w*2+2; j++)
                        max_value = (max_value > input[c][i][j]) ? 
                                                    max_value:input[c][i][j];
                }
                output[c][h][w] = max_value;

            }
}

// ReLU Layer 2
void relu2(float input[6][14][14], float output[6][14][14]) {
    for(int i = 0; i < 6; i++)
        for(int j = 0; j < 14; j++)
            for(int k = 0; k < 14; k++)
                output[i][j][k] = relu(input[i][j][k]);
}

// Convolution Layer 3
void convolution3(float input[6][14][14], float weights[16][6][5][5], 
                  float bias[16], float output[16][10][10]) {
    for(int co = 0; co < 16; co++)
        for(int h = 0; h < 10; h++)
            for(int w = 0; w < 10; w++) {
                    float sum = 0;
                    for(int i = h, m = 0; i < (h+5); i++, m++) {
                        for(int j = w, n = 0; j < (w+5); j++, n++)
                            for (int ci = 0; ci < 6; ci++)
                                sum += weights[co][ci][m][n] * input[ci][i][j];
                    }
                    output[co][h][w] = sum + bias[co];
            }
}

// ReLU Layer 3
void relu3(float input[16][10][10], float output[16][10][10]) {
    for(int i = 0; i < 16; i++)
        for(int j = 0; j < 10; j++)
            for(int k = 0; k < 10; k++)
                output[i][j][k] = relu(input[i][j][k]);
}

// Pooling Layer 4
void max_pooling4(float input[16][10][10],float output[16][5][5]) {
    for(int c = 0;c < 16; c++)
        for(int h = 0; h < 5; h++)
            for(int w = 0; w < 5; w++) {
                float max_value=-1000000000000.0;
                for(int i = h*2; i < h*2+2; i++) {
                    for(int j = w*2; j < w*2+2; j++)
                        max_value = (max_value > input[c][i][j]) ? 
                                                    max_value:input[c][i][j];
                }
                output[c][h][w] = max_value;
            }
}

// ReLU Layer 4
void relu4(float input[16][5][5], float output[16][5][5]) {
    for(int i = 0; i < 16; i++)
        for(int j = 0; j < 5; j++)
            for(int k = 0; k < 5; k++)
                output[i][j][k] = relu(input[i][j][k]);
}

// Convolution Layer 5
void convolution5(float input[16][5][5], float weights[120][16][5][5], 
                  float bias[120], float output[120][1][1]) {
    for(int co = 0; co < 120; co++) {
        float sum = 0;
        for(int i = 0, m = 0; i < 5; i++, m++) {
            for(int j = 0, n = 0; j < 5; j++, n++) {
                for (int ci = 0; ci < 16; ci++)
                    sum += weights[co][ci][m][n] * input[ci][i][j];
            }
        }
        output[co][0][0] = sum + bias[co];
    }
}

// ReLU Layer 5
void relu5(float input[120][1][1], float output[120][1][1]) {
    for(int i = 0; i < 120; i++)
        output[i][0][0] = relu(input[i][0][0]);
}

// Fully connected Layer 6
void fc6(const float input[120][1][1], const float weights[10][120][1][1], 
         const float bias[10], float output[10]) {
    for(int n = 0; n < 10; n++) {
        output[n] = 0;
        for(int c = 0; c < 120; c++) {
            output[n] += weights[n][c][0][0] * input[c][0][0];
        }
        output[n]+=bias[n];
    }
}

// ReLU Layer 6
void relu6(float input[10], float output[10]) {
    for(int i = 0; i < 10; i++)
        output[i] = relu(input[i]);
}
// ************************************************//
// End declaration of layer computation functions  //
// ************************************************//

// Parse MNIST test images
int parse_mnist_images(string filename, unsigned char *images) {
    unsigned int header[4];
    ifstream rf(filename, ios::in | ios::binary);
    if(!rf) {
        cout << "ERROR when opening MNIST images data file!" << endl;
        return 1;
    } else {
        cout << "Opened MNIST images data file" << endl;
    }

    if (!rf.read((char*)header,  sizeof(unsigned int)*4)) {
        cout << "Can't read header from file" << endl;
    } else {
        cout << "Read header from file" << endl;
    }

    if (!rf.read((char*)images,  sizeof(unsigned char)*NUM_TESTS*28*28)) {
        cout << "Can't read images from file" << endl;
    } else {
        cout << "Read images from file" << endl;
    }

    rf.close();
    return 0;
}

// Parse MNIST test image labels
int parse_mnist_labels(string filename, unsigned char *labels) {

    unsigned int header[2];

    ifstream rf(filename, ios::in | ios::binary);
    if(!rf) {
        cout << "ERROR when opening MNIST label data file!" << endl;
        return 1;
    } else {
        cout << "Opened MNIST label data file" << endl;
    }

    if (!rf.read((char*)header,  sizeof(unsigned int)*2)) {
        cout << "Can't read header from file" << endl;
    } else {
        cout << "Read header from file" << endl;
    }

    if (!rf.read((char*)labels,  sizeof(unsigned char)*NUM_TESTS)) {
        cout << "Can't read labels from file" << endl;
    } else {
        cout << "Read labels from file" << endl;
    }

    rf.close();
	return 0;
}

// Parse parameter file and load it in to the arrays
int parse_parameters(string filename) {
    ifstream rf(filename, ios::in | ios::binary);
    if(!rf) {
        cout << "ERROR when opening parameter file!" << endl;
        return 1;
    } else {
        cout << "Opened parameter file" << endl;
    }

    if (!rf.read((char*)***conv1_weights,  sizeof(float)*150)) {
        cout << "Can't read conv1_weights from file" << endl;
    } else {
        cout << "Read conv1_weights from file" << endl;
    }

    if (!rf.read((char*)conv1_bias,  sizeof(float)*6)) {
        cout << "Can't read conv1_bias from file" << endl;
    } else {
        cout << "Read conv1_bias from file" << endl;
    }

    if (!rf.read((char*)***conv3_weights,  sizeof(float)*2400)) {
        cout << "Can't read conv3_weights from file" << endl;
    } else {
        cout << "Read conv3_weights from file" << endl;
    }

    if (!rf.read((char*)conv3_bias,  sizeof(float)*16)) {
        cout << "Can't read conv3_bias from file" << endl;
    } else {
        cout << "Read conv3_bias from file" << endl;
    }

    if (!rf.read((char*)***conv5_weights,  sizeof(float)*48000)) {
        cout << "Can't read conv5_weights from file" << endl;
    } else {
        cout << "Read conv5_weights from file" << endl;
    }

    if (!rf.read((char*)conv5_bias,  sizeof(float)*120)) {
        cout << "Can't read conv5_bias from file" << endl;
    } else {
        cout << "Read conv5_bias from file" << endl;
    }


    if (!rf.read((char*)***fc6_weights,  sizeof(float)*1200)) {
        cout << "Can't read fc6_weights from file" << endl;
    } else {
        cout << "Read fc6_weights from file" << endl;
    }

    if (!rf.read((char*)fc6_bias,  sizeof(float)*10)) {
        cout << "Can't read fc6_bias from file" << endl;
    } else {
        cout << "Read fc6_bias from file" << endl;
    }

    rf.close();
    return 0;
}


// Fetch a single image to be processed.
//
void get_image(unsigned char *images, unsigned int idx,float image[1][32][32]){
    for(int i = 0; i < 32; i++)
        for(int j = 0; j < 32; j++) {
            if (i < 2 || i > 29 || j < 2 || j > 29)
                image[0][i][j] = -1.0;
            else
                image[0][i][j] = images[idx*28*28 + (i-2)*28 + j-2] 
                                                          / 255.0 * 2.0 - 1.0;
        }
}

int main(int argc, char **argv)
{
    cout << "Starting LeNet" << endl;

    cout << "Parsing MNIST images" << endl;
    parse_mnist_images("images.bin", images);

    cout << "Parsing MNIST labels" << endl;
    parse_mnist_labels("labels.bin", labels);

    cout << "Parsing parameters" << endl;
    parse_parameters("params.bin");

    // for (int i = 0; i < 10; i++)
    //     cout << fc6_bias[i] << " ";

    cout << "Running inference" << endl;
    int num_correct = 0;

    // starting time
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    for (int k = 0; k < NUM_TESTS; k++)
    {
    	// Get test image from dataset
        get_image(images, k, image);

        // Begin inference here.
        convolution1(image, conv1_weights, conv1_bias, conv1_output);
        relu1(conv1_output, conv1_output);

        max_pooling2(conv1_output, pool2_output);
        relu2(pool2_output, pool2_output);

        convolution3(pool2_output, conv3_weights, conv3_bias, conv3_output);
        relu3(conv3_output, conv3_output);

        max_pooling4(conv3_output, pool4_output);
        relu4(pool4_output, pool4_output);

        convolution5(pool4_output, conv5_weights, conv5_bias, conv5_output);
        relu5(conv5_output, conv5_output);

        fc6(conv5_output, fc6_weights, fc6_bias, fc6_output);
        // Inference ends here.

        // Index of the largest output is result
        // Check which output was largest.
        unsigned char result = 0;
        float p = -1000000.0;
        for(int i = 0; i < 10; i++) {
            if(fc6_output[i] > p) {
                p = fc6_output[i];
                result = i;
            }
        }


#ifndef BENCHMARK
        // Allow these prints when NOT profiling and benchmarking times
        if(k % 10 == 0) cout << "Test Image: ";
        cout << k;
#endif

        if(result == labels[k]) {
            num_correct++;
#ifndef BENCHMARK
            std::cout << " ";

        } else {
            std::cout << "(WRONG) ";
#endif
        }

#ifndef BENCHMARK
        if(k % 10 == 9) cout << endl;
#endif

    }
    // ending time
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

#ifdef BENCHMARK
    cout << endl;
    cout << "Total Execution Time:   " << time_span.count() << " seconds (" << 
                                        NUM_TESTS << " images)" << endl;
    cout << "Average Time per Image: " << time_span.count() / NUM_TESTS <<
                                                      " seconds" << endl;
#endif
    cout << endl << "Accuracy = " << float(num_correct)/NUM_TESTS * 100.0 << 
                                                            "%" << std::endl;
    std::cout << "Press any key to continue..." << std::endl;
    // Wait for user input
    std::cin.get();
    return 0;
}
