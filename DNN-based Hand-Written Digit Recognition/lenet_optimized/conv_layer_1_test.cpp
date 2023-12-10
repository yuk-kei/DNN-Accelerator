#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "conv_layer_1.cpp"


float relu_tb(float input) {
    return (input > 0) ? input : 0;
}

void relu1_sw(float input[6][28][28], float output[6][28][28]) {
    for(int channel = 0; channel < 6; channel++) {
        for(int row = 0; row < 28; row++) {
            for(int col = 0; col < 28; col++) {
                output[channel][row][col] = relu_tb(input[channel][row][col]);
            }
        }
    }
}

void relu2_sw(float input[6][14][14], float output[6][14][14]) {
    for(int channel = 0; channel < 6; channel++) {
        for(int row = 0; row < 14; row++) {
            for(int col = 0; col < 14; col++) {
                output[channel][row][col] = relu_tb(input[channel][row][col]);
            }
        }
    }
}

void max_pooling2_sw(float input[6][28][28], float output[6][14][14]) {
    for(int channel = 0; channel < 6; channel++) {
        for(int row = 0; row < 14; row++) {
            for(int col = 0; col < 14; col++) {
                float max_value = -INFINITY;
                for(int i = row * 2; i < row * 2 + 2; i++) {
                    for(int j = col * 2; j < col * 2 + 2; j++) {
                        max_value = fmax(max_value, input[channel][i][j]);
                    }
                }
                output[channel][row][col] = max_value;
            }
        }
    }
}

int convolution1_sw(float input[1][32][32], float weights[6][1][5][5], float bias[6], float output[6][28][28]) {
    for(int output_channel = 0; output_channel < 6; output_channel++) {
        for(int output_row = 0; output_row < 28; output_row++) {
            for(int output_col = 0; output_col < 28; output_col++) {
                float sum = 0.0;
                for(int kernel_row = 0; kernel_row < 5; kernel_row++) {
                    for(int kernel_col = 0; kernel_col < 5; kernel_col++) {
                        sum += weights[output_channel][0][kernel_row][kernel_col] * input[0][output_row + kernel_row][output_col + kernel_col];
                    }
                }
                output[output_channel][output_row][output_col] = sum + bias[output_channel];
            }
        }
    }
    return 0;
}

bool compare_outputs(float output1[6][14][14], float output2[6][14][14]) {
    for (int channel = 0; channel < 6; channel++) {
        for (int row = 0; row < 14; row++) {
            for (int col = 0; col < 14; col++) {
                if (fabs(output1[channel][row][col] - output2[channel][row][col]) > 1e-5) {
                    return false;
                }
            }
        }
    }
    return true;
}

int main() {
    float input_image[1][32][32];
    float weights[6][1][5][5];
    float biases[6];
    float output_hardware[6][14][14];

    float output_software_a[6][28][28];
    float output_software_b[6][28][28];
    float output_software_c[6][14][14];
    float output_software_final[6][14][14];

    // Initialize input, weights, and biases with random values
    for(int i = 0; i < 1; i++) {
        for(int j = 0; j < 32; j++) {
            for(int k = 0; k < 32; k++) {
                input_image[i][j][k] = (float)(rand() % 100) / 100.0 - 0.5;
            }
        }
    }

    for(int channel = 0; channel < 6; channel++) {
        for(int kernel_row = 0; kernel_row < 5; kernel_row++) {
            for(int kernel_col = 0; kernel_col < 5; kernel_col++) {
                weights[channel][0][kernel_row][kernel_col] = (float)(rand() % 100) / 100.0 - 0.5;
            }
        }
    }

    for(int channel = 0; channel < 6; channel++) {
        biases[channel] = (float)(rand() % 100) / 100.0 - 0.5;
    }

    /* Tests */
    convolution1_sw(input_image, weights, biases, output_software_a);
    relu1_sw(output_software_a, output_software_b);
    max_pooling2_sw(output_software_b, output_software_c);
    relu2_sw(output_software_c, output_software_final);
    convolution_hw(input_image, weights, biases, output_hardware);

    // Compare software and hardware outputs
    bool is_equal = compare_outputs(output_software_final, output_hardware);

    // Print comparison results
    if (is_equal) {
        printf("Software and hardware outputs are equal.\n");
    } else {
        printf("Software and hardware outputs are NOT equal.\n");
    }

    // Optional: Print outputs for detailed comparison
    printf("[TEST_BENCH] Output Software: ");
    for (int row = 0; row < 14; row++) {
        printf("%f ", output_software_final[1][row][7]);
    }
    printf("\n");

    printf("[TEST_BENCH] Output Hardware: ");
    for (int row = 0; row < 14; row++) {
        printf("%f ", output_hardware[1][row][7]);
    }
    printf("\n");

    fflush(stdout);

    return 0;
}
