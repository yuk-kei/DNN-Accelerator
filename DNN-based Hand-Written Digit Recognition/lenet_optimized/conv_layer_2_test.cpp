#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "conv_layer_2.h"


float relu_test(float input) {
    return (input > 0) ? input : 0;
}

void convolution3_sw(float input[6][14][14], float weights[16][6][5][5], 
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


void relu3_sw(float input[16][10][10], float output[6][10][10]) {
    for(int channel = 0; channel < 16; channel++) {
        for(int row = 0; row < 10; row++) {
            for(int col = 0; col < 10; col++) {
                output[channel][row][col] = relu_test(input[channel][row][col]);
            }
        }
    }
}



void max_pooling4_sw(float input[16][10][10], float output[16][5][5]) {
    for(int channel = 0; channel < 16; channel++) {
        for(int row = 0; row < 5; row++) {
            for(int col = 0; col < 5; col++) {
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


void relu4_sw(float input[16][5][5], float output[16][5][5]) {
    for(int channel = 0; channel < 16; channel++) {
        for(int row = 0; row < 5; row++) {
            for(int col = 0; col < 5; col++) {
                output[channel][row][col] = relu_test(input[channel][row][col]);
            }
        }
    }
}



bool compare_outputs(float output1[16][5][5], float output2[16][5][5]) {
    for (int channel = 0; channel < 16; channel++) {
        for (int row = 0; row < 5; row++) {
            for (int col = 0; col < 5; col++) {
                if (fabs(output1[channel][row][col] - output2[channel][row][col]) > 1e-5) {
                    return false;
                }
            }
        }
    }
    return true;
}

void print_output(float output[16][5][5]) {
    for (int channel = 0; channel < 16; channel++) {
        for (int row = 0; row < 5; row++) {
            for (int col = 0; col < 5; col++) {
                printf("%f ", output[channel][row][col]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main() {
    float input_image[6][14][14];
    float weights[16][6][5][5];
    float biases[16];
    float output_hardware[16][5][5];

    float output_software_a[16][10][10];
    float output_software_b[16][10][10];
    float output_software_c[16][5][5];
    float output_software_final[16][5][5];

    // Initialize input, weights, and biases with random values
	for(int i = 0; i < 6; i++) {
		for(int j = 0; j < 14; j++) {
			for(int k = 0; k < 14; k++) {
                input_image[i][j][k] = (float)(rand() % 100) / 100.0 - 0.5;
            }
        }
    }

    for(int channel = 0; channel < 16; channel++) {
        for(int kernel_row = 0; kernel_row < 6; kernel_row++) {
            for(int kernel_col = 0; kernel_col < 5; kernel_col++) {
                weights[channel][0][kernel_row][kernel_col] = (float)(rand() % 100) / 100.0 - 0.5;
            }
        }
    }

    for(int channel = 0; channel < 16; channel++) {
        biases[channel] = (float)(rand() % 100) / 100.0 - 0.5;
    }

    /* Tests */
    convolution3_sw(input_image, weights, biases, output_software_a);
    relu3_sw(output_software_a, output_software_b);
    max_pooling4_sw(output_software_b, output_software_c);
    relu4_sw(output_software_c, output_software_final);
    convolution3_hw(input_image, weights, biases, output_hardware);

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
    // print_output(output_software_final);
    for (int row = 0; row < 16; row++) {
        printf("%f ", output_software_final[row][0][5]);
    }
    printf("\n");

    printf("[TEST_BENCH] Output Hardware: ");
    // print_output(output_hardware);
    for (int row = 0; row < 16; row++) {
        printf("%f ", output_hardware[row][0][5]);
    }
    printf("\n");

    fflush(stdout);

    return 0;
}