#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "conv_layer_3.h"


float relu_test(float input) {
    return (input > 0) ? input : 0;
}

void convolution5_sw(float input[16][5][5], float weights[120][16][5][5],
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


void relu5_sw(float input[120][1][1], float output[120][1][1]) {
    for(int i = 0; i < 120; i++)
        output[i][0][0] = relu_test(input[i][0][0]);
}



bool compare_outputs(float output1[120][1][1], float output2[120][1][1]) {
    for (int channel = 0; channel < 120; channel++) {
        if (fabs(output1[channel][0][0] - output2[channel][0][0]) > 1e-5) {
            return false;
        }
    }
    return true;
}

void print_output(float output[120][1][1]) {
    for (int channel = 0; channel < 120; channel++) {
        printf("%f ", output[channel][0][0]);
    }
    printf("\n");
}

int main() {
    float input_image[16][5][5];
    float weights[120][16][5][5];
    float biases[120];

    float output_hardware[120][1][1];
    float output_software_a[120][1][1];
    float output_software_final[120][1][1];

    // Initialize input, weights, and biases with random values
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 5; j++) {
			for(int k = 0; k < 5; k++) {
                input_image[i][j][k] = (float)(rand() % 100) / 100.0 - 0.5;
            }
        }
    }

    for(int channel = 0; channel < 120; channel++) {
    	for(int m = 0; m < 16; m++){
    		for(int kernel_row = 0; kernel_row < 5; kernel_row++) {
    			for(int kernel_col = 0; kernel_col < 5; kernel_col++) {
    				weights[channel][m][kernel_row][kernel_col] = (float)(rand() % 100) / 100.0 - 0.5;
            	}
            }
        }
    }

    for(int channel = 0; channel < 120; channel++) {
        biases[channel] = (float)(rand() % 100) / 100.0 - 0.5;
    }

    /* Tests */
    convolution5_sw(input_image, weights, biases, output_software_a);
    relu5_sw(output_software_a, output_software_final);

    convolution5_hw(input_image, weights, biases, output_hardware);

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
    print_output(output_software_final);
    for (int row = 0; row < 120; row++) {
        printf("%f ", output_software_final[row][0][0]);
    }
    printf("\n");

    printf("[TEST_BENCH] Output Hardware: ");
    print_output(output_hardware);
    for (int row = 0; row < 120; row++) {
        printf("%f ", output_hardware[row][0][0]);
    }
    printf("\n");

    fflush(stdout);

    return 0;
}
