#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "fully_conn_layer.h"



void fully_connected6_sw(float input[120][1][1], float weights[10][120][1][1], float bias[10], float output[10]) {
	int n, c;
    for(n = 0; n < 10; n++) {
        output[n] = 0;
        for(c = 0; c < 120; c++){
            output[n] += weights[n][c][0][0] * input[c][0][0];
        }
        output[n] += bias[n];
    }
}


bool compare_outputs(float output1[10], float output2[10]) {
    for (int channel = 0; channel < 10; channel++) {
        if (fabs(output1[channel] - output2[channel]) > 1e-5) {
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
    float input_image[120][1][1];
    float weights[10][120][1][1];
    float biases[10];

    float output_hardware[10];
    float output_software[10];

    // Initialize input, weights, and biases with random values
	for(int i = 0; i < 120; i++) {
		input_image[i][0][0] = (float)(rand() % 100) / 100.0 - 0.5;
    }

    for(int channel = 0; channel < 10; channel++) {
        for(int kernel_row = 0; kernel_row < 120; kernel_row++) {
            weights[channel][kernel_row][0][0] = (float)(rand() % 100) / 100.0 - 0.5;
        }
    }

    for(int channel = 0; channel < 10; channel++) {
        biases[channel] = (float)(rand() % 100) / 100.0 - 0.5;
    }

    /* Tests */
    fully_connected6_sw(input_image, weights, biases, output_software);

    fully_connected6_hw(input_image, weights, biases, output_hardware);

    // Compare software and hardware outputs
    bool is_equal = compare_outputs(output_software, output_hardware);

    // Print comparison results
    if (is_equal) {
        printf("Software and hardware outputs are equal.\n");
    } else {
        printf("Software and hardware outputs are NOT equal.\n");
    }

    // Optional: Print outputs for detailed comparison
    printf("[TEST_BENCH] Output Software: ");
    // print_output(output_software_final);
    for (int row = 0; row < 10; row++) {
        printf("%f ", output_software[row]);
    }
    printf("\n");

    printf("[TEST_BENCH] Output Hardware: ");
   // print_output(output_hardware);
    for (int row = 0; row < 10; row++) {
        printf("%f ", output_hardware[row]);
    }
    printf("\n");

    fflush(stdout);

    return 0;
}