#include "conv_layer_3.h"

float relu(float input) {
    return (input > 0)? input:0;
}

// Layer 3 of Convolution 5
// Also includes 5 ReLU 5
void convolution5_hw(float input[16][5][5], float weights[120][16][5][5], float bias[120], float output[120][1][1]) {
	#pragma HLS INTERFACE m_axi      depth=400    port=input   offset=slave bundle=data0
	#pragma HLS INTERFACE m_axi      depth=48000    port=weights offset=slave bundle=data1
	#pragma HLS INTERFACE m_axi      depth=120      port=bias    offset=slave bundle=data2
	#pragma HLS INTERFACE m_axi      depth=120    port=output  offset=slave bundle=data3
	#pragma HLS INTERFACE s_axilite  port=return

	/* Copy inputs to Local Buffers */
	float conv_inputs[16][5][5];
	float conv_weights[120][16][5][5];
	float conv_bias[120];
	float conv_output[120][1][1];
	float relu5_output[120][1][1];


	float sum = 0.0;

	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 5; j++) {
			for(int k = 0; k < 5; k++) {
			#pragma HLS unroll FACTOR=5
				conv_inputs[i][j][k] = input[i][j][k];
			}
		}
	}

	for(int i = 0; i < 120; i++) {
		for(int j = 0; j < 16; j++) {
			for(int k = 0; k < 5; k++) {
				for(int l = 0; l < 5; l++) {
				#pragma HLS unroll FACTOR=5
					conv_weights[i][j][k][l] = weights[i][j][k][l];
				}
			}
		}
	}

	for(int i = 0; i < 120; i++) {
	#pragma HLS unroll FACTOR=20
		conv_bias[i] = bias[i];
	}

	/* Convolution 5 */
	for(int co = 0; co < 120; co++) {
		sum = 0;
		for(int i = 0, m = 0; i < 5; i++, m++) {
			for(int j = 0, n = 0; j < 5; j++, n++) {
				for (int ci = 0; ci < 16; ci++) {
				#pragma HLS unroll FACTOR=8
					sum += conv_weights[co][ci][m][n] * conv_inputs[ci][i][j];
				}
			}
		}
		conv_output[co][0][0] = sum + conv_bias[co];
	}

	/* ReLU 5 */
	for(int i = 0; i < 120; i++) {
	#pragma HLS unroll FACTOR=120
		relu5_output[i][0][0] = relu(conv_output[i][0][0]);
	}

	for(int i = 0; i < 120; i++) {
	#pragma HLS unroll FACTOR=120
		output[i][0][0] = relu5_output[i][0][0];
	}

}
