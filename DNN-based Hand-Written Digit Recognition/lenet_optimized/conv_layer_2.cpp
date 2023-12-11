#include "conv_layer_2.h"


float relu(float input) {
    return (input > 0)? input:0;
}

// Layer 2 of Convolution 3
// Also includes Pooling 4 and ReLU 3,4
void convolution3_hw(float input[6][14][14], float weights[16][6][5][5], float bias[16], float output[16][5][5]) {
	#pragma HLS INTERFACE m_axi      depth=1176    port=input   offset=slave bundle=data0
	#pragma HLS INTERFACE m_axi      depth=2400    port=weights offset=slave bundle=data1
	#pragma HLS INTERFACE m_axi      depth=16      port=bias    offset=slave bundle=data2
	#pragma HLS INTERFACE m_axi      depth=400     port=output  offset=slave bundle=data3
	#pragma HLS INTERFACE s_axilite  port=return

	/* Copy inputs to Local Buffers */
	float conv_input[6][14][14];
	float conv_weights[16][6][5][5];
	float conv_bias[16];
	float conv_output[16][10][10];
	float relu3_output[16][10][10];
	float pooling4_output[16][5][5];
	float relu4_output[16][5][5];

	float sum = 0.0;

	for(int i = 0; i < 6; i++) {
        #pragma HLS pipeline II=1
		for(int j = 0; j < 14; j++) {
			for(int k = 0; k < 14; k++) {
            #pragma HLS unroll FACTOR=32
				conv_input[i][j][k] = input[i][j][k];
			}
		}
	}

	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 6; j++) {
			for(int k = 0; k < 5; k++) {
            #pragma HLS pipeline II=1
				for(int l = 0; l < 5; l++) {
                #pragma HLS unroll FACTOR=5
					conv_weights[i][j][k][l] = weights[i][j][k][l];
				}
			}
		}
	}

	for(int i = 0; i < 16; i++) {
    #pragma HLS unroll FACTOR=16
		conv_bias[i] = bias[i];
	}

	/* Convolution 3 */
	for(int co = 0; co < 16; co++) {
		for(int h = 0; h < 10; h++) {
        #pragma HLS pipeline II=1
			for(int w = 0; w < 10; w++) {
				sum = 0;
				for(int i = 0, m = 0; i < 5; i++, m++) {
					for(int j = 0, n = 0; j < 5; j++, n++) {
						for (int ci = 0; ci < 6; ci++) {
                        #pragma HLS unroll FACTOR=6
							sum += conv_weights[co][ci][m][n] * conv_input[ci][h+i][w+j];
						}
					}
				}
				conv_output[co][h][w] = sum + conv_bias[co];
			}
		}
	}

	/* ReLU 3 */
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 10; j++) {
        #pragma HLS pipeline II=1
			for(int k = 0; k < 10; k++) {
            #pragma HLS unroll FACTOR=10
				relu3_output[i][j][k] = relu(conv_output[i][j][k]);
			}
		}
	}

	float max_value = 0.0;

	/* Pooling 4 */
	for(int c = 0; c < 16; c++) {
		for(int h = 0; h < 5; h++) {
			for(int w = 0; w < 5; w++) {
            #pragma HLS pipeline
				max_value=-1000000000000.0;
				for(int i = 0; i < 2; i++) {
					for(int j = 0;j < 2; j++) {
						max_value = (max_value > relu3_output[c][h*2 + i][w*2 + j]) ? max_value:relu3_output[c][h*2 + i][w*2 + j];
					}
				}
				pooling4_output[c][h][w] = max_value;
			}
		}
	}

	/* ReLU 4 */
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 5; j++) {
        #pragma HLS pipeline II=1
			for(int k = 0; k < 5; k++) {
            #pragma HLS unroll FACTOR=5
				relu4_output[i][j][k] = relu(pooling4_output[i][j][k]);
			}
		}
	}

	/* Copying data back to output */
	for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 5; j++) {
        #pragma HLS pipeline II=1
			for(int k = 0; k < 5; k++) {
            #pragma HLS unroll FACTOR=5
				output[i][j][k] = relu4_output[i][j][k];
			}
		}
	}

}