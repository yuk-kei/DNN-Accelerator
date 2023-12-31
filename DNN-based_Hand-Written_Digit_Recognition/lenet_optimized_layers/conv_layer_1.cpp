#include "conv_layer_1.h"

float relu(float input) {
    return (input > 0)? input:0;
}

// Convolution Layer 1
void convolution1_hw(float input[1][32][32], float weights[6][1][5][5], float bias[6], float output[6][14][14]) {
    #pragma HLS INTERFACE m_axi      depth=1024   port=input   offset=slave bundle=data0
    #pragma HLS INTERFACE m_axi      depth=150    port=weights offset=slave bundle=data1
    #pragma HLS INTERFACE m_axi      depth=6      port=bias    offset=slave bundle=data2
    #pragma HLS INTERFACE m_axi      depth=4704   port=output  offset=slave bundle=data3
    #pragma HLS INTERFACE s_axilite  register port=return

	/* Copy inputs to Local Buffers */
	float conv_input[1][32][32];
	float conv_weights[6][1][5][5];
	float conv_bias[6];
	float conv_output[6][28][28];
	float relu1_output[6][28][28];
	float pooling_output[6][14][14];
	float relu2_output[6][14][14];

	float sum = 0.0;

	for(int i = 0; i < 32; i++) {
        #pragma HLS pipeline II=1
		for(int j = 0; j < 32; j++) {
            #pragma HLS unroll FACTOR=32
			conv_input[0][i][j] = input[0][i][j];
		}
	}

	for(int i = 0; i < 6; i++) {
        #pragma HLS pipeline II=1
		for(int j = 0; j < 5; j++) {
			for(int k = 0; k < 5; k++) {
            #pragma HLS unroll FACTOR=5
				conv_weights[i][0][j][k] = weights[i][0][j][k];
			}
		}
	}

	for(int i = 0; i < 6; i++) {
    #pragma HLS unroll FACTOR=6
		conv_bias[i] = bias[i];
	}


    /* Convolution 1 */
    for(int co = 0; co < 6; co++) {
        for(int h = 0; h < 28; h++) {
            for(int w = 0; w < 28; w++) {
            #pragma HLS unroll FACTOR = 4
                sum = 0.0;
                for(int i = 0, m = 0; i <  5; i++, m++) {
                    for(int j = 0, n = 0; j <  5; j++, n++) {
                        sum += conv_weights[co][0][m][n] * conv_input[0][h+i][j + w];
                    }
                }
                conv_output[co][h][w] = sum + conv_bias[co];
            }
        }
    }

    /* ReLU 1 */
    for(int i = 0; i < 6; i++) {
		for(int j = 0; j < 28; j++) {
        #pragma HLS pipeline II=1
			for(int k = 0; k < 28; k++) {
            #pragma HLS unroll FACTOR=28
				relu1_output[i][j][k] = relu(conv_output[i][j][k]);
			}
		}
    }

    float max_value = 0.0;
    int c;

    /* Pooling 2 */
    for(int c = 0; c < 6; c++) {
		for(int h = 0; h < 14; h++) {
        #pragma HLS pipeline II=1
			for(int w = 0; w < 14; w++) {
            #pragma HLS unroll FACTOR=14
				max_value=-1000000000000.0;
				for(int i = 0; i < 2; i++) {
					for(int j = 0;j < 2; j++) {
						max_value = (max_value > relu1_output[c][i + h * 2][w * 2 + j]) ? max_value:relu1_output[c][i+h*2][w*2 + j];
					}
				}
                
				pooling_output[c][h][w] = max_value;
			}
		}
    }

    /* ReLU 2 */
    for(int i = 0; i < 6; i++) {
		for(int j = 0; j < 14; j++) {
        #pragma HLS pipeline II=1
			for(int k = 0; k < 14; k++) {
            #pragma HLS unroll FACTOR=14
				relu2_output[i][j][k] = relu(pooling_output[i][j][k]);
			}
		}
    }

    /* Copying data back to output */
    for(int i = 0; i < 6; i++) {
		for(int j = 0; j < 14; j++) {
        #pragma HLS pipeline II=1
			for(int k = 0; k < 14; k++) {
            #pragma HLS unroll FACTOR=14
				output[i][j][k] = relu2_output[i][j][k];
			}
		}

	}
}