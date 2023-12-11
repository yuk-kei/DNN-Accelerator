#include "fully_conn_layer.h"

void fully_connected6_hw(float input[120][1][1], float weights[10][120][1][1], float bias[10], float output[10]) {
    #pragma HLS INTERFACE m_axi      depth=120   port=input   offset=slave bundle=data0
    #pragma HLS INTERFACE m_axi      depth=1200  port=weights offset=slave bundle=data1
    #pragma HLS INTERFACE m_axi      depth=10    port=bias    offset=slave bundle=data2
    #pragma HLS INTERFACE m_axi      depth=10    port=output  offset=slave bundle=data3
    #pragma HLS INTERFACE s_axilite  port=return


    /* Copy inputs to Local Buffers */
	float fully_conn_input[120][1][1];
	float fully_conn_weights[10][120][1][1];
	float fully_conn_bias[10];
	float fully_conn_output[10];


	for(int i = 0; i < 120; i++) {
    #pragma HLS unroll factor=120
		fully_conn_input[i][0][0] = input[i][0][0];
	}

	for(int i = 0; i < 10; i++) {
    #pragma HLS pipeline II=1
		for(int j = 0; j < 120; j++) {
        #pragma HLS unroll factor=120
			fully_conn_weights[i][j][0][0] = weights[i][j][0][0];
		}
	}

	for(int i = 0; i < 10; i++) {
    #pragma HLS unroll factor=10
		fully_conn_bias[i] = bias[i];
	}

    /* Fully Connected Layer 6 */ 
    for(int n = 0; n < 10; n++) {
    #pragma HLS pipeline II=1
        output[n] = 0;
        for(int c = 0; c < 120; c++){
        #pragma HLS unroll factor=120
            fully_conn_output[n] += fully_conn_weights[n][c][0][0] * fully_conn_input[c][0][0];
        }
        fully_conn_output[n]+=fully_conn_bias[n];
    }

    /* Copying data back to output */
	for(int i = 0; i < 10; i++) {
    #pragma HLS unroll factor=10
		output[i] = fully_conn_output[i];
	}

}