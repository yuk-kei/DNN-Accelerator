#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

#define M 16
#define K 16
#define N 16

void matmul_plain(int A[M][K], int B[K][N], int AB[M][N]) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i][k] * B[k][j];
            }
            AB[i][j] = sum;
        }
    }
}


// Optimized matrix multiplication function
void matmul_optimized(int *A, int *B, int *AB) {
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=data0
    #pragma HLS INTERFACE s_axilite port=A bundle=ctrl
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=data1
    #pragma HLS INTERFACE s_axilite port=B bundle=ctrl
    #pragma HLS INTERFACE m_axi port=AB offset=slave bundle=data2
    #pragma HLS INTERFACE s_axilite port=AB bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    // Local arrays for matrix data
    int a_local[M][K];
    int b_local[K][N];
    int ab_local[M][N];

    // Copy data from pointers to local arrays
    // Loop for A
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            #pragma HLS PIPELINE
            a_local[i][j] = A[i * K + j];
        }
    }

    // Loop for B
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE
            b_local[i][j] = B[i * N + j];
        }
    }

    // Call to plain matrix multiplication
    matmul_plain(a_local, b_local, ab_local);

    // Copy result back to AB
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE
            AB[i * N + j] = ab_local[i][j];
        }
    }
}


