const int M = 16;
const int K = 16;
const int N = 16;

void Matrixmul(int A[M][K], int B[K][N], int AB[M][N]) {
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=data0
    #pragma HLS INTERFACE s_axilite register port=A bundle=ctrl
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=data1
    #pragma HLS INTERFACE s_axilite register port=B bundle=ctrl
    #pragma HLS INTERFACE m_axi port=AB offset=slave bundle=data2
    #pragma HLS INTERFACE s_axilite register port=AB bundle=ctrl
    #pragma HLS INTERFACE s_axilite register port=return bundle=ctrl

    int A_local[M][K];
    int B_local[K][N];

    #pragma HLS ARRAY_PARTITION variable=A_local type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=B_local type=complete dim=1

    // Copy data from A and B to local arrays
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            #pragma HLS PIPELINE II=1
            A_local[i][k] = A[i][k];
        }
    }

    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            B_local[k][j] = B[k][j];
        }
    }

    // Matrix multiplication with array partitioning
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            int ABij = 0;
            for (int k = 0; k < K; k++) {
                ABij += A_local[i][k] * B_local[k][j];
            }
            AB[i][j] = ABij;
        }
    }
}
