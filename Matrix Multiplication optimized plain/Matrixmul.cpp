const int M = 16;
const int K = 16;
const int N = 16;

void Matrixmul(int A[M][K], int B[K][N], int AB[M][N]){
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=data0
#pragma HLS INTERFACE s_axilite register port=A bundle=ctrl
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=data1
#pragma HLS INTERFACE s_axilite register port=B bundle=ctrl
#pragma HLS INTERFACE m_axi port=AB offset=slave bundle=data2
#pragma HLS INTERFACE s_axilite register port=AB bundle=ctrl
#pragma HLS INTERFACE s_axilite register port=return bundle=ctrl
	for(int i=0; i<M; i++){
		for(int j=0; j<N; j++){
			int ABij = 0;
			for(int k=0; k<K; k++){
				ABij += A[i][k] * B[k][j];
			}
			AB[i][j] = ABij;
		}
	}
}
