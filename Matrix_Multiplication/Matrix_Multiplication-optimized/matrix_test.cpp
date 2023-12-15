#include <iostream>
#include <ctime>
#include <cstdlib>

const int M = 16;
const int K = 16;
const int N = 16;

// Function to generate random integer in the range [min, max]
int generateRandomInt(int min, int max) {
    return rand() % (max - min + 1) + min;
}

void Matrixmul(int A[M][K], int B[K][N], int AB[M][N]);

// Function to initialize a matrix with random values
void initializeMatrix(int matrix[][N], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = generateRandomInt(1, 10);  // Adjust the range as needed
        }
    }
}

// Function to print a matrix
void printMatrix(const int matrix[][N], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Function to multiply matrices (for reference)
void matrixMultiplicationReference(const int A[][K], const int B[][N], int result[][N]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int ABij = 0;
            for (int k = 0; k < K; k++) {
                ABij += A[i][k] * B[k][j];
            }
            result[i][j] = ABij;
        }
    }
}

int main() {
    srand(time(0));  // Seed for random number generation

    int A[M][K];
    int B[K][N];
    int AB[M][N];
    int referenceAB[M][N];

    // Generate random input matrices
    initializeMatrix(A, M, K);
    initializeMatrix(B, K, N);

    // Print input matrices
    std::cout << "Matrix A:" << std::endl;
    printMatrix(A, M, K);

    std::cout << "Matrix B:" << std::endl;
    printMatrix(B, K, N);

    // Call the Matrixmul function
    Matrixmul(A, B, AB);

    // Call the reference matrix multiplication function
    matrixMultiplicationReference(A, B, referenceAB);

    // Print output matrix
    std::cout << "Result Matrix (Matrixmul function):" << std::endl;
    printMatrix(AB, M, N);

    // Print reference output matrix
    std::cout << "Reference Result Matrix:" << std::endl;
    printMatrix(referenceAB, M, N);

    // Compare the result with the reference (for accuracy)
    bool accuracy = true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (AB[i][j] != referenceAB[i][j]) {
                accuracy = false;
                break;
            }
        }
        if (!accuracy) {
            break;
        }
    }

    // Print accuracy
    std::cout << "Accuracy: " << (accuracy ? "Correct" : "Incorrect") << std::endl;

    // Print execution time (you can use your own timing mechanism)
    // Note: This example uses the C++ clock() function for simplicity
    clock_t start_time = clock();
    // Call the Matrixmul function again for timing
    Matrixmul(A, B, AB);
    clock_t end_time = clock();
    double execution_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Execution Time: " << execution_time << " seconds" << std::endl;

    return 0;
}
