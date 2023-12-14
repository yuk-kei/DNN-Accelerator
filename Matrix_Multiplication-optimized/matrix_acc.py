import numpy as np
import pynq
from pynq import Overlay
import time

class MatrixMultiplier:
    def __init__(self, overlay_file):
        self.overlay = Overlay(overlay_file)
        self.matrixmult_inst = self.overlay.Matrixmul_0

    def multiply_fpga(self, array_a, array_b):
        M, K = array_a.shape
        K, N = array_b.shape
        array_c = pynq.allocate(shape=(M, N), dtype=np.int32)
        
        # Sync to device
        array_a.sync_to_device()
        array_b.sync_to_device()
        array_c.sync_to_device()

        # Setup FPGA
        self.matrixmult_inst.mmio.write_reg(0x10, array_a.physical_address)
        self.matrixmult_inst.mmio.write_reg(0x1c, array_b.physical_address)
        self.matrixmult_inst.mmio.write_reg(0x28, array_c.physical_address)


        start_time = time.time()
        self.matrixmult_inst.write(0x00, 1)
        while (self.matrixmult_inst.read(0x00) & 0x4) == 0:
            pass

        array_c.sync_from_device()
        fpga_time = time.time() - start_time

        return array_c, fpga_time

    @staticmethod
    def multiply_numpy(array_a, array_b):
        start_time = time.time()
        result = np.matmul(array_a, array_b)
        numpy_time = time.time() - start_time

        return result, numpy_time

    def test_performance(self, array_a, array_b):
        # Time the FPGA implementation
        result_fpga, fpga_time = self.multiply_fpga(array_a, array_b)


        # Time the NumPy implementation 
        result_numpy, numpy_time = MatrixMultiplier.multiply_numpy(array_a, array_b)
        print(f"      FPGA Result:\n{result_fpga}")
        print(f"      NumPy Result:\n{result_numpy}")

        # Print results
        print(f"      FPGA Time: {fpga_time:.6f} seconds")
        print(f"      NumPy Time: {numpy_time:.6f} seconds")

        return result_fpga, result_numpy, fpga_time, numpy_time

def main():
    M, N, K = 16, 16, 16  # Fixed dimensions for matrices
    num_tests = 2        # Number of times to test
    passed_test = 0      # Number of tests passed

    overlay_file = "/home/ubuntu/workspace/DNN-Accelerator/Matrix_Multiplication-optimized/matrix_mult.bit"
    matrix_multiplier = MatrixMultiplier(overlay_file)

    total_time_fpga = 0
    total_time_numpy = 0

    for i in range(num_tests):
        # Generate random matrices
        print (f"Starting test case {i+1} of {num_tests}")
        array_a = pynq.allocate(shape=(M, K), dtype=np.int32)
        array_b = pynq.allocate(shape=(K, N), dtype=np.int32)
        array_a[:] = np.random.randint(0, 100, size=(M, K), dtype=np.int32)
        array_b[:] = np.random.randint(0, 100, size=(K, N), dtype=np.int32)
        print(f"Matrix A:\n{array_a}")
        print(f"Matrix B:\n{array_b}")
        
        # Perform test
        
        result_fpga, result_numpy, time_fpga, time_numpy = matrix_multiplier.test_performance(array_a, array_b)
        if np.array_equal(result_fpga, result_numpy):
            print("\nResults are identical!")
            passed_test += 1
        else:
            print("\nResults differ!")
        total_time_fpga += time_fpga
        total_time_numpy += time_numpy
    # Calculate and print average times
    avg_time_fpga = total_time_fpga / num_tests
    avg_time_numpy = total_time_numpy / num_tests
    print(f"Test cases passed: {passed_test}/{num_tests}")
    print(f"\nAverage FPGA Time: {avg_time_fpga:.6f} seconds")
    print(f"Average NumPy Time: {avg_time_numpy:.6f} seconds")
    print(f"Speedup: {avg_time_numpy / avg_time_fpga:.2f}x")
    print(f"Speed up percentage: {((avg_time_numpy / avg_time_fpga) - 1) * 100:.2f}%")
    

if __name__ == "__main__":
    main()