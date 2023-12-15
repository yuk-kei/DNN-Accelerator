import numpy as np
import pynq
from pynq import Overlay
import time

class MatrixMultiplier:
    def __init__(self, overlay_file_list, ip_list):
        self.overlay_list = overlay_file_list
        self.ip_list = ip_list
        self.overlay = Overlay(self.overlay_list[0])
        self.matrixmult_inst = getattr(self.overlay, ip_list[0])

    def multiply_fpga(self, array_a, array_b):
        """Multiply two matrices using the FPGA accelerator
        
        Arguments:
            array_a {pynq.ndarray} -- First matrix
            array_b {pynq.ndarray} -- Second matrix
        
        Returns:
            array_c {pynq.ndarray} -- Result matrix
            fpga_time {float} -- Time taken to perform multiplication
        """
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
    
    def change_overlay(self, overlay_file, ip_name):
        self.overlay = Overlay(overlay_file)
        self.matrixmult_inst = getattr(self.overlay, ip_name)

    @staticmethod
    def multiply_numpy(array_a, array_b):
        """
        Multiply two matrices using NumPy
        
        Arguments:
            array_a {numpy.ndarray} -- First matrix
            array_b {numpy.ndarray} -- Second matrix
        
        Returns:
            array_c {numpy.ndarray} -- Result matrix
            numpy_time {float} -- Time taken to perform multiplication
        """
        start_time = time.time()
        result = np.matmul(array_a, array_b)
        numpy_time = time.time() - start_time

        return result, numpy_time

    def get_performance_all(self, array_a, array_b):
        # Time the FPGA implementation
        result_fpga_plain, fpga_plain_time = self.multiply_fpga(array_a, array_b)
        
        self.change_overlay(self.overlay_list[1], self.ip_list[1])
        
        result_fpga_opt, fpga_opt_time = self.multiply_fpga(array_a, array_b)
        # Time the NumPy implementation 
        result_numpy, numpy_time = MatrixMultiplier.multiply_numpy(array_a, array_b)
        
        # print(f"      NumPy Result:\n{result_numpy}")
        # print(f"      Plain Result:\n{result_fpga_plain}")
        # print(f"      Optimized Result:\n{result_fpga_opt}")

        # Print results
        print(f"      NumPy Time: {numpy_time:.6f} seconds")
        print(f"      Plain Time: {fpga_plain_time:.6f} seconds")
        print(f"      Optimized Time: {fpga_opt_time:.6f} seconds")

        return result_fpga_plain, result_fpga_opt, result_numpy, fpga_plain_time, fpga_opt_time, numpy_time

def main():
    M, N, K = 16, 16, 16  # Fixed dimensions for matrices
    num_tests = 2        # Number of times to test
    passed_test = 0      # Number of tests passed

    opt_overlay_file = "/home/ubuntu/workspace/DNN-Accelerator/Matrix_Multiplication-optimized/matrix_mult.bit"
    plain_overlay_file = "/home/ubuntu/workspace/DNN-Accelerator/Matrix_Multiplication-plain/MatrixMultiplication.bit"
    overlay_file_list = [opt_overlay_file, plain_overlay_file]
    ip_list = ["Matrixmul_0", "Matrixmul_0"]
    matrix_multiplier = MatrixMultiplier(overlay_file_list=overlay_file_list, ip_list=ip_list)
 
   
    test_matrix_list = []
    for i in range(num_tests):
        # Generate random matrices for testing
        print (f"Starting test case {i+1} of {num_tests}")
        array_a = pynq.allocate(shape=(M, K), dtype=np.int32)
        array_b = pynq.allocate(shape=(K, N), dtype=np.int32)
        array_a[:] = np.random.randint(0, 100, size=(M, K), dtype=np.int32)
        array_b[:] = np.random.randint(0, 100, size=(K, N), dtype=np.int32)
        print (f"Test case {i+1} of {num_tests} generated")
        print("-----------------------------------------------")
        print(f"Matrix A:\n{array_a}")
        print("-----------------------------------------------")
        print(f"Matrix B:\n{array_b}")
        test_matrix_list.append((array_a, array_b))
        print(f"-----------------------------------------------\n")
        
        
    # Perform test
    total_time_plain = 0
    total_time_opt = 0
    total_time_numpy = 0
    
    for i in range(num_tests):
        print (f"Starting test case {i+1} of {num_tests}")
        array_a, array_b = test_matrix_list[i]

        (result_fpga_plain, 
         result_fpga_opt, 
         result_numpy, 
         fpga_plain_time, 
         fpga_opt_time, 
         numpy_time) = matrix_multiplier.get_performance_all(array_a, array_b)    
        if np.allclose(result_fpga_plain, result_fpga_opt, result_numpy):
            passed_test += 1
        
        total_time_numpy += numpy_time
        total_time_plain += fpga_plain_time
        total_time_opt += fpga_opt_time
        
        print(f"-----------------------------------------------\n")


    # Calculate and print average times
    avg_time_plain = total_time_plain / num_tests
    avg_time_opt = total_time_opt / num_tests
    avg_time_numpy = total_time_numpy / num_tests
    print(f"Test cases passed: {passed_test}/{num_tests}")
    print(f"Average NumPy Time: {avg_time_numpy:.6f} seconds")
    print(f"Average Plain Time: {avg_time_plain:.6f} seconds")
    print(f"Average Optimized Time: {avg_time_opt:.6f} seconds")
    
    print(f"-----------------------------------------------")
    print(f"Plain acc Speedup compare to numpy: {avg_time_numpy / avg_time_plain:.2f}x")
    print(f"Optimized acc Speedup compare to numpy: {avg_time_numpy / avg_time_opt:.2f}x")
    print(f"Optimized acc Speedup compare to plain acc: {avg_time_plain / avg_time_opt:.2f}x")
    print(f"Plain acc speed up percentage: {((avg_time_numpy / avg_time_plain) - 1) * 100:.2f}%")
    print(f"Optimized acc Speed up percentage: {((avg_time_numpy / avg_time_opt) - 1) * 100:.2f}%")

    

if __name__ == "__main__":
    main()