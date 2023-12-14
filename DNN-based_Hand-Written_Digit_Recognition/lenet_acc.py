import numpy as np
import pynq
from pynq import Overlay
import time
import struct
import os
import subprocess

class LenetAcc:
    def __init__(self, image_filename, label_filename, parameter_filename, NUM_TESTS=10000):
        self.num_tests = NUM_TESTS
        self.images = self.parse_mnist_images(image_filename)
        self.labels = self.parse_mnist_labels(label_filename)
        (self.conv1_weights, 
         self.conv1_bias, 
         self.conv3_weights, 
         self.conv3_bias, 
         self.conv5_weights, 
         self.conv5_bias, 
         self.fc6_weights, 
         self.fc6_bias) = self.parse_parameters(parameter_filename)
        
        
    
    def parse_mnist_images(self, filename):
        """Parse MNIST images into numpy array
        
        Args:
            filename (str): The filename of the MNIST images file
        Returns:
            images (shape=(num * 28 * 28)): The parsed images      
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Image file not found: {filename}")

        with open(filename, 'rb') as f:
            try:
                print("Parsing MNIST images")
                header = np.fromfile(f, dtype=np.uint32, count=4)
                if len(header) != 4:
                    print("ERROR: Failed to read the header from images file")
                print("Read header from file")

                # Read labels data (unsigned chars)
                images = np.fromfile(f, dtype=np.uint8, count=self.num_tests * 28 * 28)
            except struct.error as e:
                raise ValueError(f"Error unpacking file {filename}: {e}")
            except ValueError as e:
                raise ValueError(f"Error reshaping images from file {filename}: {e}")
            print(images.shape)
        return images


    def parse_mnist_labels(self, filename):
        """Parse MNIST labels into numpy array
        
        Args:
            filename (str): The filename of the MNIST labels file
        Returns:
            labels (shape=(num, )): The parsed labels
        """
        
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Label file not found: {filename}")

        with open(filename, 'rb') as f:
                     
            try:
                print("Parsing MNIST labels")
                header_labels = np.fromfile(f, dtype=np.uint32, count=2)
                
                # Check if the header was read successfully
                if len(header_labels) != 2:
                    print("Can't read header from file")
                print("Read header from file")

                # Read labels data (unsigned chars)
                labels = np.fromfile(f, dtype=np.uint8, count=self.num_tests)
                print("Read labels from file")
                
                if labels.size != self.num_tests:
                    print("Can't read labels from file")
                    return None
                

            except ValueError as e:
                raise ValueError(f"Error reading labels from file {filename}: {e}")

        return labels

    def parse_parameters(self, filename):
        """Parse parameters from file
        
        Args:
            filename (str): The filename of the parameters file
        Returns:
            conv1_weights (shape=(6, 1, 5, 5)): The weights of the first convolution layer
            conv1_bias (shape=(6, )): The bias of the first convolution layer
            conv3_weights (shape=(16, 6, 5, 5)): The weights of the second convolution layer
            conv3_bias (shape=(16, )): The bias of the second convolution layer
            conv5_weights (shape=(120, 16, 5, 5)): The weights of the third convolution layer
            conv5_bias (shape=(120, )): The bias of the third convolution layer
            fc6_weights (shape=(10, 120, 1, 1)): The weights of the fully connected layer
            fc6_bias (shape=(10, )): The bias of the fully connected layer
        """        
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Parameter file not found: {filename}")

        with open(filename, 'rb') as f:
            try:
                print("Parsing parameters")
                conv1_weights = pynq.allocate((6, 1, 5, 5), dtype=np.float32)
                conv1_bias = pynq.allocate(6, dtype=np.float32)
                conv3_weights = pynq.allocate((16, 6, 5, 5), dtype=np.float32)
                conv3_bias = pynq.allocate(16, dtype=np.float32)
                conv5_weights = pynq.allocate((120, 16, 5, 5), dtype=np.float32)
                conv5_bias = pynq.allocate(120, dtype=np.float32)
                fc6_weights = pynq.allocate((10, 120, 1, 1), dtype=np.float32)
                fc6_bias = pynq.allocate(10, dtype=np.float32)

                conv1_weights[:] = np.fromfile(f, dtype=np.float32, count=6 * 1 * 5 * 5).reshape((6, 1, 5, 5))
                conv1_bias[:] = np.fromfile(f, dtype=np.float32, count=6)
                conv3_weights[:] = np.fromfile(f, dtype=np.float32, count=16 * 6 * 5 * 5).reshape((16, 6, 5, 5))
                conv3_bias[:] = np.fromfile(f, dtype=np.float32, count=16)
                conv5_weights[:] = np.fromfile(f, dtype=np.float32, count=120 * 16 * 5 * 5).reshape((120, 16, 5, 5))
                conv5_bias[:] = np.fromfile(f, dtype=np.float32, count=120)
                fc6_weights[:] = np.fromfile(f, dtype=np.float32, count=10 * 120 * 1 * 1).reshape((10, 120, 1, 1))
                fc6_bias[:] = np.fromfile(f, dtype=np.float32, count=10)
   
            except struct.error as e:
                raise ValueError(f"Error reading parameters from file {filename}: {e}")
            except ValueError as e:
                raise ValueError(f"Error reshaping parameters from file {filename}: {e}")

        return conv1_weights, conv1_bias, conv3_weights, conv3_bias, conv5_weights, conv5_bias, fc6_weights, fc6_bias


    def conv_layer_1(self, input, weights, bias, output, overlay, ip_name):
        """Convolution layer 1
        
        Args:
            input (shape=(1, 32, 32)): The input image
            weights (shape=(6, 1, 5, 5)): The weights of the convolution layer
            bias (shape=(6, )): The bias of the convolution layer
            output (shape=(6, 14, 14)): The output of the convolution layer
        Returns:
            output (shape=(6, 14, 14)): The output of the convolution layer

        """
        
        #overlay = Overlay("/home/ubuntu/workspace/DNN-Accelerator/DNN-based_Hand-Written_Digit_Recognition/lenet_files/conv_layer_1.bit")
        # ip_name = "convolution1_hw_0"
        overlay = overlay
        layer_1_inst = getattr(overlay, ip_name)
        
        output = pynq.allocate(output, dtype=np.float32)
        # Sync to device
        input.sync_to_device()
        weights.sync_to_device()
        bias.sync_to_device()
        output.sync_to_device()

        # Setup FPGA
        layer_1_inst.mmio.write_reg(0x10, input.physical_address)
        layer_1_inst.mmio.write_reg(0x14, 0)
        layer_1_inst.mmio.write_reg(0x1c, weights.physical_address)
        layer_1_inst.mmio.write_reg(0x20, 0)
        layer_1_inst.mmio.write_reg(0x28, bias.physical_address)
        layer_1_inst.mmio.write_reg(0x2c, 0)
        layer_1_inst.mmio.write_reg(0x34, output.physical_address)
        layer_1_inst.mmio.write_reg(0x38, 0)
        
        layer_1_inst.write(0x00, 1)
        while (layer_1_inst.read(0x00) & 0x4) == 0:
            pass
        output.sync_from_device()

    def fully_connected_layer(self, input, weights, bias, output, overlay, ip_name):
        
        
        return output
    # def conv_layer_2(self, image):
    
    # def conv_layer_3(self, image):
    
    # def fully_connected_layer(self, image):
        
    def get_image(self, images, idx):
        """Get the specific image from the dataset and reshape it to (1, 32, 32)

        Args:
            images (shape=(num, 28, 28)):
            idx (int): The index of the image to extract 

        Returns:
            image (shape=(1, 32, 32)): The extracted image
        """
        # Allocate memory for the image with PYNQ in the shape of (1, 32, 32)
        # image = pynq.allocate(shape=(1, 32, 32), dtype=np.float32)

        # # Check if images is a flat array or already a 2D image
        # if images.ndim > 1:
        #     # Assuming images is a 3D array of shape (num_images, height, width)
        #     single_image = images[idx]
        # else:
        #     # Assuming images is a flat array where each image is 28x28 sequentially
        #     single_image = images[idx * 28 * 28 : (idx + 1) * 28 * 28].reshape(28, 28)

        # # Normalize, scale, and pad the image
        # for i in range(32):
        #     for j in range(32):
        #         if i < 2 or i > 29 or j < 2 or j > 29:
        #             image[0, i, j] = -1.0
        #         else:
        #             # Ensure that the assignment is a single float value
        #             pixel_value = single_image[i - 2, j - 2]
        #             image[0, i, j] = pixel_value / 255.0 * 2.0 - 1.0

        # return image
    
        # Extract the specific 28x28 image and reshape
        extracted_image = images[idx * 28 * 28: (idx + 1) * 28 * 28].reshape(28, 28)

        # Normalize and scale the image
        normalized_image = extracted_image / 255.0 * 2.0 - 1.0

        # Pad the image to 32x32
        padded_image = np.pad(normalized_image, pad_width=2, mode='constant', constant_values=-1.0)

        # Allocate memory for the image with PYNQ in the shape of (1, 32, 32)
        image = pynq.allocate((1, 32, 32), dtype=np.float32)
        
        # Copy the processed image data to the allocated buffer
        image[0, :, :] = padded_image

        return image

    def run_single_image(self, idx):
        """Run the accelerator on a single image
        
        Args:
            idx (int): The index of the image to run
        Returns:
            label (int): The predicted label
        """
        
        conv1_overlay = Overlay("/home/ubuntu/workspace/DNN-Accelerator/DNN-based_Hand-Written_Digit_Recognition/lenet_files/conv_layer_1.bit")
        
        ip_list = ["convolution1_hw_0", "convolution3_hw_0", "convolution5_hw_0", "fully_connected_hw_0"]
        
        # Get the image
        
        input_image = self.get_image(self.images, idx)
        start_time = time.time()
        conv_layer_1 = self.conv_layer_1(input=input_image, weights=self.conv1_weights, bias=self.conv1_bias, output=(6, 14, 14), overlay=conv1_overlay, ip_name=ip_list[0])
        print(conv_layer_1.shape)
        print("image time: ", time.time() - start_time)
        conv2_overlay = Overlay("/home/ubuntu/workspace/DNN-Accelerator/DNN-based_Hand-Written_Digit_Recognition/lenet_files/conv_layer_2.bit")
        conv_layer_2 = self.conv_layer_1(input=conv_layer_1, weights=self.conv3_weights, bias=self.conv3_bias, output=(16, 5, 5), overlay=conv2_overlay, ip_name=ip_list[1])
        print(conv_layer_2.shape)
        conv3_overlay = Overlay("/home/ubuntu/workspace/DNN-Accelerator/DNN-based_Hand-Written_Digit_Recognition/lenet_files/conv_layer_3.bit")
        conv_layer_3 = self.conv_layer_1(input=conv_layer_2, weights=self.conv5_weights, bias=self.conv5_bias, output=(120, 1, 1), overlay=conv3_overlay, ip_name=ip_list[2])
        print(conv_layer_3.shape)
        
        fn_overlay = Overlay("/home/ubuntu/workspace/DNN-Accelerator/DNN-based_Hand-Written_Digit_Recognition/lenet_files/fully_conn_layer.bit")
        
        
        # conv_layer_3 = 
        # Run the accelerator

        # return label
        
        
def run_lenet_cpp(file_name):
    # Make sure the compiled program exists
    if not os.path.isfile(file_name):
        raise FileNotFoundError("lenet executable not found.")

    start_time = time.time()
    result = subprocess.run([f"{file_name}"], capture_output=True, text=True)
    end_time = time.time()
    print(result.stdout)
    print("time: ", end_time - start_time)

    if result.returncode != 0:
        raise RuntimeError(f"lenet execution failed: {result.stderr}")

    return end_time - start_time, result.stdout

     

def main():
    
    lenet_baseline= "/home/ubuntu/workspace/DNN-Accelerator/DNN-based_Hand-Written_Digit_Recognition/lenet_bases/lenet"
    image_filename = "/home/ubuntu/workspace/DNN-Accelerator/DNN-based_Hand-Written_Digit_Recognition/lenet_files/images.bin"
    label_filename = "/home/ubuntu/workspace/DNN-Accelerator/DNN-based_Hand-Written_Digit_Recognition/lenet_files/labels.bin"
    parameter_filename = "/home/ubuntu/workspace/DNN-Accelerator/DNN-based_Hand-Written_Digit_Recognition/lenet_files/params.bin"
    print("start")

    lenet_acc = LenetAcc(image_filename=image_filename, 
                         label_filename=label_filename, 
                         parameter_filename=parameter_filename)
    lenet_acc.run_single_image(0)
    # run_lenet_cpp(lenet_baseline)
    
    print("Done")
    
    
    
if __name__ == "__main__":
    main()
    
    
    