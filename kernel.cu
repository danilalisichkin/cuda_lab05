#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#define MASK_WIDTH 3
#define MASK_RADIUS (MASK_WIDTH / 2)
#define THREADS_PER_BLOCK 32

bool loadImage(const std::string& filename, int& width, int& height, std::vector<unsigned char>& data, bool& isGrayscale) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    std::string header;
    file >> header;
    if (header == "P6") {
        std::cout << "Loaded RGB image\n";
        isGrayscale = false;
    }
    else if (header == "P5") {
        std::cout << "Loaded greyscale image\n";
        isGrayscale = true;
    }
    else {
        return false;
    }

    file >> width >> height;
    int maxval;
    file >> maxval;
    file.get();

    int numChannels = isGrayscale ? 1 : 3;
    data.resize(width * height * numChannels);
    file.read(reinterpret_cast<char*>(data.data()), data.size());
    return true;
}

bool saveImage(const std::string& filename, int width, int height, const std::vector<unsigned char>& data, bool isGrayscale) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    file << (isGrayscale ? "P5" : "P6") << "\n";
    file << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    return true;
}

void compareImages(const std::vector<unsigned char>& img1, const std::vector<unsigned char>& img2) {
    bool match = true;
    for (size_t i = 0; i < img1.size(); ++i) {
        if (img1[i] != img2[i]) {
            match = false;
            std::cout << "Pixel is not equal[" << i << "]: img1 = " << static_cast<int>(img1[i]) << ", img2 = " << static_cast<int>(img2[i]) << "\n";
            break;
        }
    }
    if (match) {
        std::cout << "Images are equal.\n";
    }
    else {
        std::cout << "Images are not equal.\n";
    }
}

void applyMinMaxFilterCpu(const std::vector<unsigned char>& input, std::vector<unsigned char>& output, int width, int height, bool isMinPass) {
    int channels = 3;
    for (int y = MASK_RADIUS; y < height - MASK_RADIUS; ++y) {
        for (int x = MASK_RADIUS; x < width - MASK_RADIUS; ++x) {
            for (int c = 0; c < channels; ++c) {
                int pixelIndex = (y * width + x) * channels + c;
                int result = isMinPass ? 255 : 0;
                for (int dy = -MASK_RADIUS; dy <= MASK_RADIUS; ++dy) {
                    for (int dx = -MASK_RADIUS; dx <= MASK_RADIUS; ++dx) {
                        int neighborIndex = ((y + dy) * width + (x + dx)) * channels + c;
                        int neighborPixel = input[neighborIndex];
                        result = isMinPass ? std::min(result, neighborPixel) : std::max(result, neighborPixel);
                    }
                }
                output[pixelIndex] = static_cast<unsigned char>(result);
            }
        }
    }
}

void applyMinMaxFilterGsCpu(const std::vector<unsigned char>& input, std::vector<unsigned char>& output, int width, int height, bool isMinPass) {
    for (int y = MASK_RADIUS; y < height - MASK_RADIUS; ++y) {
        for (int x = MASK_RADIUS; x < width - MASK_RADIUS; ++x) {
            int pixelIndex = y * width + x;
            int result = isMinPass ? 255 : 0;
            for (int dy = -MASK_RADIUS; dy <= MASK_RADIUS; ++dy) {
                for (int dx = -MASK_RADIUS; dx <= MASK_RADIUS; ++dx) {
                    int neighborIndex = (y + dy) * width + (x + dx);
                    int neighborPixel = input[neighborIndex];
                    result = isMinPass ? std::min(result, neighborPixel) : std::max(result, neighborPixel);
                }
            }
            output[pixelIndex] = static_cast<unsigned char>(result);
        }
    }
}

__global__ void applyMinMaxFilterGpu(unsigned char* input, unsigned char* output, int width, int height, bool isMinPass) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channels = 3;

    if (x >= MASK_RADIUS && x < width - MASK_RADIUS && y >= MASK_RADIUS && y < height - MASK_RADIUS) {
        for (int c = 0; c < channels; ++c) {
            int pixelIndex = (y * width + x) * channels + c;
            int result = isMinPass ? 255 : 0;
            for (int dy = -MASK_RADIUS; dy <= MASK_RADIUS; ++dy) {
                for (int dx = -MASK_RADIUS; dx <= MASK_RADIUS; ++dx) {
                    int neighborIndex = ((y + dy) * width + (x + dx)) * channels + c;
                    int neighborPixel = input[neighborIndex];
                    result = isMinPass ? min(result, neighborPixel) : max(result, neighborPixel);
                }
            }
            output[pixelIndex] = static_cast<unsigned char>(result);
        }
    }
}

__global__ void applyMinMaxFilterGsGpu(unsigned char* input, unsigned char* output, int width, int height, bool isMinPass) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= MASK_RADIUS && x < width - MASK_RADIUS && y >= MASK_RADIUS && y < height - MASK_RADIUS) {
        int pixelIndex = y * width + x;
        int result = isMinPass ? 255 : 0;
        for (int dy = -MASK_RADIUS; dy <= MASK_RADIUS; ++dy) {
            for (int dx = -MASK_RADIUS; dx <= MASK_RADIUS; ++dx) {
                int neighborIndex = (y + dy) * width + (x + dx);
                int neighborPixel = input[neighborIndex];
                result = isMinPass ? min(result, neighborPixel) : max(result, neighborPixel);
            }
        }
        output[pixelIndex] = static_cast<unsigned char>(result);
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << msg << " : " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int processImage(std::string path, bool isGsPass) {
    int width, height;
    bool isGrayscale;
    std::vector<unsigned char> image;
    std::string mode;

    if (!loadImage(path, width, height, image, isGrayscale)) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    if (isGsPass != isGrayscale) {
        std::cout << "Image processing modes are different!\n";
        return -1;
    }
    else {
        mode = isGrayscale ? "GS" : "RGB";
    }
        
    // CPU
    std::string cpuFile = mode + "_CPU";
    std::string gpuFile = mode + "_GPU";


    std::vector<unsigned char> outputCPU(image.size());
    std::vector<unsigned char> outputGPU(image.size());
    std::vector<unsigned char> tempBufferCPU(image.size());

    auto startCPU = std::chrono::high_resolution_clock::now();
    applyMinMaxFilterGsCpu(image, tempBufferCPU, width, height, false);
    applyMinMaxFilterGsCpu(tempBufferCPU, outputCPU, width, height, true);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationCPU = endCPU - startCPU;
    std::cout << mode << "\tCPU time: " << durationCPU.count() << " s\n";

    // GPU

    unsigned char* d_input, * d_output, * tempBufferGPU;
    checkCudaError(cudaMalloc(&d_input, image.size()), "Failed to allocate device input memory");
    checkCudaError(cudaMalloc(&d_output, image.size()), "Failed to allocate device output memory");
    checkCudaError(cudaMalloc(&tempBufferGPU, image.size()), "Failed to allocate tempBufferGPU buffer");
    cudaMemcpy(d_input, image.data(), image.size(), cudaMemcpyHostToDevice);

    dim3 blockSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 gridSize((width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (height + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    auto startGPU = std::chrono::high_resolution_clock::now();
    applyMinMaxFilterGsGpu << <gridSize, blockSize >> > (d_input, tempBufferGPU, width, height, false);
    checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize device");
    applyMinMaxFilterGsGpu << <gridSize, blockSize >> > (tempBufferGPU, d_output, width, height, true);
    checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize device");
    auto endGPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationGPU = endGPU - startGPU;
    std::cout << mode << "\tGPU time: " << durationGPU.count() << " s\n";

    checkCudaError(cudaMemcpy(outputGPU.data(), d_output, image.size(), cudaMemcpyDeviceToHost),
        "Failed copy data from device output to output file");

    compareImages(outputCPU, outputGPU);

    saveImage(cpuFile, width, height, outputCPU, isGrayscale);
    saveImage(gpuFile, width, height, outputGPU, isGrayscale);

    checkCudaError(cudaFree(d_input), "Failed to free memory used by d_input");
    checkCudaError(cudaFree(d_output), "Failed to free memory used by d_output");
    checkCudaError(cudaFree(tempBufferGPU), "Failed to free memory used by tempBufferGPU");

    return 0;
}

int main() {
    return processImage("C:\\Users\\danila\\Downloads\\gs.pgm", true) == 0
        ? processImage("C:\\Users\\danila\\Downloads\\rgb.ppm", false)
        : -1;
}