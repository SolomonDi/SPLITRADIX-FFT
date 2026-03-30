#pragma comment(lib, "cufft.lib")
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "Decomposition.h"   


using complex = cuComplex;

#define CUDA_CHECK(x) do { if((x) != cudaSuccess) { std::cout << "CUDA Error: " << cudaGetErrorString(x) << "\n"; exit(1); } } while(0)
#define CUFFT_CHECK(x) do { if((x) != CUFFT_SUCCESS) { std::cout << "CUFFT Error\n"; exit(1); } } while(0)

constexpr float PI = 3.14159265358979323846f;
constexpr int BLOCK = 256;


__global__ void reshape(complex* out, complex* in, int r, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = r * m; if (idx >= N) return;
    int i = idx / m, j = idx % m;
    out[j * r + i] = in[idx];
}

__global__ void inv_reshape(complex* out, complex* in, int r, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = r * m; if (idx >= N) return;
    int j = idx / r, i = idx % r;
    out[i * m + j] = in[idx];
}

__global__ void twiddle(complex* data, int r, int m, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int k = idx % r;      
    int j = idx / r;
    float angle = -2.0f * PI * k * j / (float)N;
    float c = cosf(angle), s = sinf(angle);
    complex a = data[idx];
    data[idx].x = a.x * c - a.y * s;
    data[idx].y = a.x * s + a.y * c;
}

__global__ void transpose(complex* out, complex* in, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
        out[x * height + y] = in[y * width + x];
}


void fft_stage(complex* d_data, int r, int m) {
    cufftHandle plan;
    int n[1] = { r };
    CUFFT_CHECK(cufftPlanMany(&plan, 1, n, nullptr, 1, r, nullptr, 1, r, CUFFT_C2C, m));
    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    cufftDestroy(plan);
}

void FFT_pipeline(complex* d_in_out, int N, const std::vector<uint>& factors, complex* d_temp) {
    complex* current = d_in_out;
    complex* tmp = d_temp;
    int currentN = N;

    for (size_t s = 0; s < factors.size(); ++s) {
        int r = factors[s];
        int m = currentN / r;
        int blocks = (currentN + BLOCK - 1) / BLOCK;

        reshape <<<blocks, BLOCK >> > (tmp, current, r, m);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(current, tmp);

        fft_stage(current, r, m);

        if (s != factors.size() - 1) {
            twiddle <<<blocks, BLOCK >> > (current, r, m, N);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        inv_reshape <<<blocks, BLOCK >> > (tmp, current, r, m);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(current, tmp);

        currentN = m;
    }

   
    if (current != d_in_out) {
        CUDA_CHECK(cudaMemcpy(d_in_out, current, N * sizeof(complex), cudaMemcpyDeviceToDevice));
    }
}

__host__ int main(void) {


    const int WIDTH = 8000;   
    const int HEIGHT = 6000;

    Decomposition row_dec(WIDTH);
    Decomposition col_dec(HEIGHT);
    row_dec.print();
    col_dec.print();

    complex* d_data = nullptr, * d_trans = nullptr, * d_temp = nullptr;

    CUDA_CHECK(cudaMalloc(&d_data, (size_t)WIDTH * HEIGHT * sizeof(complex)));
    CUDA_CHECK(cudaMalloc(&d_trans, (size_t)WIDTH * HEIGHT * sizeof(complex)));
    CUDA_CHECK(cudaMalloc(&d_temp, std::max(WIDTH, HEIGHT) * sizeof(complex)));




    std::vector<complex> h(WIDTH * HEIGHT, make_cuComplex(1.0f, 0.0f));
    CUDA_CHECK(cudaMemcpy(d_data, h.data(), (size_t)WIDTH * HEIGHT * sizeof(complex), cudaMemcpyHostToDevice));




    for (int y = 0; y < HEIGHT; ++y) {
        complex* row = d_data + (size_t)y * WIDTH;
        FFT_pipeline(row, WIDTH, row_dec.factors, d_temp);
    }




    dim3 block(32, 32);
    dim3 grid((WIDTH + 31) / 32, (HEIGHT + 31) / 32);
    transpose << <grid, block >> > (d_trans, d_data, WIDTH, HEIGHT);  
    CUDA_CHECK(cudaGetLastError());          
    CUDA_CHECK(cudaDeviceSynchronize());   
 

 
   
    for (int y = 0; y < WIDTH; ++y) {
        complex* trow = d_trans + (size_t)y * HEIGHT;
        FFT_pipeline(trow, HEIGHT, col_dec.factors, d_temp);
    }



    CUDA_CHECK(cudaMemcpy(h.data(), d_trans, (size_t)WIDTH * HEIGHT * sizeof(complex), cudaMemcpyDeviceToHost));
    float dc = sqrtf(h[0].x * h[0].x + h[0].y * h[0].y);
    std::cout << "2d DC component (" << WIDTH << "×" << HEIGHT << ") = " << dc
        << " (waiting " << (WIDTH * HEIGHT) << ")\n";

    cudaFree(d_data);
    cudaFree(d_trans);
    cudaFree(d_temp);

    return 0;
}