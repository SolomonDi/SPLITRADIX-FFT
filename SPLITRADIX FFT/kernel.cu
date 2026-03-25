#pragma comment(lib, "cufft.lib")

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <utility>

#include "Decomposition.h"

using complex = cuComplex;

#define CUDA_CHECK(x) if((x)!=cudaSuccess){ \
    std::cout<<"CUDA Error: "<<cudaGetErrorString(x)<<"\n"; exit(1); }

#define CUFFT_CHECK(x) if((x)!=CUFFT_SUCCESS){ \
    std::cout<<"CUFFT Error\n"; exit(1); }

constexpr float PI = 3.14159265358979323846f;
constexpr int BLOCK = 256;


__global__ void reshape(complex* out, complex* in, int r, int m) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = r * m;

    if (idx >= N) return;

    int i = idx / m;
    int j = idx % m;

    out[j * r + i] = in[idx];
}

__global__ void inv_reshape(complex* out, complex* in, int r, int m) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = r * m;

    if (idx >= N) return;

    int j = idx / r;
    int i = idx % r;

    out[i * m + j] = in[idx];
}




__global__ void twiddle(complex* data, int r, int m, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int i = idx / m;
    int j = idx % m;

    float angle = -2.0f * PI * i * j / N;

    float c = cosf(angle);
    float s = sinf(angle);

    complex a = data[idx];

    complex res;
    res.x = a.x * c - a.y * s;
    res.y = a.x * s + a.y * c;

    data[idx] = res;
}



void fft_stage(complex* d_data, int r, int m) {

    cufftHandle plan;

    int n[1] = { r };

    CUFFT_CHECK(cufftPlanMany(
        &plan,
        1,
        n,
        nullptr, 1, r,
        nullptr, 1, r,
        CUFFT_C2C,
        m
    ));

    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    cufftDestroy(plan);
}




void FFT_pipeline(complex* d_data, int N, const std::vector<uint>& factors) {

    complex* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, N * sizeof(complex)));

    int currentN = N;

    for (size_t s = 0; s < factors.size(); ++s) {

        int r = factors[s];
        int m = currentN / r;

        int blocks = (currentN + BLOCK - 1) / BLOCK;

        reshape<<<blocks, BLOCK>>>(d_temp, d_data, r, m);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(d_data, d_temp);

        fft_stage(d_data, r, m);

        if (s != factors.size() - 1) {
            twiddle<<<blocks, BLOCK>>>(d_data, r, m, N);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        inv_reshape <<<blocks, BLOCK>>>(d_temp, d_data, r, m);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(d_data, d_temp);

        currentN = m;
    }

    cudaFree(d_temp);
}




int main() {

    int N = 7000;

    Decomposition d(N);
    d.print();

    std::vector<complex> h(N);

    for (int i = 0; i < N; i++)
        h[i] = make_cuComplex(1.0f, 0.0f);

    complex* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(complex)));

    CUDA_CHECK(cudaMemcpy(d_data, h.data(),
        N * sizeof(complex), cudaMemcpyHostToDevice));

    FFT_pipeline(d_data, N, d.factors);

    CUDA_CHECK(cudaMemcpy(h.data(), d_data,
        N * sizeof(complex), cudaMemcpyDeviceToHost));

    float dc = sqrtf(h[0].x * h[0].x + h[0].y * h[0].y);

    std::cout << "DC = " << dc << " (expected " << N << ")\n";

    cudaFree(d_data);

    return 0;
}