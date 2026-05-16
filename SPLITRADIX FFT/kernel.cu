#pragma comment(lib, "cufft.lib")
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "Decomposition.h"

using complex = cuComplex;

#define CUDA_CHECK(x) do { if((x) != cudaSuccess) { std::cout << "CUDA Error: " << cudaGetErrorString(x) << "\n"; exit(1); } } while(0)
#define CUFFT_CHECK(x) do { if((x) != CUFFT_SUCCESS) { std::cout << "CUFFT Error\n"; exit(1); } } while(0)

constexpr float PI = 3.14159265358979323846f;
constexpr int BLOCK_1D = 256;
constexpr int TILE_DIM = 32;


__global__ void twiddle_batch(complex* data, int r, int m, int num_rows, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * num_rows) return;

    int elem_idx = idx % N;
    int segment_size = r * m;
    int local_idx = elem_idx % segment_size;

    int k = local_idx % r;
    int j = local_idx / r;

    float angle = -2.0f * PI * (k * j) / (float)segment_size;
    complex w = make_cuComplex(cosf(angle), sinf(angle));
    complex a = data[idx];

    data[idx].x = a.x * w.x - a.y * w.y;
    data[idx].y = a.x * w.y + a.y * w.x;
}

__global__ void transpose(complex* out, complex* in, int width, int height) {
    __shared__ complex tile[TILE_DIM][TILE_DIM + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * TILE_DIM + tx;
    int y = blockIdx.y * TILE_DIM + ty;

    if (x < width && y < height) {
        tile[ty][tx] = in[y * width + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + tx;
    y = blockIdx.x * TILE_DIM + ty;

    if (x < height && y < width) {
        out[y * height + x] = tile[tx][ty];
    }
}


__global__ void reshape_batch(complex* out, complex* in, int r, int m, int num_rows, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * num_rows) return;

    int row = idx / N;
    int elem_idx = idx % N;

    int segment_size = r * m;
    int segment_idx = elem_idx / segment_size;
    int local_idx = elem_idx % segment_size;

    int i = local_idx / m;
    int j = local_idx % m;

    int new_local_idx = j * r + i;
    int new_elem_idx = segment_idx * segment_size + new_local_idx;

    out[row * N + new_elem_idx] = in[idx];
}


__global__ void inv_reshape_batch(complex* out, complex* in, int r, int m, int num_rows, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * num_rows) return;

    int row = idx / N;
    int elem_idx = idx % N;

    int segment_size = r * m;
    int segment_idx = elem_idx / segment_size;
    int local_idx = elem_idx % segment_size;

    int j = local_idx / r;
    int i = local_idx % r;

    int new_local_idx = i * m + j;
    int new_elem_idx = segment_idx * segment_size + new_local_idx;

    out[row * N + new_elem_idx] = in[idx];
}

void fft_stage_batch(complex* d_data, int r, int N, int num_rows) {
    cufftHandle plan;
    int n[] = { r }; 
    int howmany = (N / r) * num_rows;

    CUFFT_CHECK(cufftPlanMany(&plan, 1, n,
        nullptr, 1, r,
        nullptr, 1, r,
        CUFFT_C2C, howmany));

    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    cufftDestroy(plan);
}



void FFT_pipeline_batch(complex* d_in_out, int N, const std::vector<uint>& factors,
    complex* d_temp, int num_rows) {
    complex* current = d_in_out;
    complex* tmp = d_temp;
    int currentN = N;

    int total_elements = N * num_rows;
    int blocks = (total_elements + BLOCK_1D - 1) / BLOCK_1D;

    for (size_t s = 0; s < factors.size(); ++s) {
        int r = factors[s];
        int m = currentN / r;

        reshape_batch << <blocks, BLOCK_1D >> > (tmp, current, r, m, num_rows, N);
        std::swap(current, tmp);

        fft_stage_batch(current, r, N, num_rows);

        if (s != factors.size() - 1) {
            twiddle_batch << <blocks, BLOCK_1D >> > (current, r, m, num_rows, N);
        }

        inv_reshape_batch << <blocks, BLOCK_1D >> > (tmp, current, r, m, num_rows, N);
        std::swap(current, tmp);

        currentN = m;
    }

    if (current != d_in_out) {
        size_t total_bytes = (size_t)N * num_rows * sizeof(complex);
        CUDA_CHECK(cudaMemcpy(d_in_out, current, total_bytes, cudaMemcpyDeviceToDevice));
    }
}

__host__ int main(void) {
    const int WIDTH = 12000;
    const int HEIGHT = 16000;
    const size_t TOTAL_SIZE = (size_t)WIDTH * HEIGHT;

    Decomposition row_dec(WIDTH);
    Decomposition col_dec(HEIGHT);
    row_dec.print();
    col_dec.print();

    complex* h_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned, TOTAL_SIZE * sizeof(complex)));

    for (size_t i = 0; i < TOTAL_SIZE; ++i) {
        h_pinned[i] = make_cuComplex(1.0f, 0.0f);
    }

    complex* d_data = nullptr;
    complex* d_trans = nullptr;
    complex* d_temp = nullptr;

    CUDA_CHECK(cudaMalloc(&d_data, TOTAL_SIZE * sizeof(complex)));
    CUDA_CHECK(cudaMalloc(&d_trans, TOTAL_SIZE * sizeof(complex)));
    CUDA_CHECK(cudaMalloc(&d_temp, TOTAL_SIZE * sizeof(complex)));

    CUDA_CHECK(cudaMemcpy(d_data, h_pinned, TOTAL_SIZE * sizeof(complex), cudaMemcpyHostToDevice));


    FFT_pipeline_batch(d_data, WIDTH, row_dec.factors, d_temp, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());


    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((WIDTH + TILE_DIM - 1) / TILE_DIM, (HEIGHT + TILE_DIM - 1) / TILE_DIM);
    transpose << <grid, block >> > (d_trans, d_data, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());


    FFT_pipeline_batch(d_trans, HEIGHT, col_dec.factors, d_temp, WIDTH);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_pinned, d_trans, TOTAL_SIZE * sizeof(complex), cudaMemcpyDeviceToHost));

    float dc = sqrtf(h_pinned[0].x * h_pinned[0].x + h_pinned[0].y * h_pinned[0].y);
    std::cout << "2D DC component (" << WIDTH << "*" << HEIGHT << ") = " << dc
        << " (waiting " << (long long)WIDTH * HEIGHT << ")\n";

    cudaFreeHost(h_pinned);
    cudaFree(d_data);
    cudaFree(d_trans);
    cudaFree(d_temp);

    return 0;
}
