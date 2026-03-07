#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include<cuComplex.h>
#include <cufft.h>
#pragma comment(lib, "cufft.lib")


#include<iostream>
#include"Decomposition.h"
#include<vector>
#include<cmath>

#include<complex>
#include<unordered_map>
#include<omp.h>
#include<mutex>
#include<atomic>
#include<utility>
#include <chrono>


constexpr const float M_PI = 3.14159274f;

using complex = cuComplex;

#define MakeComplex(re, im) make_cuComplex(re, im)

constexpr int BLOCK_SIZE = 512;
constexpr int BLOCK_2D = 32;
constexpr int NUM_STREAMS = 8;
constexpr int MAX_SHARED_MEM = 49152;
constexpr int MAX_REGISTERS = 65536;


__device__ __host__ inline complex addComplex(complex a, complex b) {

	return make_cuComplex(a.x + b.x, a.y + b.y);

}

__device__ __host__ inline complex mulComplex(complex a, complex b) {

	return make_cuComplex(a.x * b.x, a.y * b.y);
}

__device__ __host__ inline complex conjugate(complex a) {

	return make_cuComplex(a.x, -a.y);

}

template<class T>
class cuPMemory {

	T* ptr = nullptr;
	size_t m_size = {};

public:

	cuPMemory() = default;

	explicit cuPMemory(size_t elements) : m_size(elements) {

		cudaError_t error = cudaMallocHost(reinterpret_cast<void**>(&ptr), elements * sizeof(T));

		if (error != cudaSuccess) {

			std::cerr << "failed" << cudaGetErrorString(error) << std::endl;
			ptr = nullptr;
			m_size = 0;
		}

	}

	~cuPMemory() {
		if (ptr)
			cudaFreeHost(ptr);
	}

	cuPMemory(const cuPMemory&) = delete;
	cuPMemory& operator=(const cuPMemory&) = delete;


	cuPMemory(cuPMemory&& other) noexcept : ptr(other.ptr), m_size(other.m_size) {

		other.ptr = nullptr;
		other.m_size = 0;
	}

	cuPMemory& operator=(cuPMemory&& other) noexcept {

		if (this != other) {

			if (ptr) {

				cudaFreeHost(ptr);
			}

			ptr = other.ptr;
			m_size = other.m_size;
			other.ptr = nullptr;
			other.m_size = 0;
		}

		return *this;
	}

public:

	T* data() const { return this->ptr; }

	size_t size() const { return this->m_size; }

	bool isValid() const { return this->ptr != nullptr; }

	void release() {

		if (ptr) {

			cudaFreeHost(ptr);
			this->ptr = nullptr;
			this->m_size = 0;
		}

	}
};

class cuStreamPool {

	std::vector<cudaStream_t> streams;
	std::atomic<size_t> curr{};

public:

	cuStreamPool(size_t streamNums = NUM_STREAMS) {

		streams.reserve(streamNums);

		for (size_t i = 0; i < streamNums; ++i) {

			cudaStream_t stream;

			cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
			streams.emplace_back(stream);

		}

	}

	~cuStreamPool() {

		for (auto& stream : streams) {

			cudaStreamDestroy(stream);
		}
	}

public:


	cudaStream_t getStream() {

		size_t idx = curr.fetch_add(1, std::memory_order_relaxed);
		return streams[idx % streams.size()]; // robin round ("looped array") 

	}

	cudaStream_t getFixedStream(size_t idx) const {

		return streams[idx % streams.size()];
	}

	void syncAll() {

		for (auto& stream : streams) {
			cudaStreamSynchronize(stream);
		}
	}

	size_t threadsNum() const { return streams.size(); }
};

class DMAManager {

	cuStreamPool& stream_pool;

public:

	explicit DMAManager(cuStreamPool& pool) : stream_pool(pool) {}


	template<class T>
	void HostDeviceAsync(T* Ddev, T* Hsrc, size_t count, size_t stream_idx = 0) {

		cudaStream_t stream = stream_pool.getFixedStream(stream_idx);

		cudaMemcpyAsync(Ddev, Hsrc, count * sizeof(T), cudaMemcpyHostToDevice, stream);

		int idDevice;
		cudaGetDevice(&idDevice);
		cudaMemPrefetchAsync(Ddev, count * sizeof(T), idDevice, stream);
	}


	template<class T>
	void DeviceHostAsync(T* Hsrc, T* Ddev, size_t count, size_t stream_idx = 0) {


		cudaStream_t stream = stream_pool.getFixedStream(stream_idx);

		cudaMemcpyAsync(Hsrc, Ddev, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
	}


	//batch copy
	template<class T>
	void batchCopy(std::vector<std::pair<T*, T*>>& copy, size_t countInCopy, cudaMemcpyKind kind, size_t stream_ofset = 0) {

		for (size_t i = 0; i < copy.size(); ++i) {

			cudaStream_t stream = stream_pool.getFixedStream((stream_ofset + i) % stream_pool.threadsNum());

			cudaMemcpyAsync(copy[i].first, copy[i].second, countInCopy * sizeof(T), kind, stream);
		}
	}
};


__global__ void reshape2DVector(complex* out, complex* in, int rows, int cols, int stride) {

	int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

	if (idx < rows * cols) {

#pragma unroll 
		for (int i = 0; i < 4 && (idx + i) < rows * cols; ++i) {

			int pos = idx + i;
			int r = pos / cols;
			int c = pos % cols;
			out[pos] = in[r * stride + c];
		}
	}

}

__global__ void apllyTwiddlesFactors(complex* data, complex* twiddles, int rows, int colums) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int colum = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rows && colum < colums) {

		int idx = row * colums + colum;
		complex a = data[idx];
		complex b = twiddles[idx];

		data[idx].x = __fmaf_rn(a.x, b.x, -__fmul_rn(a.y, b.y));
		data[idx].y = __fmaf_rn(a.x, b.y, __fmul_rn(a.y, b.x));
	}
}

__global__ void transposeMatrix(complex* out, complex* in, int rows, int cols) {

	__shared__ complex tile[32][32 + 1];

	int x = blockIdx.x * 32 + threadIdx.x;
	int y = blockIdx.y * 32 + threadIdx.y;

	if (x < cols && y < rows) {

		tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
	}
	__syncthreads();

	x = blockIdx.y * 32 + threadIdx.x;
	y = blockIdx.x * 32 + threadIdx.y;

	if (x < rows && y < cols) {

		out[y * rows + x] = tile[threadIdx.x][threadIdx.y];

	}
}


class TwiddleCache {
private:

	struct CacheEntry {

		complex* device_ptr = nullptr;
		int nums_elem{};
		int key{};

		CacheEntry(complex* device_ptr_, int nums_elem_, int key_) : device_ptr(device_ptr_), nums_elem(nums_elem), key(key) {}

	};

	std::vector<CacheEntry> cache;

public:



	complex* get_twiddles(int N, int rows, int cols) {

		int key = N * 1000000 + rows * 1000 + cols;

		for (auto& entry : cache) {

			if (entry.key == key) {

				return entry.device_ptr;
			}

		}

		cuPMemory<complex> TwiddleHost(rows * cols);

#pragma omp parallel for collapse(2)
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {

				float angle = -2.f * M_PI * i * j / N;
				TwiddleHost.data()[i * cols + j] = MakeComplex(cosf(angle), sinf(angle));
			}
		}

		complex* twiddleDevice;
		cudaMalloc(reinterpret_cast<void**>(&twiddleDevice), rows * cols * sizeof(complex));
		cudaMemcpy(twiddleDevice, TwiddleHost.data(), rows * cols * sizeof(complex), cudaMemcpyHostToDevice);


		cache.emplace_back(twiddleDevice, rows * cols, key);

		return twiddleDevice;
	}

	~TwiddleCache() {

		for (auto& el : cache) {

			cudaFree(el.device_ptr);
		}
	}
};


class FFTCACHE {
private:

	std::unordered_map<int, cufftHandle> plans;
	std::unordered_map<int, void*> workAreas;

public:

	cufftHandle getPlan(int size, cudaStream_t stream = 0) {

		auto it = plans.find(size);
		if (it != plans.end()) {

			cufftSetStream(it->second, stream);

			return it->second;
		}

		cufftHandle plan;
		cufftCreate(&plan);

		cufftSetStream(plan, stream);

		size_t workSize = {};

		cufftMakePlan1d(plan, size, CUFFT_C2C, 1, &workSize);

		if (size >= 4096) {

			void* workArea;
			cudaMalloc(reinterpret_cast<void**>(&workArea), workSize);
			workAreas[size] = workArea;
			cufftSetWorkArea(plan, workArea);

		}

		plans[size] = plan;

		return plan;
	}

	~FFTCACHE() {

		for (auto& [size, plan] : plans) cufftDestroy(plan);

		for (auto& [size, ptr] : workAreas) cudaFree(ptr);
	}
};



class FFTProcessor {
private:
	int N;

	Decomposition decomp;
	cuStreamPool stream_pool;
	DMAManager dma_manager;
	TwiddleCache twiddle_cache;
	FFTCACHE fft_plans;

	complex* d_l = nullptr;
	complex* d_N = nullptr;
	complex* d_temp = nullptr;

	cudaEvent_t ev_start, ev_end;

	float totalTime{};

	int framePers{};

public:

	FFTProcessor(int size) : N(size), decomp(size), stream_pool(NUM_STREAMS), dma_manager(stream_pool) {

		decomp.print();

		int L_size = decomp.lrows * decomp.lcols;
		int N_size = decomp.InRows * decomp.InCols;

		cudaMalloc(&d_l, L_size * sizeof(complex));
		cudaMalloc(&d_N, N_size * sizeof(complex));
		cudaMalloc(&d_temp, N * sizeof(complex));

		int device_id;
		cudaGetDevice(&device_id);

		cudaMemLocation location;

		location.type = cudaMemLocationTypeDevice;
		location.id = device_id;

		for (int i = 0; i < 3; ++i) {

			cudaStream_t stream = stream_pool.getFixedStream(i);

			if (i == 0)
				cudaMemPrefetchAsync(d_l, L_size * sizeof(complex), location, 0, stream);

			if (i == 1)
				cudaMemPrefetchAsync(d_N, L_size * sizeof(complex), location, 0, stream);

			else
				cudaMemPrefetchAsync(d_temp, L_size * sizeof(complex), location, 0, stream);
		}

		stream_pool.syncAll();

		cudaEventCreate(&ev_start);
		cudaEventCreate(&ev_end);

	}


	~FFTProcessor() {

		cudaFree(d_l);
		cudaFree(d_N);
		cudaFree(d_temp);

		cudaEventDestroy(ev_start);
		cudaEventDestroy(ev_end);

	}

public:

	bool FFTExecute(complex* d_input, complex* d_out) {

		cudaEventRecord(ev_start, stream_pool.getFixedStream(0));

		int M1 = decomp.lrows;
		int M2 = decomp.lcols;
		int N1 = decomp.InRows;
		int N2 = decomp.InCols;


		//Reshape to L_Matrix
		dim3 block1d(BLOCK_SIZE);
		dim3 grid1d((N + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4));
		reshape2DVector << <grid1d, block1d, 0, stream_pool.getFixedStream(0) >> > (d_l, d_input, M1, M2, N2);

		cufftHandle plan_rows = fft_plans.getPlan(M2, stream_pool.getFixedStream(1));

		cufftExecC2C(plan_rows, d_l, d_l, CUFFT_FORWARD);

		complex* twiddles = twiddle_cache.get_twiddles(N, M1, M2);

		dim3 block2d(BLOCK_2D, BLOCK_2D);

		dim3 grid2d((M2 + BLOCK_2D - 1) / BLOCK_2D, (M1 + BLOCK_2D - 1) / BLOCK_2D);

		apllyTwiddlesFactors << <grid2d, block2d, 0, stream_pool.getFixedStream(2) >> > (
			d_l, twiddles, M1, M2
			);


		transposeMatrix << <grid2d, block2d, 0, stream_pool.getFixedStream(3) >> > (
			d_N, d_l, M1, M2
			);

		cufftHandle plan_cols = fft_plans.getPlan(N1, stream_pool.getFixedStream(4));

		cufftExecC2C(plan_cols, d_N, d_N, CUFFT_FORWARD);


		dim3 grid2d_trans((N2 + BLOCK_2D - 1) / BLOCK_2D, (N1 + BLOCK_2D - 1) / BLOCK_2D);

		transposeMatrix << <grid2d_trans, block2d, 0, stream_pool.getFixedStream(5) >> > (
			d_temp, d_N, N1, N2
			);


		reshape2DVector << <grid1d, block1d, 0, stream_pool.getFixedStream(6) >> > (
			d_out, d_temp, N1, N2, N2
			);

		cudaEventRecord(ev_end, stream_pool.getFixedStream(0));
		cudaEventSynchronize(ev_end);

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, ev_start, ev_end);
		totalTime += milliseconds;
		framePers++;

		return true;
	}

	float getAverageTime() const {
		return framePers > 0 ? totalTime / framePers : 0;
	}

	void resetStats() {
		totalTime = 0;
		framePers = 0;
	}

};

template<typename T>
class CudaDeviceMemory {
	T* ptr = nullptr;
	size_t m_size = 0;

public:
	CudaDeviceMemory() = default;

	explicit CudaDeviceMemory(size_t elements) : m_size(elements) {
		cudaError_t error = cudaMalloc(&ptr, elements * sizeof(T));
		if (error != cudaSuccess) {
			std::cerr << "CUDA malloc failed: " << cudaGetErrorString(error) << std::endl;
			ptr = nullptr;
			m_size = 0;
		}
	}

	~CudaDeviceMemory() {
		if (ptr) cudaFree(ptr);
	}

	CudaDeviceMemory(const CudaDeviceMemory&) = delete;
	CudaDeviceMemory& operator=(const CudaDeviceMemory&) = delete;

	CudaDeviceMemory(CudaDeviceMemory&& other) noexcept
		: ptr(other.ptr), m_size(other.m_size) {
		other.ptr = nullptr;
		other.m_size = 0;
	}

	CudaDeviceMemory& operator=(CudaDeviceMemory&& other) noexcept {
		if (this != &other) {
			if (ptr) cudaFree(ptr);
			ptr = other.ptr;
			m_size = other.m_size;
			other.ptr = nullptr;
			other.m_size = 0;
		}
		return *this;
	}

	T* data() const { return ptr; }
	size_t size() const { return m_size; }
	bool isValid() const { return ptr != nullptr; }

};




__host__ int main(void) {

	FFTProcessor fft(700000);

	int N = 700000;
	CudaDeviceMemory<complex> d_input(N);
	CudaDeviceMemory<complex> d_output(N);

	cuPMemory<complex> h_input(N);
	cuPMemory<complex> h_output(N);

	fft.FFTExecute(d_input.data(), d_output.data());
	fft.resetStats(); 

	for (int i = 0; i < N; ++i) {
		h_input.data()[i] = make_cuComplex(1.0f, 0.0f);
	}

	cudaMemcpy(d_input.data(), h_input.data(), N * sizeof(complex), cudaMemcpyHostToDevice);

	fft.FFTExecute(d_input.data(), d_output.data());

	cudaMemcpy(h_output.data(), d_output.data(), N * sizeof(complex), cudaMemcpyDeviceToHost);

	std::cout << "Average FFT time: " << fft.getAverageTime() << " ms" << std::endl;

	return 0;

}

