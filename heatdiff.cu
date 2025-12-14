#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define SIZE 512
#define STEPS 100
#define ALPHA 0.25f

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

__global__ void diffusion_kernel(float* curr, float* next, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || y <= 0 || x >= size - 1 || y >= size - 1) return;

    int idx = y * size + x;

    float c = curr[idx];
    float lap =
        curr[idx - 1] +
        curr[idx + 1] +
        curr[idx - size] +
        curr[idx + size] -
        4.0f * c;

    next[idx] = c + ALPHA * lap * 1.2f;
}

void diffusion_cpu(float* curr, float* next, int size) {
    for (int y = 1; y < size - 1; y++) {
        for (int x = 1; x < size - 1; x++) {
            int idx = y * size + x;

            float lap =
                curr[idx - 1] +
                curr[idx + 1] +
                curr[idx - size] +
                curr[idx + size] -
                4.0f * curr[idx];

            next[idx] = curr[idx] + ALPHA * lap;
        }
    }
}

void init_grid(float* grid, int size) {
    for (int i = 0; i < size * size; i++)
        grid[i] = 0.0f;

    int c = size / 2;
    int r = size / 16;

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int dx = x - c;
            int dy = y - c;
            if (dx * dx + dy * dy <= r * r)
                grid[y * size + x] = 1.0f;
        }
    }
}

double run_gpu(float* h_grid, int size) {
    float *d_a, *d_b;
    size_t bytes = size * size * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMemcpy(d_a, h_grid, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((size + 15) / 16, (size + 15) / 16);

    double start = get_time();

    for (int i = 0; i < STEPS; i++) {
        diffusion_kernel<<<grid, block>>>(d_a, d_b, size);
    }

    cudaDeviceSynchronize();
    double end = get_time();

    cudaMemcpy(h_grid, d_a, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);

    return end - start;
}

double run_cpu(float* a, float* b, int size) {
    double start = get_time();

    for (int i = 0; i < STEPS; i++) {
        diffusion_cpu(a, b, size);
        float* tmp = a;
        a = b;
        b = tmp;
    }

    double end = get_time();
    return end - start;
}

int main() {
    float* grid = (float*)malloc(SIZE * SIZE * sizeof(float));
    float* tmp  = (float*)malloc(SIZE * SIZE * sizeof(float));

    init_grid(grid, SIZE);
    double gpu_time = run_gpu(grid, SIZE);

    init_grid(grid, SIZE);
    double cpu_time = run_cpu(grid, tmp, SIZE);

    printf("GPU Time: %.4f sec\n", gpu_time);
    printf("CPU Time: %.4f sec\n", cpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);

    free(grid);
    free(tmp);
    return 0;
}

