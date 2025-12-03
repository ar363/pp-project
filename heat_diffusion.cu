#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__
void heat_step(const float* curr, float* next,
               int nx, int ny,
               float alpha, float dx, float dy, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y index

    if (i >= nx || j >= ny) return;

    int idx = j * nx + i;

    // Keep boundary fixed (Dirichlet)
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        next[idx] = curr[idx];
        return;
    }

    float uC = curr[idx];
    float uL = curr[j * nx + (i - 1)];
    float uR = curr[j * nx + (i + 1)];
    float uD = curr[(j - 1) * nx + i];
    float uU = curr[(j + 1) * nx + i];

    float dx2 = dx * dx;
    float dy2 = dy * dy;

    float laplacian = (uL - 2.0f * uC + uR) / dx2
                    + (uD - 2.0f * uC + uU) / dy2;

    next[idx] = uC + alpha * dt * laplacian;
}

void save_to_ppm(const char* filename, const float* h_u,
                 int nx, int ny, float minT, float maxT)
{
    FILE* f = fopen(filename, "wb");
    if (!f) {
        perror("fopen");
        return;
    }

    fprintf(f, "P6\n%d %d\n255\n", nx, ny);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            float val = h_u[j * nx + i];
            float t = (val - minT) / (maxT - minT + 1e-6f);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;

            unsigned char r = (unsigned char)(255.0f * t);
            unsigned char g = 0;
            unsigned char b = (unsigned char)(255.0f * (1.0f - t));

            unsigned char pixel[3] = { r, g, b };
            fwrite(pixel, 1, 3, f);
        }
    }

    fclose(f);
}

int main()
{
    // Grid parameters
    const int nx = 256;
    const int ny = 256;
    const int N = nx * ny;

    const float Lx = 1.0f;
    const float Ly = 1.0f;
    const float dx = Lx / (nx - 1);
    const float dy = Ly / (ny - 1);

    const float alpha = 0.01f; // thermal diffusivity

    // Stable dt (for dx=dy)
    float dt = 0.25f * dx * dx / alpha;

    const int steps = 3000;
    const int output_every = 500;

    // Host memory
    float* h_u = (float*)malloc(N * sizeof(float));

    // Initial condition: cold plate, hot square in center
    float T_cold = 0.0f;
    float T_hot = 1.0f;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = j * nx + i;
            h_u[idx] = T_cold;
        }
    }

    // Hot region
    for (int j = ny / 4; j < 3 * ny / 4; ++j) {
        for (int i = nx / 4; i < 3 * nx / 4; ++i) {
            h_u[j * nx + i] = T_hot;
        }
    }

    // Device memory
    float *d_curr, *d_next;
    cudaMalloc(&d_curr, N * sizeof(float));
    cudaMalloc(&d_next, N * sizeof(float));

    cudaMemcpy(d_curr, h_u, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    for (int step = 0; step < steps; ++step) {
        heat_step<<<grid, block>>>(d_curr, d_next, nx, ny, alpha, dx, dy, dt);
        cudaDeviceSynchronize();

        // Swap buffers
        float* tmp = d_curr;
        d_curr = d_next;
        d_next = tmp;

        if ((step % output_every) == 0) {
            cudaMemcpy(h_u, d_curr, N * sizeof(float), cudaMemcpyDeviceToHost);

            char filename[256];
            std::snprintf(filename, sizeof(filename),
                          "heat_%04d.ppm", step);

            save_to_ppm(filename, h_u, nx, ny, T_cold, T_hot);
            printf("Saved %s\n", filename);
        }
    }

    cudaFree(d_curr);
    cudaFree(d_next);
    free(h_u);

    return 0;
}
