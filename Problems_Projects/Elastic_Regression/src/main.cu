/********************************************************************
 Elastic Net Regression (PGD Solver)

 Objective:
    min_beta (1/2n)||y - Xβ||²₂ + λ1||β||₁ + λ2||β||²₂

 Symbols:
    X  : Feature matrix (n x d)
    y  : Target vector (n)
    β  : Model coefficients (d)
    z  : Prediction vector = Xβ
    r  : Residual vector = z - y
    g  : Gradient
    η  : Step size

 Kernel Decomposition:
    1. Xβ
    2. residual
    3. Xᵀr
    4. gradient + proximal update
********************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#define BLOCK_SIZE 256

// ============================================================
// Utility Macro
// ============================================================

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << std::endl; \
        exit(EXIT_FAILURE); \
    }

// ============================================================
// Soft Threshold Operator
// ============================================================

__device__
float soft_threshold(float x, float kappa)
{
    if (x > kappa) return x - kappa;
    if (x < -kappa) return x + kappa;
    return 0.0f;
}

// ============================================================
// Kernel 1: z = Xβ
// ============================================================

__global__
void matvec_Xbeta(
    const float* X,
    const float* beta,
    float* z,
    int n,
    int d)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    float sum = 0.0f;

    for (int j = 0; j < d; j++)
        sum += X[row * d + j] * beta[j];

    z[row] = sum;
}

// ============================================================
// Kernel 2: r = z − y
// ============================================================

__global__
void compute_residual(
    const float* z,
    const float* y,
    float* r,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        r[i] = z[i] - y[i];
}

// ============================================================
// Kernel 3: Xt_r = Xᵀ r
// ============================================================

__global__
void matvec_XT_r(
    const float* X,
    const float* r,
    float* Xt_r,
    int n,
    int d)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= d) return;

    float sum = 0.0f;

    for (int i = 0; i < n; i++)
        sum += X[i * d + col] * r[i];

    Xt_r[col] = sum;
}

// ============================================================
// Kernel 4: Gradient + Proximal Update
// ============================================================

__global__
void gradient_prox_update(
    float* beta,
    const float* Xt_r,
    float eta,
    float lambda1,
    float lambda2,
    int n,
    int d)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d) return;

    float g =
        (1.0f / n) * Xt_r[j]
        + 2.0f * lambda2 * beta[j];

    float temp = beta[j] - eta * g;

    beta[j] = soft_threshold(temp, lambda1 * eta);
}

// ============================================================
// Host Helper: Initialize Data
// ============================================================

void initialize_data(std::vector<float>& X,
                     std::vector<float>& y,
                     int n, int d)
{
    for (int i = 0; i < n * d; i++)
        X[i] = static_cast<float>(rand()) / RAND_MAX;

    for (int i = 0; i < n; i++)
        y[i] = static_cast<float>(rand()) / RAND_MAX;
}

// ============================================================
// Main Program
// ============================================================

int main()
{
    // Problem size
    int n = 1024;   // samples
    int d = 256;    // features

    int max_iter = 100;

    float eta = 0.001f;
    float lambda1 = 0.01f;
    float lambda2 = 0.01f;

    std::cout << "Elastic Net PGD CUDA Example\n";

    // Host memory
    std::vector<float> h_X(n * d);
    std::vector<float> h_y(n);
    std::vector<float> h_beta(d, 0.0f);

    initialize_data(h_X, h_y, n, d);

    // Device memory
    float *d_X, *d_y, *d_beta, *d_z, *d_r, *d_Xt_r;

    CUDA_CHECK(cudaMalloc(&d_X, n*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Xt_r, d*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(),
                          n*d*sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(),
                          n*sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(),
                          d*sizeof(float),
                          cudaMemcpyHostToDevice));

    // Launch configs
    dim3 block(BLOCK_SIZE);
    dim3 gridN((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 gridD((d + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // ========================================================
    // PGD Iterations
    // ========================================================

    for (int iter = 0; iter < max_iter; iter++)
    {
        matvec_Xbeta<<<gridN, block>>>(d_X, d_beta, d_z, n, d);

        compute_residual<<<gridN, block>>>(d_z, d_y, d_r, n);

        matvec_XT_r<<<gridD, block>>>(d_X, d_r, d_Xt_r, n, d);

        gradient_prox_update<<<gridD, block>>>(
            d_beta,
            d_Xt_r,
            eta,
            lambda1,
            lambda2,
            n,
            d);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_beta.data(),
                          d_beta,
                          d*sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::cout << "Training completed.\n";
    std::cout << "First 10 beta values:\n";

    for (int i = 0; i < 10; i++)
        std::cout << h_beta[i] << " ";

    std::cout << std::endl;

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_beta);
    cudaFree(d_z);
    cudaFree(d_r);
    cudaFree(d_Xt_r);

    return 0;
}