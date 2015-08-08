#include "cuda.h"

#include <stdio.h>
#include <cublas_v2.h>

#include "cpu.h"  // SgemmCpu

cublasHandle_t handle;

int CopyToDevice(std::shared_ptr<Mat>& mat)
{
  size_t len = mat->size_[0] * mat->size_[1];

  cudaError_t error =
      cudaMalloc((void**)&mat->data_device_, len * sizeof(float));
  if (error != cudaSuccess)
  {
    return -1;
  }

  cublasStatus_t status = cublasSetVector(len, sizeof(float), &mat->data_[0], 1,
                                          mat->data_device_, 1);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    return -1;
  }

  return 0;
}

int CopyToHost(std::shared_ptr<Mat>& mat)
{
  size_t len = mat->size_[0] * mat->size_[1];

  cublasStatus_t status = cublasGetVector(len, sizeof(float), mat->data_device_,
                                          1, &mat->data_[0], 1);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    return -1;
  }

  return 0;
}

__device__ inline float Relu(float x)
{
  return ((x > 0) ? x : 0);
}

__device__ inline float Sigm(float x)
{
  return 1.0f / (1.0f + __expf(-x));
}

__device__ inline float Tanh(float x)
{
  // return (1.0f - __expf(-x)) / (1.0f + __expf(-x));
  return 1.0f - 2.0f / (__expf(2.0f * x) + 1.0f);
}

__device__ inline float ReluDeriv(float y)
{
  return ((y > 0) ? 1 : 0);
}

__device__ inline float SigmDeriv(float y)
{
  return y * (1 - y);
}

__device__ inline float TanhDeriv(float y)
{
  return 1 - y * y;
}

__global__ void kRelu(float* mat, float* out, unsigned int len)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += num_threads)
  {
    out[i] = Relu(mat[i]);
  }
}

__global__ void kSigm(float* mat, float* out, unsigned int len)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += num_threads)
  {
    out[i] = Sigm(mat[i]);
  }
}

__global__ void kTanh(float* mat, float* out, unsigned int len)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += num_threads)
  {
    out[i] = Tanh(mat[i]);
  }
}

__global__ void kReluDeriv(float* mat1, float* mat2, float* out,
                           unsigned int num_elems)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < num_elems; i += num_threads)
  {
    out[i] = mat1[i] * ReluDeriv(mat2[i]);
  }
}

__global__ void kSigmDeriv(float* mat1, float* mat2, float* out,
                           unsigned int num_elems)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < num_elems; i += num_threads)
  {
    out[i] = mat1[i] * SigmDeriv(mat2[i]);
  }
}

__global__ void kTanhDeriv(float* mat1, float* mat2, float* out,
                           unsigned int num_elems)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < num_elems; i += num_threads)
  {
    out[i] = mat1[i] * TanhDeriv(mat2[i]);
  }
}

#define NUM_BLOCKS 4096
#define NUM_THREADS 512

void MathCuda::Init()
{
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    return;
  }

  cudaSetDevice(gpu_id_);
}

void MathCuda::Deinit()
{
  cublasDestroy(handle);
  cudaThreadExit();
}

int MathCuda::Mul(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                  std::shared_ptr<Mat>& out)
{
  int m = mat1->size_[0];
  int k2 = mat1->size_[1];
  int k = mat2->size_[0];
  int n = mat2->size_[1];
  int m2 = out->size_[0];
  int n2 = out->size_[1];
  if (m != m2 || n != n2 || k != k2)
  {
    printf("%d %d %d %d %d %d\n", m, k2, k, n, m2, n2);
    return -1;
  }

  float alpha = 1.0f;
  float beta = 0.0f;

  // Process small matrices on cpu.
  if (m == 1 || n == 1 || k == 1)
  {
    SgemmCpu(true, false, false, m, n, k, alpha, &mat1->data_[0],
             mat1->size_[1], &mat2->data_[0], mat2->size_[1], beta,
             &out->data_[0], out->size_[1]);
  }
  else
  {
    CopyToDevice(mat1);
    CopyToDevice(mat2);
    CopyToDevice(out);

    cublasStatus_t status =
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                    mat2->data_device_, mat2->size_[1], mat1->data_device_,
                    mat1->size_[1], &beta, out->data_device_, out->size_[1]);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
      return -1;
    }

    CopyToHost(out);

    cudaFree(out->data_device_);
    cudaFree(mat2->data_device_);
    cudaFree(mat1->data_device_);
  }

  return 0;
}

int MathCuda::Add(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                  std::shared_ptr<Mat>& out)
{
  int m = mat1->size_[0];
  int k = mat2->size_[0];
  int n = mat2->size_[1];

  float alpha = 1.0f;
  float beta = 0.0f;

  // Process small matrices on cpu.
  if (m == 1 || n == 1 || k == 1)
  {
    for (int i = 0; i < mat1->data_.size(); i++)
    {
      out->data_[i] = mat1->data_[i] + mat2->data_[i];
    }
  }
  else
  {
    CopyToDevice(mat1);
    CopyToDevice(mat2);
    CopyToDevice(out);

    cublasStatus_t status = cublasSgeam(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alpha, mat1->data_device_,
        mat1->size_[1], &beta, mat2->data_device_, mat2->size_[1],
        out->data_device_, out->size_[1]);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
      return -1;
    }

    CopyToHost(out);

    cudaFree(out->data_device_);
    cudaFree(mat2->data_device_);
    cudaFree(mat1->data_device_);
  }

  return 0;
}

int MathCuda::ElmtMul(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                      std::shared_ptr<Mat>& out)
{
  CopyToDevice(mat1);
  CopyToDevice(mat2);
  CopyToDevice(out);

  int len = mat1->size_[0] * mat1->size_[1];

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasStatus_t status = cublasSgbmv(
      handle, CUBLAS_OP_N, len, len, 0, 0, &alpha, mat1->data_device_, 1,
      mat2->data_device_, 1, &beta, out->data_device_, 1);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    return -1;
  }

  CopyToHost(out);

  cudaFree(out->data_device_);
  cudaFree(mat2->data_device_);
  cudaFree(mat1->data_device_);

  return 0;
}

int MathCuda::AddDeriv(std::shared_ptr<Mat>& mat1d, std::shared_ptr<Mat>& mat2d,
                       std::shared_ptr<Mat>& out)
{
  for (int i = 0; i < mat1d->data_.size(); i++)
  {
    float curr = out->data_[i];
    mat1d->data_[i] += curr;
    mat2d->data_[i] += curr;
  }
  return 0;
}

int MathCuda::ElmtMulDeriv(std::shared_ptr<Mat>& mat1,
                           std::shared_ptr<Mat>& mat2,
                           std::shared_ptr<Mat>& mat1d,
                           std::shared_ptr<Mat>& mat2d,
                           std::shared_ptr<Mat>& out)
{
  for (int i = 0; i < mat1->data_.size(); i++)
  {
    float curr = out->data_[i];
    mat1d->data_[i] += mat2->data_[i] * curr;
    mat2d->data_[i] += mat1->data_[i] * curr;
  }
  return 0;
}

int MathCuda::MulDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                       std::shared_ptr<Mat>& mat1d, std::shared_ptr<Mat>& mat2d,
                       std::shared_ptr<Mat>& out)
{
  int mat1_size1 = mat1->size_[1];
  int mat2_size1 = mat2->size_[1];
  for (int i = 0; i < mat1->size_[0]; i++)
  {  // loop over rows of m1
    for (int j = 0; j < mat2_size1; j++)
    {  // loop over cols of m2
      for (int k = 0; k < mat1_size1; k++)
      {  // dot product loop
        float b = out->data_[mat2_size1 * i + j];
        mat1d->data_[mat1_size1 * i + k] += mat2->data_[mat2_size1 * k + j] * b;
        mat2d->data_[mat2_size1 * k + j] += mat1->data_[mat1_size1 * i + k] * b;
      }
    }
  }
  return 0;
}

int MathCuda::Relu(std::shared_ptr<Mat>& mat, std::shared_ptr<Mat>& out)
{
  unsigned int len = mat->size_[0] * mat->size_[1];

  CopyToDevice(mat);
  CopyToDevice(out);

  if (mat->size_[0] != out->size_[0] || mat->size_[1] != out->size_[1])
  {
    return -1;
  }

  kRelu << <NUM_BLOCKS, NUM_THREADS>>>
      (mat->data_device_, out->data_device_, len);

  CopyToHost(out);

  cudaFree(out->data_device_);
  cudaFree(mat->data_device_);

  return 0;
}

int MathCuda::Sigm(std::shared_ptr<Mat>& mat, std::shared_ptr<Mat>& out)
{
  unsigned int len = mat->size_[0] * mat->size_[1];

  CopyToDevice(mat);
  CopyToDevice(out);

  if (mat->size_[0] != out->size_[0] || mat->size_[1] != out->size_[1])
  {
    return -1;
  }

  kSigm << <NUM_BLOCKS, NUM_THREADS>>>
      (mat->data_device_, out->data_device_, len);

  CopyToHost(out);

  cudaFree(out->data_device_);
  cudaFree(mat->data_device_);

  return 0;
}

int MathCuda::Tanh(std::shared_ptr<Mat>& mat, std::shared_ptr<Mat>& out)
{
  unsigned int len = mat->size_[0] * mat->size_[1];

  CopyToDevice(mat);
  CopyToDevice(out);

  if (mat->size_[0] != out->size_[0] || mat->size_[1] != out->size_[1])
  {
    return -1;
  }

  kTanh << <NUM_BLOCKS, NUM_THREADS>>>
      (mat->data_device_, out->data_device_, len);

  CopyToHost(out);

  cudaFree(out->data_device_);
  cudaFree(mat->data_device_);

  return 0;
}

int MathCuda::ReluDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                        std::shared_ptr<Mat>& out)
{
  int len = mat1->size_[0] * mat1->size_[1];

  CopyToDevice(mat1);
  CopyToDevice(mat2);
  CopyToDevice(out);

  if (mat1->size_[0] != mat2->size_[0] || mat1->size_[1] != mat2->size_[1] ||
      mat1->size_[0] != out->size_[0] || mat1->size_[1] != out->size_[1])
  {
    return -1;
  }

  kReluDeriv << <NUM_BLOCKS, NUM_THREADS>>>
      (mat1->data_device_, mat2->data_device_, out->data_device_, len);

  CopyToHost(out);

  cudaFree(out->data_device_);
  cudaFree(mat2->data_device_);
  cudaFree(mat1->data_device_);

  return 0;
}

int MathCuda::SigmDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                        std::shared_ptr<Mat>& out)
{
  int len = mat1->size_[0] * mat1->size_[1];

  CopyToDevice(mat1);
  CopyToDevice(mat2);
  CopyToDevice(out);

  if (mat1->size_[0] != mat2->size_[0] || mat1->size_[1] != mat2->size_[1] ||
      mat1->size_[0] != out->size_[0] || mat1->size_[1] != out->size_[1])
  {
    return -1;
  }

  kSigmDeriv << <NUM_BLOCKS, NUM_THREADS>>>
      (mat1->data_device_, mat2->data_device_, out->data_device_, len);

  CopyToHost(out);

  cudaFree(out->data_device_);
  cudaFree(mat2->data_device_);
  cudaFree(mat1->data_device_);

  return 0;
}

int MathCuda::TanhDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                        std::shared_ptr<Mat>& out)
{
  int len = mat1->size_[0] * mat1->size_[1];

  CopyToDevice(mat1);
  CopyToDevice(mat2);
  CopyToDevice(out);

  if (mat1->size_[0] != mat2->size_[0] || mat1->size_[1] != mat2->size_[1] ||
      mat1->size_[0] != out->size_[0] || mat1->size_[1] != out->size_[1])
  {
    return -1;
  }

  kTanhDeriv << <NUM_BLOCKS, NUM_THREADS>>>
      (mat1->data_device_, mat2->data_device_, out->data_device_, len);

  CopyToHost(out);

  cudaFree(out->data_device_);
  cudaFree(mat2->data_device_);
  cudaFree(mat1->data_device_);

  return 0;
}
