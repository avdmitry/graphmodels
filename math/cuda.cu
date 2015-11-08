#include "cuda.h"

#include <stdio.h>
#include <cublas_v2.h>

#include "cpu.h"  // SgemmCpu, default implementation

using std::string;
using std::vector;
using std::shared_ptr;

static shared_ptr<MathCpu> math_cpu(new MathCpu);

cublasHandle_t handle;

int CopyToDevice(shared_ptr<Mat> &mat)
{
  size_t len = mat->size_[0] * mat->size_[1] * mat->size_[2] * mat->size_[3];

  cudaError_t error =
      cudaMalloc((void **)&mat->data_device_, len * sizeof(float));
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

int CopyToHost(shared_ptr<Mat> &mat)
{
  size_t len = mat->size_[0] * mat->size_[1] * mat->size_[2] * mat->size_[3];

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

__global__ void kRelu(float *mat, float *out, unsigned int len)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += num_threads)
  {
    out[i] = Relu(mat[i]);
  }
}

__global__ void kSigm(float *mat, float *out, unsigned int len)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += num_threads)
  {
    out[i] = Sigm(mat[i]);
  }
}

__global__ void kTanh(float *mat, float *out, unsigned int len)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += num_threads)
  {
    out[i] = Tanh(mat[i]);
  }
}

__global__ void kReluDeriv(float *mat1, float *mat2, float *out,
                           unsigned int num_elems)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < num_elems; i += num_threads)
  {
    out[i] = mat1[i] * ReluDeriv(mat2[i]);
  }
}

__global__ void kSigmDeriv(float *mat1, float *mat2, float *out,
                           unsigned int num_elems)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < num_elems; i += num_threads)
  {
    out[i] = mat1[i] * SigmDeriv(mat2[i]);
  }
}

__global__ void kTanhDeriv(float *mat1, float *mat2, float *out,
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
    printf("cublas create failed\n");
    return;
  }

  cudaSetDevice(gpu_id_);
}

void MathCuda::Deinit()
{
  cublasDestroy(handle);
  cudaThreadExit();
}

int MathCuda::Mul(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                  shared_ptr<Mat> &out)
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
    math_cpu->Mul(mat1, mat2, out);
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

int MathCuda::Add(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                  shared_ptr<Mat> &out)
{
  int m = mat1->size_[0];
  int k = mat2->size_[0];
  int n = mat2->size_[1];

  float alpha = 1.0f;
  float beta = 1.0f;

  // Process small matrices on cpu.
  if (m == 1 || n == 1 || k == 1)
  {
    math_cpu->Add(mat1, mat2, out);
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

int MathCuda::ElmtMul(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                      shared_ptr<Mat> &out)
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

int MathCuda::AddDeriv(shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                       shared_ptr<Mat> &out)
{
  return math_cpu->AddDeriv(mat1d, mat2d, out);
}

int MathCuda::ElmtMulDeriv(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                           shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                           shared_ptr<Mat> &out)
{
  return math_cpu->ElmtMulDeriv(mat1, mat2, mat1d, mat2d, out);
}

int MathCuda::MulDeriv(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                       shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                       shared_ptr<Mat> &out)
{
  return math_cpu->MulDeriv(mat1, mat2, mat1d, mat2d, out);
}

int MathCuda::Relu(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w)
{
  unsigned int len = in_w->size_[0] * in_w->size_[1];

  CopyToDevice(in_w);
  CopyToDevice(out_w);

  if (in_w->size_[0] != out_w->size_[0] || in_w->size_[1] != out_w->size_[1])
  {
    return -1;
  }

  kRelu << <NUM_BLOCKS, NUM_THREADS>>>
      (in_w->data_device_, out_w->data_device_, len);

  CopyToHost(out_w);

  cudaFree(out_w->data_device_);
  cudaFree(in_w->data_device_);

  return 0;
}

int MathCuda::Sigm(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w)
{
  unsigned int len = in_w->size_[0] * in_w->size_[1];

  CopyToDevice(in_w);
  CopyToDevice(out_w);

  if (in_w->size_[0] != out_w->size_[0] || in_w->size_[1] != out_w->size_[1])
  {
    return -1;
  }

  kSigm << <NUM_BLOCKS, NUM_THREADS>>>
      (in_w->data_device_, out_w->data_device_, len);

  CopyToHost(out_w);

  cudaFree(out_w->data_device_);
  cudaFree(in_w->data_device_);

  return 0;
}

int MathCuda::Tanh(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w)
{
  unsigned int len = in_w->size_[0] * in_w->size_[1];

  CopyToDevice(in_w);
  CopyToDevice(out_w);

  if (in_w->size_[0] != out_w->size_[0] || in_w->size_[1] != out_w->size_[1])
  {
    return -1;
  }

  kTanh << <NUM_BLOCKS, NUM_THREADS>>>
      (in_w->data_device_, out_w->data_device_, len);

  CopyToHost(out_w);

  cudaFree(out_w->data_device_);
  cudaFree(in_w->data_device_);

  return 0;
}

int MathCuda::ReluDeriv(shared_ptr<Mat> &in_dw, shared_ptr<Mat> &out_w,
                        shared_ptr<Mat> &out_dw)
{
  int len = out_dw->size_[0] * out_dw->size_[1];

  CopyToDevice(out_dw);
  CopyToDevice(out_w);
  CopyToDevice(in_dw);

  if (out_dw->size_[0] != out_w->size_[0] ||
      out_dw->size_[1] != out_w->size_[1] ||
      out_dw->size_[0] != in_dw->size_[0] ||
      out_dw->size_[1] != in_dw->size_[1])
  {
    return -1;
  }

  kReluDeriv << <NUM_BLOCKS, NUM_THREADS>>>
      (out_dw->data_device_, out_w->data_device_, in_dw->data_device_, len);

  CopyToHost(in_dw);

  cudaFree(in_dw->data_device_);
  cudaFree(out_w->data_device_);
  cudaFree(out_dw->data_device_);

  return 0;
}

int MathCuda::SigmDeriv(shared_ptr<Mat> &in_dw, shared_ptr<Mat> &out_w,
                        shared_ptr<Mat> &out_dw)
{
  int len = out_dw->size_[0] * out_dw->size_[1];

  CopyToDevice(out_dw);
  CopyToDevice(out_w);
  CopyToDevice(in_dw);

  if (out_dw->size_[0] != out_w->size_[0] ||
      out_dw->size_[1] != out_w->size_[1] ||
      out_dw->size_[0] != in_dw->size_[0] ||
      out_dw->size_[1] != in_dw->size_[1])
  {
    return -1;
  }

  kSigmDeriv << <NUM_BLOCKS, NUM_THREADS>>>
      (out_dw->data_device_, out_w->data_device_, in_dw->data_device_, len);

  CopyToHost(in_dw);

  cudaFree(in_dw->data_device_);
  cudaFree(out_w->data_device_);
  cudaFree(out_dw->data_device_);

  return 0;
}

int MathCuda::TanhDeriv(shared_ptr<Mat> &in_dw, shared_ptr<Mat> &out_w,
                        shared_ptr<Mat> &out_dw)
{
  int len = out_dw->size_[0] * out_dw->size_[1];

  CopyToDevice(out_dw);
  CopyToDevice(out_w);
  CopyToDevice(in_dw);

  if (out_dw->size_[0] != out_w->size_[0] ||
      out_dw->size_[1] != out_w->size_[1] ||
      out_dw->size_[0] != in_dw->size_[0] ||
      out_dw->size_[1] != in_dw->size_[1])
  {
    return -1;
  }

  kTanhDeriv << <NUM_BLOCKS, NUM_THREADS>>>
      (out_dw->data_device_, out_w->data_device_, in_dw->data_device_, len);

  CopyToHost(in_dw);

  cudaFree(in_dw->data_device_);
  cudaFree(out_w->data_device_);
  cudaFree(out_dw->data_device_);

  return 0;
}

shared_ptr<Mat> MathCuda::Softmax(shared_ptr<Mat> &mat)
{
  return math_cpu->Softmax(mat);
}

int MathCuda::Conv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &filters_w,
                   shared_ptr<Mat> &biases_w, shared_ptr<Mat> &out_w,
                   ConvParams &conv_params)
{
  return math_cpu->Conv(in_w, filters_w, biases_w, out_w, conv_params);
}

int MathCuda::ConvDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &in_dw,
                        shared_ptr<Mat> &filters_w, shared_ptr<Mat> &filters_dw,
                        shared_ptr<Mat> &biases_dw, shared_ptr<Mat> &out_w,
                        shared_ptr<Mat> &out_dw, ConvParams &conv_params)
{
  return math_cpu->ConvDeriv(in_w, in_dw, filters_w, filters_dw, biases_dw,
                             out_w, out_dw, conv_params);
}

int MathCuda::MaxPool(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                      ConvParams &conv_params)
{
  return math_cpu->MaxPool(in_w, out_w, conv_params);
}

int MathCuda::MaxPoolDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &in_dw,
                           shared_ptr<Mat> &out_w, shared_ptr<Mat> &out_dw,
                           ConvParams &conv_params)
{
  return math_cpu->MaxPoolDeriv(in_w, in_dw, out_w, out_dw, conv_params);
}
