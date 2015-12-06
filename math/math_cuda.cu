#include "math_cuda.h"

#include <stdio.h>
#include <cublas_v2.h>

//#include "math_cudnn.h"  // default implementation

using std::string;
using std::vector;
using std::shared_ptr;

static shared_ptr<MathCudnn> math_cudnn(new MathCudnn(0));

static const float kZero = 0.0f;
static const float kOne = 1.0f;

static cublasHandle_t cublas_handle;

static inline void CheckCuda(cudaError_t status)
{
  if (status != cudaSuccess)
  {
    printf("cuda error: %s\n", cudaGetErrorString(status));
  }
}

static inline void CheckCublas(int status)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    printf("cublas error: %u\n", status);
  }
}

inline void MathCuda::MemoryAlloc(shared_ptr<Mat> &mat)
{
  if (mat->data_device_ == nullptr)
  {
    CheckCuda(cudaMalloc((void **)&mat->data_device_,
                         mat->data_.size() * sizeof(float)));
    MemoryClear(mat);
  }
}

inline void MathCuda::MemoryFree(float *ptr)
{
  cudaFree(ptr);
}

inline void MathCuda::MemoryClear(std::shared_ptr<Mat> &mat)
{
  CheckCuda(
      cudaMemset(mat->data_device_, 0, mat->data_.size() * sizeof(float)));
}

inline void MathCuda::CopyToDevice(shared_ptr<Mat> &mat)
{
  MemoryAlloc(mat);

  CheckCublas(cublasSetVector(mat->data_.size(), sizeof(float), &mat->data_[0],
                              1, mat->data_device_, 1));
}

inline void MathCuda::CopyToHost(shared_ptr<Mat> &mat)
{
  CheckCublas(cublasGetVector(mat->data_.size(), sizeof(float),
                              mat->data_device_, 1, &mat->data_[0], 1));
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
  CheckCublas(cublasCreate(&cublas_handle));

  cudaSetDevice(gpu_id_);

  math_cudnn->Init();
}

void MathCuda::Deinit()
{
  CheckCublas(cublasDestroy(cublas_handle));
}

void MathCuda::Add(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                   shared_ptr<Mat> &out)
{
  int m = mat1->size_[0];
  int k = mat2->size_[0];
  int n = mat2->size_[1];

  // Process small matrices on cpu.
  if (m == 1 || n == 1 || k == 1)
  {
    math_cudnn->Add(mat1, mat2, out);
  }
  else
  {
    CheckCublas(cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n,
                            &kOne, mat1->data_device_, mat1->size_[1], &kOne,
                            mat2->data_device_, mat2->size_[1],
                            out->data_device_, out->size_[1]));
  }
}

void MathCuda::ElmtMul(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                       shared_ptr<Mat> &out)
{
  int len = mat1->data_.size();

  CheckCublas(cublasSgbmv(cublas_handle, CUBLAS_OP_N, len, len, 0, 0, &kOne,
                          mat1->data_device_, 1, mat2->data_device_, 1, &kZero,
                          out->data_device_, 1));
}

void MathCuda::Mul(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
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
    return;
  }

  // Process small matrices on cpu.
  // if (m == 1 || n == 1 || k == 1)
  //{
  //  math_cudnn->Mul(mat1, mat2, out);
  //}
  // else
  {
    CheckCublas(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                            &kOne, mat2->data_device_, mat2->size_[1],
                            mat1->data_device_, mat1->size_[1], &kZero,
                            out->data_device_, out->size_[1]));
  }
}

void MathCuda::AddDeriv(shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                        shared_ptr<Mat> &out)
{
  math_cudnn->AddDeriv(mat1d, mat2d, out);
}

void MathCuda::ElmtMulDeriv(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                            shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                            shared_ptr<Mat> &out)
{
  math_cudnn->ElmtMulDeriv(mat1, mat2, mat1d, mat2d, out);
}

void MathCuda::MulDeriv(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                        shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                        shared_ptr<Mat> &out)
{
  math_cudnn->MulDeriv(mat1, mat2, mat1d, mat2d, out);
}

void MathCuda::Relu(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                    Params &params)
{
  kRelu << <NUM_BLOCKS, NUM_THREADS>>>
      (in_w->data_device_, out_w->data_device_, in_w->data_.size());
}

void MathCuda::Sigm(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                    Params &params)
{
  kSigm << <NUM_BLOCKS, NUM_THREADS>>>
      (in_w->data_device_, out_w->data_device_, in_w->data_.size());
}

void MathCuda::Tanh(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                    Params &params)
{
  kTanh << <NUM_BLOCKS, NUM_THREADS>>>
      (in_w->data_device_, out_w->data_device_, in_w->data_.size());
}

void MathCuda::ReluDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                         Params &params)
{
  shared_ptr<Mat> &in_dw = in_w->dw_;
  shared_ptr<Mat> &out_dw = out_w->dw_;
  kReluDeriv << <NUM_BLOCKS, NUM_THREADS>>>
      (out_dw->data_device_, out_w->data_device_, in_dw->data_device_,
       out_dw->data_.size());
}

void MathCuda::SigmDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                         Params &params)
{
  shared_ptr<Mat> &in_dw = in_w->dw_;
  shared_ptr<Mat> &out_dw = out_w->dw_;
  kSigmDeriv << <NUM_BLOCKS, NUM_THREADS>>>
      (out_dw->data_device_, out_w->data_device_, in_dw->data_device_,
       out_dw->data_.size());
}

void MathCuda::TanhDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                         Params &params)
{
  shared_ptr<Mat> &in_dw = in_w->dw_;
  shared_ptr<Mat> &out_dw = out_w->dw_;
  kTanhDeriv << <NUM_BLOCKS, NUM_THREADS>>>
      (out_dw->data_device_, out_w->data_device_, in_dw->data_device_,
       out_dw->data_.size());
}

float MathCuda::Softmax(shared_ptr<Mat> &mat, shared_ptr<Mat> &out,
                        shared_ptr<Mat> &labels)
{
  return math_cudnn->Softmax(mat, out, labels);
}

void MathCuda::Fc(shared_ptr<Mat> &in, shared_ptr<Mat> &filters,
                  shared_ptr<Mat> &biases, shared_ptr<Mat> &out)
{
  math_cudnn->Fc(in, filters, biases, out);
}

void MathCuda::FcDeriv(shared_ptr<Mat> &in, shared_ptr<Mat> &filters,
                       shared_ptr<Mat> &biases, shared_ptr<Mat> &out)
{
  math_cudnn->FcDeriv(in, filters, biases, out);
}

void MathCuda::Conv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &filters_w,
                    shared_ptr<Mat> &biases_w, shared_ptr<Mat> &out_w,
                    Params &params)
{
  math_cudnn->Conv(in_w, filters_w, biases_w, out_w, params);
}

void MathCuda::ConvDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &filters_w,
                         shared_ptr<Mat> &biases_w, shared_ptr<Mat> &out_w,
                         Params &params)
{
  math_cudnn->ConvDeriv(in_w, filters_w, biases_w, out_w, params);
}

void MathCuda::MaxPool(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                       Params &params)
{
  math_cudnn->MaxPool(in_w, out_w, params);
}

void MathCuda::MaxPoolDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                            Params &params)
{
  math_cudnn->MaxPoolDeriv(in_w, out_w, params);
}

void MathCuda::AvePool(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                       Params &params)
{
  math_cudnn->AvePool(in_w, out_w, params);
}

void MathCuda::AvePoolDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                            Params &params)
{
  math_cudnn->AvePoolDeriv(in_w, out_w, params);
}

// learning

__global__ void kSGD(float *mat, float *mat_dw, unsigned int len,
                     float learning_rate, int batch_size)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += num_threads)
  {
    if (mat[i] != 0)
    {
      mat[i] += -learning_rate * (mat_dw[i] / batch_size);
    }
  }
}

__global__ void kRMSProp(float *mat, float *mat_dw, float *mat_prev,
                         unsigned int len, float learning_rate, int batch_size)
{
  float decay_rate = 0.999;
  float smooth_eps = 1e-8;
  float regc = 0.000001;  // L2 regularization strength
  float clipval = 5.0;    // clip gradients at this value

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += num_threads)
  {
    // Rmsprop adaptive learning rate.
    float mdwi = mat_dw[i] / batch_size;
    mat_prev[i] = decay_rate * mat_prev[i] + (1.0 - decay_rate) * mdwi * mdwi;

    // Gradient clip.
    if (mdwi > clipval)
    {
      mdwi = clipval;
    }
    if (mdwi < -clipval)
    {
      mdwi = -clipval;
    }

    // Update (and regularize).
    mat[i] +=
        -learning_rate * mdwi / sqrt(mat_prev[i] + smooth_eps) - regc * mat[i];

    mat_dw[i] = 0;
  }
}

void MathCuda::SGD(shared_ptr<Mat> &mat, float learning_rate, int batch_size)
{
  kSGD << <NUM_BLOCKS, NUM_THREADS>>>
      (mat->data_device_, mat->dw_->data_device_, mat->data_.size(),
       learning_rate, batch_size);
}

void MathCuda::Rmsprop(shared_ptr<Mat> &mat, shared_ptr<Mat> &mat_prev,
                       float learning_rate, int batch_size)
{
  kRMSProp << <NUM_BLOCKS, NUM_THREADS>>>
      (mat->data_device_, mat->dw_->data_device_, mat_prev->data_device_,
       mat->data_.size(), learning_rate, batch_size);
}
