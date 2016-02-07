#include "math_cudnn.h"

#include <stdio.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <float.h>

#include "math_cpu.h"  // SgemmCpu, default implementation

using std::string;
using std::vector;
using std::shared_ptr;

static shared_ptr<MathCpu> math_cpu(new MathCpu);

const cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
static const float kZero = 0.0f;
static const float kOne = 1.0f;
static const int kMb = 1024 * 1024;
void *work_space;
size_t work_space_size = 10 * kMb;

cudnnHandle_t cudnn_handle;
cublasHandle_t cublas_handle;

inline void CheckCuda(cudaError_t status)
{
  if (status != cudaSuccess)
  {
    printf("cuda error: %s\n", cudaGetErrorString(status));
    exit(-1);
  }
}

inline void CheckCublas(int status)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    printf("cublas error: %u\n", status);
    exit(-1);
  }
}

inline void CheckCudnn(cudnnStatus_t status)
{
  if (status != CUDNN_STATUS_SUCCESS)
  {
    printf("cudnn error: %s\n", cudnnGetErrorString(status));
    exit(-1);
  }
}

inline void SetTensorDescriptor(cudnnTensorDescriptor_t descr_tensor, int n,
                                int c, int h, int w)
{
  int stride_w = 1;
  int stride_h = w * stride_w;
  int stride_c = h * stride_h;
  int stride_n = c * stride_c;
  CheckCudnn(cudnnSetTensor4dDescriptorEx(descr_tensor, data_type, n, c, h, w,
                                          stride_n, stride_c, stride_h,
                                          stride_w));
}

inline void MathCudnn::MemoryAlloc(shared_ptr<Mat> &mat)
{
  if (mat->data_device_ == nullptr)
  {
    CheckCuda(cudaMalloc((void **)&mat->data_device_,
                         mat->data_.size() * sizeof(float)));
    MemoryClear(mat);
  }
}

inline void MathCudnn::MemoryFree(float *ptr)
{
  cudaFree(ptr);
}

inline void MathCudnn::MemoryClear(std::shared_ptr<Mat> &mat)
{
  CheckCuda(
      cudaMemset(mat->data_device_, 0, mat->data_.size() * sizeof(float)));
}

inline void MathCudnn::CopyToDevice(shared_ptr<Mat> &mat)
{
  MemoryAlloc(mat);

  CheckCublas(cublasSetVector(mat->data_.size(), sizeof(float), &mat->data_[0],
                              1, mat->data_device_, 1));
}

inline void MathCudnn::CopyToHost(shared_ptr<Mat> &mat)
{
  CheckCublas(cublasGetVector(mat->data_.size(), sizeof(float),
                              mat->data_device_, 1, &mat->data_[0], 1));
}

void MathCudnn::Init()
{
  CheckCudnn(cudnnCreate(&cudnn_handle));
  CheckCublas(cublasCreate(&cublas_handle));

  cudaSetDevice(gpu_id_);

  size_t free_mem, total_mem;
  CheckCuda(cudaMemGetInfo(&free_mem, &total_mem));
  printf("GPU memory: total %.3f Mb, free %.3f Mb\n", 1.0 * free_mem / kMb,
         1.0 * total_mem / kMb);
  CheckCuda(cudaMalloc(&work_space, work_space_size));
  printf("cudnn workspace size: %.3f Mb\n", 1.0 * work_space_size / kMb);
}

void MathCudnn::Deinit()
{
  if (work_space_size != 0)
  {
    CheckCuda(cudaFree(work_space));
  }

  CheckCublas(cublasDestroy(cublas_handle));
  CheckCudnn(cudnnDestroy(cudnn_handle));
}

void MathCudnn::Sync()
{
  cudaDeviceSynchronize();
}

void MathCudnn::Add(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                    shared_ptr<Mat> &out)
{
  int m = mat1->size_[0];
  int k = mat2->size_[0];
  int n = mat2->size_[1];

  // Process small matrices on cpu.
  if (m == 1 || n == 1 || k == 1)
  {
    math_cpu->Add(mat1, mat2, out);
  }
  else
  {
    CheckCublas(cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n,
                            &kOne, mat1->data_device_, mat1->size_[1], &kOne,
                            mat2->data_device_, mat2->size_[1],
                            out->data_device_, out->size_[1]));
  }
}

void MathCudnn::ElmtMul(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                        shared_ptr<Mat> &out)
{
  int len = mat1->data_.size();
  CheckCublas(cublasSgbmv(cublas_handle, CUBLAS_OP_N, len, len, 0, 0, &kOne,
                          mat1->data_device_, 1, mat2->data_device_, 1, &kZero,
                          out->data_device_, 1));
}

void MathCudnn::Mul(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
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
  //  math_cpu->Mul(mat1, mat2, out);
  // }
  // else
  {
    CheckCublas(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                            &kOne, mat2->data_device_, mat2->size_[1],
                            mat1->data_device_, mat1->size_[1], &kZero,
                            out->data_device_, out->size_[1]));
  }
}

void MathCudnn::AddDeriv(shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                         shared_ptr<Mat> &out)
{
  math_cpu->AddDeriv(mat1d, mat2d, out);
}

void MathCudnn::ElmtMulDeriv(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                             shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                             shared_ptr<Mat> &out)
{
  math_cpu->ElmtMulDeriv(mat1, mat2, mat1d, mat2d, out);
}

void MathCudnn::MulDeriv(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                         shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                         shared_ptr<Mat> &out)
{
  int m = mat1d->size_[0];
  int n = mat1d->size_[1];
  int k = mat2->size_[1];
  // SgemmCpu(true, false, true, m, n, k, alpha, &out->data_[0], k,
  //         &mat2->data_[0], k, beta, &mat1d->data_[0], n);
  CheckCublas(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                          &kOne, mat2->data_device_, k, out->data_device_,
                          out->size_[1], &kZero, mat1d->data_device_, n));

  m = mat2d->size_[1];
  n = mat1->size_[1];
  k = mat1->size_[0];
  // SgemmCpu(false, false, true, m, n, k, alpha, &out->data_[0], m,
  //         &mat1->data_[0], n, beta, &mat2d->data_[0], m);
  CheckCublas(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                          &kOne, out->data_device_, m, mat1->data_device_, n,
                          &kZero, mat2d->data_device_, m));
}

class BatchNormContextCudnn : public Context
{
 public:
  BatchNormContextCudnn()
  {
    CheckCudnn(cudnnCreateTensorDescriptor(&descr_tensor_src));
    CheckCudnn(cudnnCreateTensorDescriptor(&descr_tensor_scale_bias_mean_var));
  }
  ~BatchNormContextCudnn()
  {
    CheckCuda(cudaFree(cashed_mean_));
    CheckCuda(cudaFree(cashed_variance_));

    CheckCudnn(cudnnDestroyTensorDescriptor(descr_tensor_src));
    CheckCudnn(cudnnDestroyTensorDescriptor(descr_tensor_scale_bias_mean_var));
  }

  cudnnTensorDescriptor_t descr_tensor_src, descr_tensor_scale_bias_mean_var;
  float *cashed_mean_, *cashed_variance_;
};

void MathCudnn::BatchNormSetUp(shared_ptr<Mat> &in_w, Params &params)
{
  shared_ptr<BatchNormContextCudnn> context(new BatchNormContextCudnn);
  params.context = shared_ptr<Context>(context);

  int n = in_w->size_[3];
  int c = in_w->size_[2];
  int h = in_w->size_[1];
  int w = in_w->size_[0];
  SetTensorDescriptor(context->descr_tensor_src, n, c, h, w);
  SetTensorDescriptor(context->descr_tensor_scale_bias_mean_var, 1, c, 1, 1);

  CheckCuda(cudaMalloc((void **)&context->cashed_mean_, c * sizeof(float)));
  CheckCuda(cudaMalloc((void **)&context->cashed_variance_, c * sizeof(float)));
}

void MathCudnn::BatchNorm(shared_ptr<Mat> &in_w, shared_ptr<Mat> &scale,
                          shared_ptr<Mat> &bias, shared_ptr<Mat> &mean,
                          shared_ptr<Mat> &variance, shared_ptr<Mat> &out_w,
                          Params &params, bool train)
{
  BatchNormContextCudnn *context =
      static_cast<BatchNormContextCudnn *>(params.context.get());

  if (train)
  {
    CheckCudnn(cudnnBatchNormalizationForwardTraining(
        cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &kOne, &kZero,
        context->descr_tensor_src, in_w->data_device_,
        context->descr_tensor_src, out_w->data_device_,
        context->descr_tensor_scale_bias_mean_var, scale->data_device_,
        bias->data_device_, 1, mean->data_device_, variance->data_device_,
        CUDNN_BN_MIN_EPSILON, context->cashed_mean_,
        context->cashed_variance_));
  }
  else
  {
    CheckCudnn(cudnnBatchNormalizationForwardInference(
        cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &kOne, &kZero,
        context->descr_tensor_src, in_w->data_device_,
        context->descr_tensor_src, out_w->data_device_,
        context->descr_tensor_scale_bias_mean_var, scale->data_device_,
        bias->data_device_, mean->data_device_, variance->data_device_,
        CUDNN_BN_MIN_EPSILON));
  }
}

void MathCudnn::BatchNormDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &scale,
                               shared_ptr<Mat> &bias, shared_ptr<Mat> &mean,
                               shared_ptr<Mat> &variance,
                               shared_ptr<Mat> &out_w, Params &params)
{
  BatchNormContextCudnn *context =
      static_cast<BatchNormContextCudnn *>(params.context.get());

  CheckCudnn(cudnnBatchNormalizationBackward(
      cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &kOne, &kZero,
      context->descr_tensor_src, in_w->data_device_, context->descr_tensor_src,
      out_w->dw_->data_device_, context->descr_tensor_src,
      in_w->dw_->data_device_, context->descr_tensor_scale_bias_mean_var,
      scale->data_device_, scale->dw_->data_device_, bias->dw_->data_device_,
      CUDNN_BN_MIN_EPSILON, context->cashed_mean_, context->cashed_variance_));
}

class ActivContextCudnn : public Context
{
 public:
  ActivContextCudnn()
  {
    CheckCudnn(cudnnCreateTensorDescriptor(&descr_tensor_src));
    CheckCudnn(cudnnCreateTensorDescriptor(&descr_tensor_dst));
  }
  ~ActivContextCudnn()
  {
    CheckCudnn(cudnnDestroyTensorDescriptor(descr_tensor_src));
    CheckCudnn(cudnnDestroyTensorDescriptor(descr_tensor_dst));
  }

  cudnnTensorDescriptor_t descr_tensor_src, descr_tensor_dst;
};

void MathCudnn::ActivSetUp(shared_ptr<Mat> &in_w, Params &params)
{
  shared_ptr<ActivContextCudnn> context(new ActivContextCudnn);
  params.context = shared_ptr<Context>(context);

  int n = in_w->size_[3];
  int c = in_w->size_[2];
  int h = in_w->size_[1];
  int w = in_w->size_[0];
  SetTensorDescriptor(context->descr_tensor_src, n, c, h, w);
  SetTensorDescriptor(context->descr_tensor_dst, n, c, h, w);
}

void MathCudnn::Relu(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                     Params &params)
{
  ActivContextCudnn *context =
      static_cast<ActivContextCudnn *>(params.context.get());

  CheckCudnn(cudnnActivationForward(
      cudnn_handle, CUDNN_ACTIVATION_RELU, &kOne, context->descr_tensor_src,
      in_w->data_device_, &kZero, context->descr_tensor_dst,
      out_w->data_device_));
}

void MathCudnn::Sigm(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                     Params &params)
{
  ActivContextCudnn *context =
      static_cast<ActivContextCudnn *>(params.context.get());

  CheckCudnn(cudnnActivationForward(
      cudnn_handle, CUDNN_ACTIVATION_SIGMOID, &kOne, context->descr_tensor_src,
      in_w->data_device_, &kZero, context->descr_tensor_dst,
      out_w->data_device_));
}

void MathCudnn::Tanh(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                     Params &params)
{
  ActivContextCudnn *context =
      static_cast<ActivContextCudnn *>(params.context.get());

  CheckCudnn(cudnnActivationForward(
      cudnn_handle, CUDNN_ACTIVATION_TANH, &kOne, context->descr_tensor_src,
      in_w->data_device_, &kZero, context->descr_tensor_dst,
      out_w->data_device_));
}

void MathCudnn::ReluDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                          Params &params)
{
  shared_ptr<Mat> &in_dw = in_w->dw_;
  shared_ptr<Mat> &out_dw = out_w->dw_;

  ActivContextCudnn *context =
      static_cast<ActivContextCudnn *>(params.context.get());

  CheckCudnn(cudnnActivationBackward(
      cudnn_handle, CUDNN_ACTIVATION_RELU, &kOne, context->descr_tensor_src,
      out_w->data_device_, context->descr_tensor_src, out_dw->data_device_,
      context->descr_tensor_dst, in_w->data_device_, &kZero,
      context->descr_tensor_dst, in_dw->data_device_));
}

void MathCudnn::SigmDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                          Params &params)
{
  shared_ptr<Mat> &in_dw = in_w->dw_;
  shared_ptr<Mat> &out_dw = out_w->dw_;

  ActivContextCudnn *context =
      static_cast<ActivContextCudnn *>(params.context.get());

  CheckCudnn(cudnnActivationBackward(
      cudnn_handle, CUDNN_ACTIVATION_SIGMOID, &kOne, context->descr_tensor_src,
      out_w->data_device_, context->descr_tensor_src, out_dw->data_device_,
      context->descr_tensor_dst, in_w->data_device_, &kZero,
      context->descr_tensor_dst, in_dw->data_device_));
}

void MathCudnn::TanhDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                          Params &params)
{
  shared_ptr<Mat> &in_dw = in_w->dw_;
  shared_ptr<Mat> &out_dw = out_w->dw_;

  ActivContextCudnn *context =
      static_cast<ActivContextCudnn *>(params.context.get());

  CheckCudnn(cudnnActivationBackward(
      cudnn_handle, CUDNN_ACTIVATION_TANH, &kOne, context->descr_tensor_src,
      out_w->data_device_, context->descr_tensor_src, out_dw->data_device_,
      context->descr_tensor_dst, in_w->data_device_, &kZero,
      context->descr_tensor_dst, in_dw->data_device_));
}

static __global__ void kSoftmax(float *mat, float *out, unsigned int len,
                                int num_elements)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int batch = idx; batch < len; batch += num_threads)
  {
    int offset = batch * num_elements;
    float maxval = mat[offset];
    for (int i = 0; i < num_elements; i++)
    {
      if (mat[offset + i] > maxval)
      {
        maxval = mat[offset + i];
      }
    }

    float sum = 0.0;
    for (int i = 0; i < num_elements; i++)
    {
      out[offset + i] = exp(mat[offset + i] - maxval);
      sum += out[offset + i];
    }
    for (int i = 0; i < num_elements; i++)
    {
      out[offset + i] /= sum;
    }
  }
}

static __global__ void kSoftmaxDeriv(float *mat_dw, float *labels, float *out,
                                     unsigned int len, int num_elements,
                                     float *loss)
{
  *loss = 0;
  for (int batch = 0; batch < len; ++batch)
  {
    int offset = batch * num_elements;
    for (int i = 0; i < num_elements; i++)
    {
      mat_dw[offset + i] = out[offset + i];
    }
    int idx = batch * num_elements + labels[batch];
    mat_dw[idx] -= 1;
    *loss -= log(out[idx]);
  }
}

#define NUM_BLOCKS 4096
#define NUM_THREADS 512

float MathCudnn::Softmax(shared_ptr<Mat> &mat, shared_ptr<Mat> &out,
                         shared_ptr<Mat> &labels)
{
  out = shared_ptr<Mat>(new Mat(mat->size_, false));
  math->CopyToDevice(out);
  int num_elements = mat->size_[0] * mat->size_[1] * mat->size_[2];
  kSoftmax << <256, 1>>>
      (mat->data_device_, out->data_device_, mat->size_[3], num_elements);

  float *loss_device, loss;
  CheckCuda(cudaMalloc(&loss_device, sizeof(float)));
  kSoftmaxDeriv << <1, 1>>> (mat->dw_->data_device_, labels->data_device_,
                             out->data_device_, mat->size_[3], num_elements,
                             loss_device);
  CheckCublas(cublasGetVector(1, sizeof(float), loss_device, 1, &loss, 1));

  return loss;
}

static __global__ void kAddBias(float *biases, float *out, unsigned int len,
                                int num_out)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int batch = idx; batch < len; batch += num_threads)
  {
    int out_offset = num_out * batch;
    for (int i = 0; i < num_out; ++i)
    {
      out[out_offset + i] += biases[i];
    }
  }
}

static __global__ void kAddBiasDeriv(float *biases_dw, float *out_dw,
                                     unsigned int len, int num_out)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int batch = idx; batch < len; batch += num_threads)
  {
    int out_offset = num_out * batch;
    for (int i = 0; i < num_out; ++i)
    {
      float dw = out_dw[out_offset + i];
      biases_dw[i] += dw;
    }
  }
}

void MathCudnn::Fc(shared_ptr<Mat> &in, shared_ptr<Mat> &filters,
                   shared_ptr<Mat> &biases, shared_ptr<Mat> &out)
{
  int num_out = out->size_[2];
  int num_in = filters->size_[0];
  int num_batch = in->size_[3];

  std::vector<int> in_size(in->size_);
  std::vector<int> out_size(out->size_);
  in->size_[0] = num_batch;
  in->size_[1] = num_in;
  in->size_[2] = 1;
  in->size_[3] = 1;
  out->size_[0] = num_batch;
  out->size_[1] = num_out;
  out->size_[2] = 1;
  out->size_[3] = 1;
  Mul(in, filters, out);
  in->size_ = in_size;
  out->size_ = out_size;

  kAddBias << <256, 1>>>
      (biases->data_device_, out->data_device_, num_batch, num_out);
}

void MathCudnn::FcDeriv(shared_ptr<Mat> &in, shared_ptr<Mat> &filters,
                        shared_ptr<Mat> &biases, shared_ptr<Mat> &out)
{
  int num_out = out->size_[2];
  int num_in = filters->size_[0];
  int num_batch = in->size_[3];

  std::vector<int> in_size(in->size_);
  std::vector<int> in_dw_size(in->dw_->size_);
  std::vector<int> out_dw_size(out->dw_->size_);
  in->size_[0] = num_batch;
  in->size_[1] = num_in;
  in->size_[2] = 1;
  in->size_[3] = 1;
  in->dw_->size_[0] = num_batch;
  in->dw_->size_[1] = num_in;
  in->dw_->size_[2] = 1;
  in->dw_->size_[3] = 1;
  out->dw_->size_[0] = num_batch;
  out->dw_->size_[1] = num_out;
  out->dw_->size_[2] = 1;
  out->dw_->size_[3] = 1;
  MulDeriv(in, filters, in->dw_, filters->dw_, out->dw_);
  in->size_ = in_size;
  in->dw_->size_ = in_dw_size;
  out->dw_->size_ = out_dw_size;

  kAddBiasDeriv << <256, 1>>>
      (biases->dw_->data_device_, out->dw_->data_device_, num_batch, num_out);
}

class ConvContextCudnn : public Context
{
 public:
  ConvContextCudnn()
  {
    CheckCudnn(cudnnCreateTensorDescriptor(&descr_tensor_src));
    CheckCudnn(cudnnCreateTensorDescriptor(&descr_tensor_dst));
    CheckCudnn(cudnnCreateTensorDescriptor(&descr_tensor_bias));
    CheckCudnn(cudnnCreateFilterDescriptor(&descr_filter));
    CheckCudnn(cudnnCreateConvolutionDescriptor(&descr_conv));
  }
  ~ConvContextCudnn()
  {
    CheckCudnn(cudnnDestroyConvolutionDescriptor(descr_conv));
    CheckCudnn(cudnnDestroyFilterDescriptor(descr_filter));
    CheckCudnn(cudnnDestroyTensorDescriptor(descr_tensor_src));
    CheckCudnn(cudnnDestroyTensorDescriptor(descr_tensor_dst));
    CheckCudnn(cudnnDestroyTensorDescriptor(descr_tensor_bias));
  }

  cudnnTensorDescriptor_t descr_tensor_src, descr_tensor_dst, descr_tensor_bias;
  cudnnFilterDescriptor_t descr_filter;
  cudnnConvolutionDescriptor_t descr_conv;

  cudnnConvolutionFwdAlgo_t algo;
  size_t size_in_bytes;
  cudnnConvolutionBwdFilterAlgo_t algo_filter;
  size_t size_in_bytes_bf;
  cudnnConvolutionBwdDataAlgo_t algo_bwd;
  size_t size_in_bytes_bd;
};

void ConvChooseAlgo(shared_ptr<ConvContextCudnn> &context)
{
  {
    printf("\t\tFwd\n");
    cudnnConvolutionFwdAlgo_t algo;
    CheckCudnn(cudnnGetConvolutionForwardAlgorithm(
        cudnn_handle, context->descr_tensor_src, context->descr_filter,
        context->descr_conv, context->descr_tensor_dst,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
    printf("Fastest is algo: %u\n", algo);

    int requested_algo_count = 5;
    int returned_algo_count[1];
    cudnnConvolutionFwdAlgoPerf_t *results =
        (cudnnConvolutionFwdAlgoPerf_t *)malloc(
            sizeof(cudnnConvolutionFwdAlgoPerf_t) * requested_algo_count);
    CheckCudnn(cudnnFindConvolutionForwardAlgorithm(
        cudnn_handle, context->descr_tensor_src, context->descr_filter,
        context->descr_conv, context->descr_tensor_dst, requested_algo_count,
        returned_algo_count, results));
    for (int idx = 0; idx < *returned_algo_count; ++idx)
    {
      cudnnStatus_t status = results[idx].status;
      if (status != CUDNN_STATUS_SUCCESS)
      {
        printf("%s algo %d\n", cudnnGetErrorString(results[idx].status),
               results[idx].algo);
      }
      else
      {
        printf("algo %d: %f time, requiring %llu memory Mb\n",
               results[idx].algo, results[idx].time,
               (unsigned long long)results[idx].memory / kMb);
      }
    }
    free(results);
  }

  {
    printf("\t\tBwd\n");
    cudnnConvolutionBwdFilterAlgo_t algo;
    CheckCudnn(cudnnGetConvolutionBackwardFilterAlgorithm(
        cudnn_handle, context->descr_tensor_src, context->descr_tensor_dst,
        context->descr_conv, context->descr_filter,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo));
    printf("Fastest is algo: %u\n", algo);

    int requested_algo_count = 5;
    int returned_algo_count[1];
    cudnnConvolutionBwdFilterAlgoPerf_t *results =
        (cudnnConvolutionBwdFilterAlgoPerf_t *)malloc(
            sizeof(cudnnConvolutionBwdFilterAlgoPerf_t) * requested_algo_count);
    CheckCudnn(cudnnFindConvolutionBackwardFilterAlgorithm(
        cudnn_handle, context->descr_tensor_src, context->descr_tensor_dst,
        context->descr_conv, context->descr_filter, requested_algo_count,
        returned_algo_count, results));
    for (int idx = 0; idx < *returned_algo_count; ++idx)
    {
      cudnnStatus_t status = results[idx].status;
      if (status != CUDNN_STATUS_SUCCESS)
      {
        printf("%s algo %d\n", cudnnGetErrorString(results[idx].status),
               results[idx].algo);
      }
      else
      {
        printf("algo %d: %f time, requiring %llu memory Mb\n",
               results[idx].algo, results[idx].time,
               (unsigned long long)results[idx].memory / kMb);
      }
    }
    free(results);
  }

  {
    printf("\t\tBwd data\n");
    cudnnConvolutionBwdDataAlgo_t algo;
    CheckCudnn(cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn_handle, context->descr_filter, context->descr_tensor_dst,
        context->descr_conv, context->descr_tensor_src,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo));
    printf("Fastest is algo: %u\n", algo);

    int requested_algo_count = 5;
    int returned_algo_count[1];
    cudnnConvolutionBwdDataAlgoPerf_t *results =
        (cudnnConvolutionBwdDataAlgoPerf_t *)malloc(
            sizeof(cudnnConvolutionBwdDataAlgoPerf_t) * requested_algo_count);
    CheckCudnn(cudnnFindConvolutionBackwardDataAlgorithm(
        cudnn_handle, context->descr_filter, context->descr_tensor_dst,
        context->descr_conv, context->descr_tensor_src, requested_algo_count,
        returned_algo_count, results));
    for (int idx = 0; idx < *returned_algo_count; ++idx)
    {
      cudnnStatus_t status = results[idx].status;
      if (status != CUDNN_STATUS_SUCCESS)
      {
        printf("%s algo %d\n", cudnnGetErrorString(results[idx].status),
               results[idx].algo);
      }
      else
      {
        printf("algo %d: %f time, requiring %llu memory Mb\n",
               results[idx].algo, results[idx].time,
               (unsigned long long)results[idx].memory / kMb);
      }
    }
    free(results);
  }
}

void MathCudnn::ConvSetUp(shared_ptr<Mat> &in_w, Params &params)
{
  int padding_x = params.padding_x;
  int padding_y = params.padding_y;
  int stride_x = params.stride_x;
  int stride_y = params.stride_y;
  int filter_width = params.filter_width;
  int filter_height = params.filter_height;
  int num_input = params.num_input;
  int num_filters = params.num_output;
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  int batch_size = in_w->size_[3];
  int out_width = params.out_width;
  int out_height = params.out_height;

  shared_ptr<ConvContextCudnn> context(new ConvContextCudnn);
  params.context = shared_ptr<Context>(context);

  SetTensorDescriptor(context->descr_tensor_src, batch_size, num_input,
                      in_height, in_width);

  CheckCudnn(cudnnSetFilter4dDescriptor(context->descr_filter, data_type,
                                        num_filters, num_input, filter_height,
                                        filter_width));

  CheckCudnn(cudnnSetConvolution2dDescriptor(context->descr_conv, padding_y,
                                             padding_x, stride_y, stride_x, 1,
                                             1, CUDNN_CROSS_CORRELATION));
  // find dimension of convolution output
  /*int tensor_ouput_dim[tensorDims];
  CheckCudnn(cudnnGetConvolutionNdForwardOutputDim(
      convDesc, srcTensorDesc, filterDesc, tensorDims, tensor_ouput_dim));
  n = tensor_ouput_dim[0];
  c = tensor_ouput_dim[1];
  h = tensor_ouput_dim[2];
  w = tensor_ouput_dim[3];*/

  SetTensorDescriptor(context->descr_tensor_dst, batch_size, num_filters,
                      out_height, out_width);
  SetTensorDescriptor(context->descr_tensor_bias, 1, num_filters, 1, 1);

  // ConvChooseAlgo(context);

  // Allow algorithms with small memory consumption.
  // context->algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  CheckCudnn(cudnnGetConvolutionForwardAlgorithm(
      cudnn_handle, context->descr_tensor_src, context->descr_filter,
      context->descr_conv, context->descr_tensor_dst,
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, work_space_size,
      &context->algo));

  context->size_in_bytes = 0;
  CheckCudnn(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn_handle, context->descr_tensor_src, context->descr_filter,
      context->descr_conv, context->descr_tensor_dst, context->algo,
      &context->size_in_bytes));

  // Allow algorithms with small memory consumption.
  // context->algo_filter = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  CheckCudnn(cudnnGetConvolutionBackwardFilterAlgorithm(
      cudnn_handle, context->descr_tensor_src, context->descr_tensor_dst,
      context->descr_conv, context->descr_filter,
      CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, work_space_size,
      &context->algo_filter));

  context->size_in_bytes_bf = 0;
  CheckCudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      cudnn_handle, context->descr_tensor_src, context->descr_tensor_dst,
      context->descr_conv, context->descr_filter, context->algo_filter,
      &context->size_in_bytes_bf));

  // Allow algorithms with small memory consumption.
  // context->algo_bwd = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  CheckCudnn(cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn_handle, context->descr_filter, context->descr_tensor_dst,
      context->descr_conv, context->descr_tensor_src,
      CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, work_space_size,
      &context->algo_bwd));

  context->size_in_bytes_bd = 0;
  CheckCudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(
      cudnn_handle, context->descr_filter, context->descr_tensor_dst,
      context->descr_conv, context->descr_tensor_src, context->algo_bwd,
      &context->size_in_bytes_bd));
}

void MathCudnn::Conv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &filters_w,
                     shared_ptr<Mat> &biases_w, shared_ptr<Mat> &out_w,
                     Params &params)
{
  ConvContextCudnn *context =
      static_cast<ConvContextCudnn *>(params.context.get());

  CheckCudnn(cudnnConvolutionForward(
      cudnn_handle, &kOne, context->descr_tensor_src, in_w->data_device_,
      context->descr_filter, filters_w->data_device_, context->descr_conv,
      context->algo, work_space, context->size_in_bytes, &kZero,
      context->descr_tensor_dst, out_w->data_device_));

  CheckCudnn(cudnnAddTensor_v3(cudnn_handle, &kOne, context->descr_tensor_bias,
                               biases_w->data_device_, &kOne,
                               context->descr_tensor_dst, out_w->data_device_));
}

void MathCudnn::ConvDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &filters_w,
                          shared_ptr<Mat> &biases_w, shared_ptr<Mat> &out_w,
                          Params &params)
{
  float *in_w_data = in_w->data_device_;
  float *in_dw_data = in_w->dw_->data_device_;
  float *filters_w_data = filters_w->data_device_;
  float *filters_dw_data = filters_w->dw_->data_device_;
  float *biases_dw_data = biases_w->dw_->data_device_;
  float *out_dw_data = out_w->dw_->data_device_;

  ConvContextCudnn *context =
      static_cast<ConvContextCudnn *>(params.context.get());

  CheckCudnn(cudnnConvolutionBackwardBias(
      cudnn_handle, &kOne, context->descr_tensor_dst, out_dw_data, &kOne,
      context->descr_tensor_bias, biases_dw_data));

  CheckCudnn(cudnnConvolutionBackwardFilter_v3(
      cudnn_handle, &kOne, context->descr_tensor_src, in_w_data,
      context->descr_tensor_dst, out_dw_data, context->descr_conv,
      context->algo_filter, work_space, context->size_in_bytes_bf, &kOne,
      context->descr_filter, filters_dw_data));

  CheckCudnn(cudnnConvolutionBackwardData_v3(
      cudnn_handle, &kOne, context->descr_filter, filters_w_data,
      context->descr_tensor_dst, out_dw_data, context->descr_conv,
      context->algo_bwd, work_space, context->size_in_bytes_bd, &kZero,
      context->descr_tensor_src, in_dw_data));
}

class PoolContextCudnn : public Context
{
 public:
  PoolContextCudnn()
  {
    CheckCudnn(cudnnCreateTensorDescriptor(&descr_tensor_src));
    CheckCudnn(cudnnCreateTensorDescriptor(&descr_tensor_dst));
    CheckCudnn(cudnnCreatePoolingDescriptor(&descr_pooling));
  }
  ~PoolContextCudnn()
  {
    CheckCudnn(cudnnDestroyPoolingDescriptor(descr_pooling));
    CheckCudnn(cudnnDestroyTensorDescriptor(descr_tensor_src));
    CheckCudnn(cudnnDestroyTensorDescriptor(descr_tensor_dst));
  }

  cudnnPoolingDescriptor_t descr_pooling;
  cudnnTensorDescriptor_t descr_tensor_src, descr_tensor_dst;
};

void MathCudnn::PoolSetUp(shared_ptr<Mat> &in_w, PoolType &type, Params &params)
{
  int padding_x = params.padding_x;
  int padding_y = params.padding_y;
  int stride_x = params.stride_x;
  int stride_y = params.stride_y;
  int filter_width = params.filter_width;
  int filter_height = params.filter_height;
  int out_width = params.out_width;
  int out_height = params.out_height;
  int num_filters = params.num_output;
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  int batch_size = in_w->size_[3];

  shared_ptr<PoolContextCudnn> context(new PoolContextCudnn);
  params.context = shared_ptr<Context>(context);

  cudnnPoolingMode_t mode;
  switch (type)
  {
    case MAX:
      mode = CUDNN_POOLING_MAX;
      break;
    case AVE:
      mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
      break;
  }

  CheckCudnn(cudnnSetPooling2dDescriptor(context->descr_pooling, mode,
                                         filter_height, filter_width, padding_y,
                                         padding_x, stride_y, stride_x));

  SetTensorDescriptor(context->descr_tensor_src, batch_size, num_filters,
                      in_height, in_width);

  /*const int tensorDims = 4;
  int tensor_ouput_dim[tensorDims] = {n, c, h, w};
  checkCUDNN(cudnnGetPoolingNdForwardOutputDim(poolingDesc, srcTensorDesc,
                                               tensorDims, tensor_ouput_dim));
  n = tensor_ouput_dim[0];
  c = tensor_ouput_dim[1];
  h = tensor_ouput_dim[2];
  w = tensor_ouput_dim[3];*/

  SetTensorDescriptor(context->descr_tensor_dst, batch_size, num_filters,
                      out_height, out_width);
}

void MathCudnn::MaxPool(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                        Params &params)
{
  PoolContextCudnn *context =
      static_cast<PoolContextCudnn *>(params.context.get());

  CheckCudnn(cudnnPoolingForward(cudnn_handle, context->descr_pooling, &kOne,
                                 context->descr_tensor_src, in_w->data_device_,
                                 &kZero, context->descr_tensor_dst,
                                 out_w->data_device_));
}

void MathCudnn::MaxPoolDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                             Params &params)
{
  shared_ptr<Mat> &in_dw = in_w->dw_;
  shared_ptr<Mat> &out_dw = out_w->dw_;

  PoolContextCudnn *context =
      static_cast<PoolContextCudnn *>(params.context.get());

  CheckCudnn(cudnnPoolingBackward(
      cudnn_handle, context->descr_pooling, &kOne, context->descr_tensor_dst,
      out_w->data_device_, context->descr_tensor_dst, out_dw->data_device_,
      context->descr_tensor_src, in_w->data_device_, &kZero,
      context->descr_tensor_src, in_dw->data_device_));
}

void MathCudnn::AvePool(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                        Params &params)
{
  PoolContextCudnn *context =
      static_cast<PoolContextCudnn *>(params.context.get());

  CheckCudnn(cudnnPoolingForward(cudnn_handle, context->descr_pooling, &kOne,
                                 context->descr_tensor_src, in_w->data_device_,
                                 &kZero, context->descr_tensor_dst,
                                 out_w->data_device_));
}

void MathCudnn::AvePoolDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                             Params &params)
{
  shared_ptr<Mat> &in_dw = in_w->dw_;
  shared_ptr<Mat> &out_dw = out_w->dw_;

  PoolContextCudnn *context =
      static_cast<PoolContextCudnn *>(params.context.get());

  CheckCudnn(cudnnPoolingBackward(
      cudnn_handle, context->descr_pooling, &kOne, context->descr_tensor_dst,
      out_w->data_device_, context->descr_tensor_dst, out_dw->data_device_,
      context->descr_tensor_src, in_w->data_device_, &kZero,
      context->descr_tensor_src, in_dw->data_device_));
}

// learning

static __global__ void kSGD(float *mat, float *mat_dw, float *mat_prev,
                            unsigned int len, float learning_rate,
                            int batch_size, float decay_rate)
{
  const float momentum = 0.9;

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int num_threads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += num_threads)
  {
    float curr_dw = mat_dw[i] / batch_size;
    float dw = momentum * mat_prev[i] + mat[i] * learning_rate * decay_rate +
               /*(1 - momentum) **/ learning_rate * curr_dw;
    mat_prev[i] = dw;
    mat_dw[i] = 0;
    mat[i] -= dw;
  }
}

static __global__ void kRMSProp(float *mat, float *mat_dw, float *mat_prev,
                                unsigned int len, float learning_rate,
                                int batch_size)
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
    float curr_dw = mat_dw[i] / batch_size;
    mat_prev[i] =
        decay_rate * mat_prev[i] + (1.0 - decay_rate) * curr_dw * curr_dw;

    // Gradient clip.
    if (curr_dw > clipval)
    {
      curr_dw = clipval;
    }
    if (curr_dw < -clipval)
    {
      curr_dw = -clipval;
    }

    // Update (and regularize).
    mat[i] += -learning_rate * curr_dw / sqrt(mat_prev[i] + smooth_eps) -
              regc * mat[i];

    mat_dw[i] = 0;
  }
}

void MathCudnn::SGD(shared_ptr<Mat> &mat, shared_ptr<Mat> &mat_prev,
                    float learning_rate, int batch_size, float decay_rate)
{
  kSGD << <NUM_BLOCKS, NUM_THREADS>>>
      (mat->data_device_, mat->dw_->data_device_, mat_prev->data_device_,
       mat->data_.size(), learning_rate, batch_size, decay_rate);
}

void MathCudnn::Rmsprop(shared_ptr<Mat> &mat, shared_ptr<Mat> &mat_prev,
                        float learning_rate, int batch_size)
{
  kRMSProp << <NUM_BLOCKS, NUM_THREADS>>>
      (mat->data_device_, mat->dw_->data_device_, mat_prev->data_device_,
       mat->data_.size(), learning_rate, batch_size);
}
