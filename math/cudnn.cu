#include "cudnn.h"

#include <stdio.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <float.h>

#include "cpu.h"  // SgemmCpu, default implementation

using std::string;
using std::vector;
using std::shared_ptr;

static shared_ptr<MathCpu> math_cpu(new MathCpu);

const cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
const cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;

cudnnTensorDescriptor_t descr_tensor_src, descr_tensor_dst, descr_tensor_bias;
cudnnFilterDescriptor_t descr_filter;
cudnnConvolutionDescriptor_t descr_conv;
cudnnPoolingDescriptor_t descr_pooling;
cudnnHandle_t cudnn_handle;
cublasHandle_t cublas_handle;

inline void CheckCuda(cudaError_t status)
{
  if (status != 0)
  {
    printf("cuda error: %s\n", cudaGetErrorString(status));
  }
}

inline void CheckCudnn(cudnnStatus_t status)
{
  if (status != CUDNN_STATUS_SUCCESS)
  {
    printf("cudnn error: %s\n", cudnnGetErrorString(status));
  }
}

inline void CheckCublas(int status)
{
  if (status != 0)
  {
    printf("cublas error: %u\n", status);
  }
}

inline static int CopyToDevice(shared_ptr<Mat> &mat)
{
  size_t len = mat->size_[0] * mat->size_[1] * mat->size_[2] * mat->size_[3];

  if (mat->data_device_ == nullptr)
  {
    cudaError_t error =
        cudaMalloc((void **)&mat->data_device_, len * sizeof(float));
    if (error != cudaSuccess)
    {
      return -1;
    }
  }

  CheckCublas(cublasSetVector(len, sizeof(float), &mat->data_[0], 1,
                              mat->data_device_, 1));

  return 0;
}

inline static int CopyToHost(shared_ptr<Mat> &mat)
{
  size_t len = mat->size_[0] * mat->size_[1] * mat->size_[2] * mat->size_[3];

  CheckCublas(cublasGetVector(len, sizeof(float), mat->data_device_, 1,
                              &mat->data_[0], 1));

  return 0;
}

void MathCudnn::Init()
{
  CheckCudnn(cudnnCreate(&cudnn_handle));
  CheckCudnn(cudnnCreateTensorDescriptor(&descr_tensor_src));
  CheckCudnn(cudnnCreateTensorDescriptor(&descr_tensor_dst));
  CheckCudnn(cudnnCreateTensorDescriptor(&descr_tensor_bias));
  CheckCudnn(cudnnCreateFilterDescriptor(&descr_filter));
  CheckCudnn(cudnnCreateConvolutionDescriptor(&descr_conv));
  CheckCudnn(cudnnCreatePoolingDescriptor(&descr_pooling));

  CheckCublas(cublasCreate(&cublas_handle));

  cudaSetDevice(0);
}

void MathCudnn::Deinit()
{
  CheckCublas(cublasDestroy(cublas_handle));

  CheckCudnn(cudnnDestroyPoolingDescriptor(descr_pooling));
  CheckCudnn(cudnnDestroyConvolutionDescriptor(descr_conv));
  CheckCudnn(cudnnDestroyFilterDescriptor(descr_filter));
  CheckCudnn(cudnnDestroyTensorDescriptor(descr_tensor_src));
  CheckCudnn(cudnnDestroyTensorDescriptor(descr_tensor_dst));
  CheckCudnn(cudnnDestroyTensorDescriptor(descr_tensor_bias));
  CheckCudnn(cudnnDestroy(cudnn_handle));
}

int MathCudnn::FreeMatMemory(float *ptr)
{
  cudaFree(ptr);
  return 0;
}

int MathCudnn::Mul(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
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

    float alpha = 1.0f;
    float beta = 0.0f;
    CheckCublas(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                            &alpha, mat2->data_device_, mat2->size_[1],
                            mat1->data_device_, mat1->size_[1], &beta,
                            out->data_device_, out->size_[1]));

    CopyToHost(out);
  }

  return 0;
}

int MathCudnn::Add(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
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
    CopyToDevice(mat1);
    CopyToDevice(mat2);
    CopyToDevice(out);

    float alpha = 1.0f;
    float beta = 1.0f;
    cublasStatus_t status = cublasSgeam(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alpha,
        mat1->data_device_, mat1->size_[1], &beta, mat2->data_device_,
        mat2->size_[1], out->data_device_, out->size_[1]);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
      return -1;
    }

    CopyToHost(out);
  }

  return 0;
}

int MathCudnn::ElmtMul(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                       shared_ptr<Mat> &out)
{
  CopyToDevice(mat1);
  CopyToDevice(mat2);
  CopyToDevice(out);

  int len = mat1->size_[0] * mat1->size_[1] * mat1->size_[2] * mat1->size_[3];

  float alpha = 1.0f;
  float beta = 0.0f;
  cublasStatus_t status = cublasSgbmv(
      cublas_handle, CUBLAS_OP_N, len, len, 0, 0, &alpha, mat1->data_device_, 1,
      mat2->data_device_, 1, &beta, out->data_device_, 1);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    return -1;
  }

  CopyToHost(out);

  return 0;
}

int MathCudnn::AddDeriv(shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                        shared_ptr<Mat> &out)
{
  return math_cpu->AddDeriv(mat1d, mat2d, out);
}

int MathCudnn::ElmtMulDeriv(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                            shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                            shared_ptr<Mat> &out)
{
  return math_cpu->ElmtMulDeriv(mat1, mat2, mat1d, mat2d, out);
}

int MathCudnn::MulDeriv(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                        shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                        shared_ptr<Mat> &out)
{
  int m = mat1d->size_[0];
  int n = mat1d->size_[1];
  int k = mat2->size_[1];
  float alpha = 1.0f;
  float beta = 0.0f;
  // SgemmCpu(true, false, true, m, n, k, alpha, &out->data_[0], k,
  //         &mat2->data_[0], k, beta, &mat1d->data_[0], n);

  CopyToDevice(mat1);
  CopyToDevice(mat1d);
  CopyToDevice(mat2);
  CopyToDevice(mat2d);
  CopyToDevice(out);

  CheckCublas(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                          &alpha, mat2->data_device_, k, out->data_device_,
                          out->size_[1], &beta, mat1d->data_device_, n));

  m = mat2d->size_[1];
  n = mat1->size_[1];
  k = mat1->size_[0];
  alpha = 1.0f;
  beta = 0.0f;
  // SgemmCpu(false, false, true, m, n, k, alpha, &out->data_[0], m,
  //         &mat1->data_[0], n, beta, &mat2d->data_[0], m);

  CheckCublas(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                          &alpha, out->data_device_, m, mat1->data_device_, n,
                          &beta, mat2d->data_device_, m));

  CopyToHost(mat1d);
  CopyToHost(mat2d);

  return 0;
}

int MathCudnn::Relu(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w)
{
  CopyToDevice(in_w);
  CopyToDevice(out_w);

  if (in_w->size_[0] != out_w->size_[0] || in_w->size_[1] != out_w->size_[1])
  {
    return -1;
  }

  int n = in_w->size_[3];
  int c = in_w->size_[2];
  int h = in_w->size_[1];
  int w = in_w->size_[0];

  CheckCudnn(cudnnSetTensor4dDescriptor(descr_tensor_src, tensor_format,
                                        data_type, n, c, h, w));
  CheckCudnn(cudnnSetTensor4dDescriptor(descr_tensor_dst, tensor_format,
                                        data_type, n, c, h, w));

  float alpha = 1;
  float beta = 0;
  CheckCudnn(cudnnActivationForward(cudnn_handle, CUDNN_ACTIVATION_RELU, &alpha,
                                    descr_tensor_src, in_w->data_device_, &beta,
                                    descr_tensor_dst, out_w->data_device_));

  CopyToHost(out_w);

  return 0;
}

int MathCudnn::Sigm(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w)
{
  CopyToDevice(in_w);
  CopyToDevice(out_w);

  if (in_w->size_[0] != out_w->size_[0] || in_w->size_[1] != out_w->size_[1])
  {
    return -1;
  }

  int n = in_w->size_[3];
  int c = in_w->size_[2];
  int h = in_w->size_[1];
  int w = in_w->size_[0];

  CheckCudnn(cudnnSetTensor4dDescriptor(descr_tensor_src, tensor_format,
                                        data_type, n, c, h, w));
  CheckCudnn(cudnnSetTensor4dDescriptor(descr_tensor_dst, tensor_format,
                                        data_type, n, c, h, w));

  float alpha = 1;
  float beta = 0;
  CheckCudnn(cudnnActivationForward(
      cudnn_handle, CUDNN_ACTIVATION_SIGMOID, &alpha, descr_tensor_src,
      in_w->data_device_, &beta, descr_tensor_dst, out_w->data_device_));

  CopyToHost(out_w);

  return 0;
}

int MathCudnn::Tanh(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w)
{
  CopyToDevice(in_w);
  CopyToDevice(out_w);

  if (in_w->size_[0] != out_w->size_[0] || in_w->size_[1] != out_w->size_[1])
  {
    return -1;
  }

  int n = in_w->size_[3];
  int c = in_w->size_[2];
  int h = in_w->size_[1];
  int w = in_w->size_[0];

  CheckCudnn(cudnnSetTensor4dDescriptor(descr_tensor_src, tensor_format,
                                        data_type, n, c, h, w));
  CheckCudnn(cudnnSetTensor4dDescriptor(descr_tensor_dst, tensor_format,
                                        data_type, n, c, h, w));

  float alpha = 1;
  float beta = 0;
  CheckCudnn(cudnnActivationForward(cudnn_handle, CUDNN_ACTIVATION_TANH, &alpha,
                                    descr_tensor_src, in_w->data_device_, &beta,
                                    descr_tensor_dst, out_w->data_device_));

  CopyToHost(out_w);

  return 0;
}

int MathCudnn::ReluDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &in_dw,
                         shared_ptr<Mat> &out_w, shared_ptr<Mat> &out_dw)
{
  CopyToDevice(in_w);
  CopyToDevice(in_dw);
  CopyToDevice(out_w);
  CopyToDevice(out_dw);

  if (out_dw->size_[0] != out_w->size_[0] ||
      out_dw->size_[1] != out_w->size_[1] ||
      out_dw->size_[0] != in_dw->size_[0] ||
      out_dw->size_[1] != in_dw->size_[1])
  {
    return -1;
  }

  int n = in_w->size_[3];
  int c = in_w->size_[2];
  int h = in_w->size_[1];
  int w = in_w->size_[0];

  CheckCudnn(cudnnSetTensor4dDescriptor(descr_tensor_src, tensor_format,
                                        data_type, n, c, h, w));
  CheckCudnn(cudnnSetTensor4dDescriptor(descr_tensor_dst, tensor_format,
                                        data_type, n, c, h, w));

  float alpha = 1;
  float beta = 0;
  CheckCudnn(cudnnActivationBackward(
      cudnn_handle, CUDNN_ACTIVATION_RELU, &alpha, descr_tensor_src,
      out_w->data_device_, descr_tensor_src, out_dw->data_device_,
      descr_tensor_dst, in_w->data_device_, &beta, descr_tensor_dst,
      in_dw->data_device_));

  CopyToHost(in_dw);

  return 0;
}

int MathCudnn::SigmDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &in_dw,
                         shared_ptr<Mat> &out_w, shared_ptr<Mat> &out_dw)
{
  CopyToDevice(in_w);
  CopyToDevice(in_dw);
  CopyToDevice(out_w);
  CopyToDevice(out_dw);

  if (out_dw->size_[0] != out_w->size_[0] ||
      out_dw->size_[1] != out_w->size_[1] ||
      out_dw->size_[0] != in_dw->size_[0] ||
      out_dw->size_[1] != in_dw->size_[1])
  {
    return -1;
  }

  int n = in_w->size_[3];
  int c = in_w->size_[2];
  int h = in_w->size_[1];
  int w = in_w->size_[0];

  CheckCudnn(cudnnSetTensor4dDescriptor(descr_tensor_src, tensor_format,
                                        data_type, n, c, h, w));
  CheckCudnn(cudnnSetTensor4dDescriptor(descr_tensor_dst, tensor_format,
                                        data_type, n, c, h, w));

  float alpha = 1;
  float beta = 0;
  CheckCudnn(cudnnActivationBackward(
      cudnn_handle, CUDNN_ACTIVATION_SIGMOID, &alpha, descr_tensor_src,
      out_w->data_device_, descr_tensor_src, out_dw->data_device_,
      descr_tensor_dst, in_w->data_device_, &beta, descr_tensor_dst,
      in_dw->data_device_));

  CopyToHost(in_dw);

  return 0;
}

int MathCudnn::TanhDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &in_dw,
                         shared_ptr<Mat> &out_w, shared_ptr<Mat> &out_dw)
{
  CopyToDevice(in_w);
  CopyToDevice(in_dw);
  CopyToDevice(out_w);
  CopyToDevice(out_dw);

  if (out_dw->size_[0] != out_w->size_[0] ||
      out_dw->size_[1] != out_w->size_[1] ||
      out_dw->size_[0] != in_dw->size_[0] ||
      out_dw->size_[1] != in_dw->size_[1])
  {
    return -1;
  }

  int n = in_w->size_[3];
  int c = in_w->size_[2];
  int h = in_w->size_[1];
  int w = in_w->size_[0];

  CheckCudnn(cudnnSetTensor4dDescriptor(descr_tensor_src, tensor_format,
                                        data_type, n, c, h, w));
  CheckCudnn(cudnnSetTensor4dDescriptor(descr_tensor_dst, tensor_format,
                                        data_type, n, c, h, w));

  float alpha = 1;
  float beta = 0;
  CheckCudnn(cudnnActivationBackward(
      cudnn_handle, CUDNN_ACTIVATION_TANH, &alpha, descr_tensor_src,
      out_w->data_device_, descr_tensor_src, out_dw->data_device_,
      descr_tensor_dst, in_w->data_device_, &beta, descr_tensor_dst,
      in_dw->data_device_));

  CopyToHost(in_dw);

  return 0;
}

shared_ptr<Mat> MathCudnn::Softmax(shared_ptr<Mat> &mat)
{
  return math_cpu->Softmax(mat);
}

int MathCudnn::Fc(shared_ptr<Mat> &in, shared_ptr<Mat> &filters,
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

  // math->Add(biases, out, out);
  for (int batch = 0; batch < num_batch; ++batch)
  {
    int out_offset = num_out * batch;
    for (int i = 0; i < num_out; ++i)
    {
      out->data_[out_offset + i] += biases->data_[i];
    }
  }

  return 0;
}

int MathCudnn::FcDeriv(shared_ptr<Mat> &in, shared_ptr<Mat> &filters,
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

  // math->AddDeriv(biases->dw_, out->dw_, out->dw_);
  for (int batch = 0; batch < num_batch; ++batch)
  {
    int out_offset = num_out * batch;
    for (int i = 0; i < num_out; ++i)
    {
      float dw = out->dw_->data_[out_offset + i];
      biases->dw_->data_[i] += dw;
    }
  }

  return 0;
}

int MathCudnn::Conv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &filters_w,
                    shared_ptr<Mat> &biases_w, shared_ptr<Mat> &out_w,
                    ConvParams &conv_params)
{
  int padding_x = conv_params.padding_x;
  int padding_y = conv_params.padding_y;
  int stride_x = conv_params.stride_x;
  int stride_y = conv_params.stride_y;
  int filter_width = conv_params.filter_width;
  int filter_height = conv_params.filter_height;
  int num_input = in_w->size_[2];
  int num_filters = filters_w->size_[3];
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  int batch_size = in_w->size_[3];

  int out_width = (in_width + padding_x * 2 - filter_width) / stride_x + 1;
  int out_height = (in_height + padding_y * 2 - filter_height) / stride_y + 1;

  CopyToDevice(in_w);
  CopyToDevice(filters_w);
  CopyToDevice(biases_w);
  CopyToDevice(out_w);

  int n = batch_size;
  int c = num_input;
  int h = in_height;
  int w = in_width;
  // printf("conv %u %u %u %u\n", n, c, h, w);
  cudnnSetTensor4dDescriptor(descr_tensor_src, tensor_format, data_type, n, c,
                             h, w);

  static const int kDims = 4;
  const int filterDimA[kDims] = {num_filters, c, filter_height, filter_width};
  CheckCudnn(
      cudnnSetFilterNdDescriptor(descr_filter, data_type, kDims, filterDimA));

  static const int kConvDims = 2;
  int padding[kConvDims] = {padding_x, padding_y};
  int stride[kConvDims] = {stride_x, stride_y};
  int upscale[kConvDims] = {1, 1};  // _v3 TODO
  CheckCudnn(cudnnSetConvolutionNdDescriptor(descr_conv, kConvDims, padding,
                                             stride, upscale,
                                             CUDNN_CROSS_CORRELATION));
  // find dimension of convolution output
  /*int tensor_ouput_dim[tensorDims];
  CheckCudnn(cudnnGetConvolutionNdForwardOutputDim(
      convDesc, srcTensorDesc, filterDesc, tensorDims, tensor_ouput_dim));
  n = tensor_ouput_dim[0];
  c = tensor_ouput_dim[1];
  h = tensor_ouput_dim[2];
  w = tensor_ouput_dim[3];*/

  n = batch_size;
  c = num_filters;
  h = out_height;
  w = out_width;
  // printf("%u %u %u %u\n", n, c, h, w);
  cudnnSetTensor4dDescriptor(descr_tensor_dst, tensor_format, data_type, n, c,
                             h, w);
  cudnnSetTensor4dDescriptor(descr_tensor_bias, tensor_format, data_type, 1, c,
                             1, 1);

  /*
    // Choose the best algo according to the preference
    cout << "Testing cudnnGetConvolutionForwardAlgorithm ...\n";
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
        cudnnHandle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
    cout << "Fastest algorithm is Algo " << algo << "\n";
    convAlgorithm = algo;
    // New way of finding the fastest config
    // Setup for findFastest call
    cout << "Testing cudnnFindConvolutionForwardAlgorithm ...\n";
    int requestedAlgoCount = 5;
    int returnedAlgoCount[1];
    cudnnConvolutionFwdAlgoPerf_t* results =
        (cudnnConvolutionFwdAlgoPerf_t*)malloc(
            sizeof(cudnnConvolutionFwdAlgoPerf_t) * requestedAlgoCount);
    checkCUDNN(cudnnFindConvolutionForwardAlgorithm(
        cudnnHandle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
        requestedAlgoCount, returnedAlgoCount, results));
    for (int algoIndex = 0; algoIndex < *returnedAlgoCount; ++algoIndex)
    {
      printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n",
             cudnnGetErrorString(results[algoIndex].status),
             results[algoIndex].algo, results[algoIndex].time,
             (unsigned long long)results[algoIndex].memory);
    }
    free(results);*/

  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  size_t sizeInBytes = 0;
  void *workSpace = NULL;
  CheckCudnn(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn_handle, descr_tensor_src, descr_filter, descr_conv,
      descr_tensor_dst, algo, &sizeInBytes));
  if (sizeInBytes != 0)
  {
    CheckCuda(cudaMalloc(&workSpace, sizeInBytes));
  }

  float alpha = 1;
  float beta = 0;  // 1
  CheckCudnn(cudnnConvolutionForward(
      cudnn_handle, &alpha, descr_tensor_src, in_w->data_device_, descr_filter,
      filters_w->data_device_, descr_conv, algo, workSpace, sizeInBytes, &beta,
      descr_tensor_dst, out_w->data_device_));
  if (sizeInBytes != 0)
  {
    CheckCuda(cudaFree(workSpace));
  }

  float bias_alpha = 1;
  float bias_beta = 1;
  CheckCudnn(cudnnAddTensor(cudnn_handle, CUDNN_ADD_SAME_C, &bias_alpha,
                            descr_tensor_bias, biases_w->data_device_,
                            &bias_beta, descr_tensor_dst, out_w->data_device_));

  CopyToHost(out_w);

  return 0;
}

int MathCudnn::ConvDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &in_dw,
                         shared_ptr<Mat> &filters_w,
                         shared_ptr<Mat> &filters_dw,
                         shared_ptr<Mat> &biases_dw, shared_ptr<Mat> &out_w,
                         shared_ptr<Mat> &out_dw, ConvParams &conv_params)
{
  int padding_x = conv_params.padding_x;
  int padding_y = conv_params.padding_y;
  int stride_x = conv_params.stride_x;
  int stride_y = conv_params.stride_y;
  int filter_width = conv_params.filter_width;
  int filter_height = conv_params.filter_height;
  int num_input = in_w->size_[2];
  int num_filters = filters_w->size_[3];
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  int batch_size = in_w->size_[3];

  int out_width = (in_width + padding_x * 2 - filter_width) / stride_x + 1;
  int out_height = (in_height + padding_y * 2 - filter_height) / stride_y + 1;

  CopyToDevice(in_w);
  CopyToDevice(in_dw);
  CopyToDevice(filters_w);
  CopyToDevice(filters_dw);
  CopyToDevice(biases_dw);
  CopyToDevice(out_dw);

  int n = batch_size;
  int c = num_filters;
  int h = out_height;
  int w = out_width;
  // printf("convderiv %u %u %u %u\n", n, c, h, w);
  cudnnSetTensor4dDescriptor(descr_tensor_src, tensor_format, data_type, n, c,
                             h, w);
  cudnnSetTensor4dDescriptor(descr_tensor_bias, tensor_format, data_type, 1, c,
                             1, 1);

  static const int kDims = 4;
  const int filterDimA[kDims] = {num_filters, num_input, filter_height,
                                 filter_width};
  CheckCudnn(
      cudnnSetFilterNdDescriptor(descr_filter, data_type, kDims, filterDimA));

  static const int kConvDims = 2;
  int padding[kConvDims] = {padding_x, padding_y};
  int stride[kConvDims] = {stride_x, stride_y};
  int upscale[kConvDims] = {1, 1};  // _v3 TODO
  CheckCudnn(cudnnSetConvolutionNdDescriptor(descr_conv, kConvDims, padding,
                                             stride, upscale,
                                             CUDNN_CROSS_CORRELATION));

  n = batch_size;
  c = num_input;
  h = in_height;
  w = in_width;
  // printf("%u %u %u %u\n", n, c, h, w);
  cudnnSetTensor4dDescriptor(descr_tensor_dst, tensor_format, data_type, n, c,
                             h, w);

  // get workspace for backwards filter algorithm
  cudnnConvolutionBwdFilterAlgo_t algo_filter =
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  size_t size_in_bytes_f = 0;
  void *work_space_f = NULL;
  CheckCudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      cudnn_handle, descr_tensor_dst, descr_tensor_src, descr_conv,
      descr_filter, algo_filter, &size_in_bytes_f));
  // printf("filters size: %u\n", sizeInBytesF);
  if (size_in_bytes_f != 0)
  {
    CheckCuda(cudaMalloc(&work_space_f, size_in_bytes_f));
  }

  float bias_alpha = 1;
  float bias_beta = 1;
  CheckCudnn(cudnnConvolutionBackwardBias(
      cudnn_handle, &bias_alpha, descr_tensor_src, out_dw->data_device_,
      &bias_beta, descr_tensor_bias, biases_dw->data_device_));

  float alpha_f = 1;
  float beta_f = 1;
  CheckCudnn(cudnnConvolutionBackwardFilter_v3(
      cudnn_handle, &alpha_f, descr_tensor_dst, in_w->data_device_,
      descr_tensor_src, out_dw->data_device_, descr_conv, algo_filter,
      work_space_f, size_in_bytes_f, &beta_f, descr_filter,
      filters_dw->data_device_));

  if (size_in_bytes_f != 0)
  {
    CheckCuda(cudaFree(work_space_f));
  }

  cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  size_t size_in_bytes = 0;
  void *work_space = NULL;
  CheckCudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(
      cudnn_handle, descr_filter, descr_tensor_src, descr_conv,
      descr_tensor_dst, algo, &size_in_bytes));
  // printf("data size: %u\n", sizeInBytes);
  if (size_in_bytes != 0)
  {
    CheckCuda(cudaMalloc(&work_space, size_in_bytes));
  }

  float alpha = 1;
  float beta = 0;  // 1
  CheckCudnn(cudnnConvolutionBackwardData_v3(
      cudnn_handle, &alpha, descr_filter, filters_w->data_device_,
      descr_tensor_src, out_dw->data_device_, descr_conv, algo, work_space,
      size_in_bytes, &beta, descr_tensor_dst, in_dw->data_device_));
  if (size_in_bytes != 0)
  {
    CheckCuda(cudaFree(work_space));
  }

  CopyToHost(in_dw);
  CopyToHost(filters_dw);
  CopyToHost(biases_dw);

  return 0;
}

int MathCudnn::MaxPool(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                       ConvParams &conv_params)
{
  int padding_x = conv_params.padding_x;
  int padding_y = conv_params.padding_y;
  int stride_x = conv_params.stride_x;
  int stride_y = conv_params.stride_y;
  int filter_width = conv_params.filter_width;
  int filter_height = conv_params.filter_height;
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  int num_filters = in_w->size_[2];
  int batch_size = in_w->size_[3];

  int out_width = (in_width + padding_x * 2 - filter_width) / stride_x + 1;
  int out_height = (in_height + padding_y * 2 - filter_height) / stride_y + 1;

  CopyToDevice(in_w);
  CopyToDevice(out_w);

  int n = batch_size;
  int c = num_filters;
  int h = in_height;
  int w = in_width;
  // printf("%u %u %u %u\n", n, c, h, w);

  static const int kDims = 2;
  int filter_dim[kDims] = {filter_width, filter_height};
  int padding[kDims] = {padding_x, padding_y};
  int stride[kDims] = {stride_x, stride_y};
  CheckCudnn(cudnnSetPoolingNdDescriptor(descr_pooling, CUDNN_POOLING_MAX,
                                         kDims, filter_dim, padding, stride));

  cudnnSetTensor4dDescriptor(descr_tensor_src, tensor_format, data_type, n, c,
                             h, w);

  /*const int tensorDims = 4;
  int tensor_ouput_dim[tensorDims] = {n, c, h, w};
  checkCUDNN(cudnnGetPoolingNdForwardOutputDim(poolingDesc, srcTensorDesc,
                                               tensorDims, tensor_ouput_dim));
  n = tensor_ouput_dim[0];
  c = tensor_ouput_dim[1];
  h = tensor_ouput_dim[2];
  w = tensor_ouput_dim[3];*/

  n = batch_size;
  c = num_filters;
  h = out_height;
  w = out_width;
  // printf("%u %u %u %u\n", n, c, h, w);

  cudnnSetTensor4dDescriptor(descr_tensor_dst, tensor_format, data_type, n, c,
                             h, w);
  float alpha = 1;
  float beta = 0;
  CheckCudnn(cudnnPoolingForward(cudnn_handle, descr_pooling, &alpha,
                                 descr_tensor_src, in_w->data_device_, &beta,
                                 descr_tensor_dst, out_w->data_device_));

  CopyToHost(out_w);

  return 0;
}

int MathCudnn::MaxPoolDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &in_dw,
                            shared_ptr<Mat> &out_w, shared_ptr<Mat> &out_dw,
                            ConvParams &conv_params)
{
  int padding_x = conv_params.padding_x;
  int padding_y = conv_params.padding_y;
  int stride_x = conv_params.stride_x;
  int stride_y = conv_params.stride_y;
  int filter_width = conv_params.filter_width;
  int filter_height = conv_params.filter_height;
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  int num_filters = in_w->size_[2];
  int batch_size = in_w->size_[3];

  int out_width = (in_width + padding_x * 2 - filter_width) / stride_x + 1;
  int out_height = (in_height + padding_y * 2 - filter_height) / stride_y + 1;

  CopyToDevice(in_w);
  CopyToDevice(in_dw);
  CopyToDevice(out_w);
  CopyToDevice(out_dw);

  int n = batch_size;
  int c = num_filters;
  int h = out_height;
  int w = out_width;
  // printf("%u %u %u %u\n", n, c, h, w);

  static const int kDims = 2;
  int filter_dim[kDims] = {filter_width, filter_height};
  int padding[kDims] = {padding_x, padding_y};
  int stride[kDims] = {stride_x, stride_y};
  CheckCudnn(cudnnSetPoolingNdDescriptor(descr_pooling, CUDNN_POOLING_MAX,
                                         kDims, filter_dim, padding, stride));

  cudnnSetTensor4dDescriptor(descr_tensor_src, tensor_format, data_type, n, c,
                             h, w);

  n = batch_size;
  c = num_filters;
  h = in_height;
  w = in_width;
  // printf("%u %u %u %u\n", n, c, h, w);

  cudnnSetTensor4dDescriptor(descr_tensor_dst, tensor_format, data_type, n, c,
                             h, w);
  float alpha = 1;
  float beta = 0;
  CheckCudnn(cudnnPoolingBackward(cudnn_handle, descr_pooling, &alpha,
                                  descr_tensor_src, out_w->data_device_,
                                  descr_tensor_src, out_dw->data_device_,
                                  descr_tensor_dst, in_w->data_device_, &beta,
                                  descr_tensor_dst, in_dw->data_device_));

  CopyToHost(in_dw);

  return 0;
}

int MathCudnn::AvePool(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                       ConvParams &conv_params)
{
  int padding_x = conv_params.padding_x;
  int padding_y = conv_params.padding_y;
  int stride_x = conv_params.stride_x;
  int stride_y = conv_params.stride_y;
  int filter_width = conv_params.filter_width;
  int filter_height = conv_params.filter_height;
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  int num_filters = in_w->size_[2];
  int batch_size = in_w->size_[3];

  int out_width = (in_width + padding_x * 2 - filter_width) / stride_x + 1;
  int out_height = (in_height + padding_y * 2 - filter_height) / stride_y + 1;

  CopyToDevice(in_w);
  CopyToDevice(out_w);

  int n = batch_size;
  int c = num_filters;
  int h = in_height;
  int w = in_width;
  // printf("%u %u %u %u\n", n, c, h, w);

  static const int kDims = 2;
  int filter_dim[kDims] = {filter_width, filter_height};
  int padding[kDims] = {padding_x, padding_y};
  int stride[kDims] = {stride_x, stride_y};
  CheckCudnn(cudnnSetPoolingNdDescriptor(
      descr_pooling, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, kDims,
      filter_dim, padding, stride));

  cudnnSetTensor4dDescriptor(descr_tensor_src, tensor_format, data_type, n, c,
                             h, w);

  /*const int tensorDims = 4;
  int tensor_ouput_dim[tensorDims] = {n, c, h, w};
  checkCUDNN(cudnnGetPoolingNdForwardOutputDim(poolingDesc, srcTensorDesc,
                                               tensorDims, tensor_ouput_dim));
  n = tensor_ouput_dim[0];
  c = tensor_ouput_dim[1];
  h = tensor_ouput_dim[2];
  w = tensor_ouput_dim[3];*/

  n = batch_size;
  c = num_filters;
  h = out_height;
  w = out_width;
  // printf("%u %u %u %u\n", n, c, h, w);

  cudnnSetTensor4dDescriptor(descr_tensor_dst, tensor_format, data_type, n, c,
                             h, w);
  float alpha = 1;
  float beta = 0;
  CheckCudnn(cudnnPoolingForward(cudnn_handle, descr_pooling, &alpha,
                                 descr_tensor_src, in_w->data_device_, &beta,
                                 descr_tensor_dst, out_w->data_device_));

  CopyToHost(out_w);

  return 0;
}

int MathCudnn::AvePoolDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &in_dw,
                            shared_ptr<Mat> &out_w, shared_ptr<Mat> &out_dw,
                            ConvParams &conv_params)
{
  int padding_x = conv_params.padding_x;
  int padding_y = conv_params.padding_y;
  int stride_x = conv_params.stride_x;
  int stride_y = conv_params.stride_y;
  int filter_width = conv_params.filter_width;
  int filter_height = conv_params.filter_height;
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  int num_filters = in_w->size_[2];
  int batch_size = in_w->size_[3];

  int out_width = (in_width + padding_x * 2 - filter_width) / stride_x + 1;
  int out_height = (in_height + padding_y * 2 - filter_height) / stride_y + 1;

  CopyToDevice(in_w);
  CopyToDevice(in_dw);
  CopyToDevice(out_w);
  CopyToDevice(out_dw);

  int n = batch_size;
  int c = num_filters;
  int h = out_height;
  int w = out_width;
  // printf("%u %u %u %u\n", n, c, h, w);

  static const int kDims = 2;
  int filter_dim[kDims] = {filter_width, filter_height};
  int padding[kDims] = {padding_x, padding_y};
  int stride[kDims] = {stride_x, stride_y};
  CheckCudnn(cudnnSetPoolingNdDescriptor(
      descr_pooling, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, kDims,
      filter_dim, padding, stride));

  cudnnSetTensor4dDescriptor(descr_tensor_src, tensor_format, data_type, n, c,
                             h, w);

  n = batch_size;
  c = num_filters;
  h = in_height;
  w = in_width;
  // printf("%u %u %u %u\n", n, c, h, w);

  cudnnSetTensor4dDescriptor(descr_tensor_dst, tensor_format, data_type, n, c,
                             h, w);
  float alpha = 1;
  float beta = 0;
  CheckCudnn(cudnnPoolingBackward(cudnn_handle, descr_pooling, &alpha,
                                  descr_tensor_src, out_w->data_device_,
                                  descr_tensor_src, out_dw->data_device_,
                                  descr_tensor_dst, in_w->data_device_, &beta,
                                  descr_tensor_dst, in_dw->data_device_));

  CopyToHost(in_dw);

  return 0;
}
