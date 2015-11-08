#include "mat.h"

#include "common.h"

Mat::~Mat()
{
  if (data_device_ != nullptr)
  {
    math->FreeMatMemory(data_device_);
    data_device_ = nullptr;
  }
}
