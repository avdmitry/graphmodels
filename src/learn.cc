#include "learn.h"

using std::shared_ptr;

void LearnSGD(shared_ptr<Model> &model, float learning_rate, int batch_size)
{
  for (size_t j = 0; j < model->params_.size(); ++j)
  {
    shared_ptr<Mat> &mat = model->params_[j];
    math->SGD(mat, learning_rate, batch_size);
  }

  std::fill(model->output_->dw_->data_.begin(),
            model->output_->dw_->data_.end(), 0);
  math->MemoryClear(model->output_->dw_);
  model->graph_->ClearDw();
}

void LearnRmsprop(shared_ptr<Model> &model, float learning_rate, int batch_size)
{
  for (size_t j = 0; j < model->params_.size(); ++j)
  {
    shared_ptr<Mat> &mat = model->params_[j];
    shared_ptr<Mat> &mat_prev = model->params_prev_[j];
    math->Rmsprop(mat, mat_prev, learning_rate, batch_size);
  }

  std::fill(model->output_->dw_->data_.begin(),
            model->output_->dw_->data_.end(), 0);
  math->MemoryClear(model->output_->dw_);
  model->graph_->ClearDw();
}
