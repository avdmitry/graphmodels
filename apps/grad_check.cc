#include "utils.h"
#include "rnn.h"
#include "lstm.h"
#include "text_gen_utils.h"

using std::string;
using std::vector;
using std::shared_ptr;

int main(int argc, char *argv[])
{
  // srand(time(NULL));
  srand(3);

  math = shared_ptr<Math>(new MathCpu);
  math->Init();

  if (argc != 3)
  {
    printf("usage: file_with_sentences lstm/rnn\n");
    return -1;
  }
  string data_file_name(argv[1]);
  string model_type(argv[2]);

  printf("init data\n");
  shared_ptr<Data> data(new Data);
  LoadData(data_file_name, data);

  // Init model.
  vector<int> hs = {20, 20};
  int inout_size = data->vocab_.size() + 1;
  shared_ptr<Model> model;
  if (model_type == "rnn")
  {
    model = shared_ptr<Model>(new Rnn(kEmbedSize, hs, inout_size));
  }
  else if (model_type == "lstm")
  {
    model = shared_ptr<Model>(new Lstm(kEmbedSize, hs, inout_size));
  }
  else
  {
    printf("unknown model type\n");
    return -1;
  }

  printf("checking gradients\n");

  string sent("test sentence");
  model->graph_ = shared_ptr<Graph>(new Graph);
  float cost = CalcCost(model, sent, data);
  (void)cost;

  model->graph_->Backward();

  float eps = 0.1;

  int num_total = 0;
  int num_error = 0;
  int non_zero = 0;
  float error_sum = 0.0;
  for (size_t j = 0; j < model->params_.size(); ++j)
  {
    shared_ptr<MatWdw> &mat = model->params_[j];
    for (size_t i = 0; i < mat->w_->data_.size(); ++i)
    {
      float oldval = mat->w_->data_[i];
      mat->w_->data_[i] = oldval + eps;
      float c0 = CalcCost(model, sent, data);
      mat->w_->data_[i] = oldval - eps;
      float c1 = CalcCost(model, sent, data);
      mat->w_->data_[i] = oldval;

      num_total += 1;
      float grad_numeric = (c0 - c1) / (2 * eps);
      float grad_analitic = mat->dw_->data_[i];
      if (grad_numeric == 0 && grad_analitic == 0)
      {
        continue;
      }
      float mean = (grad_numeric + grad_analitic) / 2;
      float relerr = fabs((grad_numeric - grad_analitic) / mean);
      error_sum += relerr;
      non_zero += 1;
      if (relerr > 0.1)
      {
        num_error += 1;
        printf("%lu: numeric: %.9f, analytic: %.9f, relerr: %.9f\n", j,
               grad_numeric, grad_analitic, relerr);
      }
      else
      {
        printf(
            "ok| %lu: numeric: %.9f, analytic: %.9f, relerr: %.9f (%.9f "
            "%.9f)\n",
            j, grad_numeric, grad_analitic, relerr, c0, c1);
      }
    }
  }
  printf("num errors: %u from: %u, non_zero: %u, mean_error: %f\n", num_error,
         num_total, non_zero, error_sum / non_zero);

  return 0;
}
