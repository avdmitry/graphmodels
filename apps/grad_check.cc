#include "utils.h"
#include "rnn.h"
#include "lstm.h"
#include "text_gen_utils.h"

using namespace std;

int main(int argc, char *argv[])
{
  // srand(time(NULL));
  srand(3);

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
  shared_ptr<Model> model1;
  shared_ptr<Model> model2;
  if (model_type == "rnn")
  {
    model = shared_ptr<Model>(new rnn(kEmbedSize, hs, inout_size));
    model1 = shared_ptr<Model>(new rnn(kEmbedSize, hs, inout_size));
    model2 = shared_ptr<Model>(new rnn(kEmbedSize, hs, inout_size));
  }
  else if (model_type == "lstm")
  {
    model = shared_ptr<Model>(new lstm(kEmbedSize, hs, inout_size));
    model1 = shared_ptr<Model>(new lstm(kEmbedSize, hs, inout_size));
    model2 = shared_ptr<Model>(new lstm(kEmbedSize, hs, inout_size));
  }
  else
  {
    printf("unknown model type\n");
    return -1;
  }

  printf("checking gradients\n");

  string sent("test sentence");
  shared_ptr<Graph> graph(new Graph);
  float cost = CalcCost(graph, model, sent, data);
  (void)cost;

  graph->Backward();

  float eps = 0.000001;

  for (size_t j = 0; j < model->params_.size(); ++j)
  {
    shared_ptr<Mat> &m = model->params_[j];
    for (size_t i = 0; i < m->w_.size(); ++i)
    {
      float oldval = m->w_[i];
      m->w_[i] = oldval + eps;
      shared_ptr<Graph> graph1(new Graph);
      float c0 = CalcCost(graph1, model1, sent, data);
      m->w_[i] = oldval - eps;
      shared_ptr<Graph> graph2(new Graph);
      float c1 = CalcCost(graph2, model2, sent, data);
      m->w_[i] = oldval;

      float gnum = (c0 - c1) / (2 * eps);
      float ganal = m->dw_[i];
      float relerr = (gnum - ganal) / (fabs(gnum) + fabs(ganal));
      if (relerr > 1e-1)
      {
        printf("%lu: numeric: %f, analytic: %f, err: %f\n", j, gnum, ganal,
               relerr);
      }
    }
  }

  return 0;
}
