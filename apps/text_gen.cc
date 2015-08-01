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
  if (model_type == "rnn")
  {
    model = shared_ptr<Model>(new rnn(kEmbedSize, hs, inout_size));
  }
  else if (model_type == "lstm")
  {
    model = shared_ptr<Model>(new lstm(kEmbedSize, hs, inout_size));
  }
  else
  {
    printf("unknown model type\n");
    return -1;
  }

  float cost_epoch = 0;
  int num_epoch = 0;
  clock_t begin_time = clock();
  for (int step = 0; step < 100000; ++step)
  {
    string sent = data->sentences_[Randi(0, data->sentences_.size() - 1)];

    shared_ptr<Graph> graph(new Graph);
    cost_epoch += CalcCost(graph, model, sent, data);

    graph->Backward();

    Learn(model);

    if (step % data->sentences_.size() == 0 && step != 0)
    {
      float time_epoch = float(clock() - begin_time) / CLOCKS_PER_SEC;
      cost_epoch /= data->sentences_.size();
      num_epoch += 1;
      printf("%u epoch, cost: %.3f, time: %.3f s\n", num_epoch, cost_epoch,
             time_epoch);
      cost_epoch = 0;

      printf("\tSamples:\n");
      string predict;
      vector<bool> sample_type = {true, true, true, true, true, false};
      for (int i = 0; i < sample_type.size(); ++i)
      {
        predict = PredictSentence(model, data, sample_type[i]);
        printf("\t%s\n", predict.c_str());
      }

      begin_time = clock();
    }
  }

  return 0;
}