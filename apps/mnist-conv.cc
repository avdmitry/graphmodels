#include "utils.h"
#include "layers.h"
#include "learn.h"
#include "datasets/mnist.h"

using std::string;
using std::vector;
using std::shared_ptr;

// 3.333 epoch| cost: 0.022| test acc: 0.990

class CnnNet : public Model
{
 public:
  CnnNet(int num_input_x, int num_input_y, int num_output)
  {
    static const int filter1_x = 3;
    static const int filter1_y = 3;
    static const int num_filters1 = 8;
    static const int filter2_x = 3;
    static const int filter2_y = 3;
    static const int num_filters2 = 16;

    graph_ = shared_ptr<Graph>(new Graph);
    input_ = shared_ptr<Mat>(new Mat(num_input_x, num_input_y));

    shared_ptr<Mat> conv1, rel1, mp1;
    graph_->Process(shared_ptr<Object>(new ConvLayer(
        input_, &conv1, num_filters1, filter1_x, filter1_y, 1, 1, 1, 1)));
    graph_->Process(shared_ptr<Object>(new ReluOp(conv1, &rel1)));
    graph_->Process(
        shared_ptr<Object>(new MaxPoolLayer(rel1, &mp1, 3, 3, 1, 1, 2, 2)));

    shared_ptr<Mat> conv2, rel2, mp2;
    graph_->Process(shared_ptr<Object>(new ConvLayer(
        mp1, &conv2, num_filters2, filter2_x, filter2_y, 1, 1, 1, 1)));
    graph_->Process(shared_ptr<Object>(new ReluOp(conv2, &rel2)));
    graph_->Process(
        shared_ptr<Object>(new MaxPoolLayer(rel2, &mp2, 3, 3, 1, 1, 2, 2)));

    graph_->Process(shared_ptr<Object>(new FCLayer(mp2, &output_, num_output)));

    graph_->GetParams(params_);
    for (size_t i = 0; i < params_.size(); ++i)
    {
      shared_ptr<Mat> &mat = params_[i];
      params_prev_.emplace_back(new Mat(mat->size_));
    }
  }

  // Virtual functions stubs.
  void Create(int idx)
  {
  }

  void ClearPrevState()
  {
  }
};

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    printf("usage: mnist_data_path\n");
    return -1;
  }
  string data_path(argv[1]);

  // srand(time(NULL));
  srand(0);

  math = shared_ptr<Math>(new MathCpu);
  // math = shared_ptr<Math>(new MathCudnn);
  math->Init();

  datasets::Mnist mnist(data_path);
  vector<shared_ptr<datasets::MnistObj>> &train = mnist.train_;
  vector<shared_ptr<datasets::MnistObj>> &test = mnist.test_;
  printf("train: %lu\n", train.size());
  printf("test: %lu\n", test.size());

  shared_ptr<Model> net(new CnnNet(28, 28, 10));
  float learning_rate = 0.001;

  int epoch_num = 0;
  float cost = 0.0;
  clock_t begin_time = clock();
  for (int step = 0; step < 1000000; ++step)
  {
    int train_idx = step % train.size();
    for (int x = 0; x < 28; ++x)
    {
      for (int y = 0; y < 28; ++y)
      {
        int idx = x + y * 28;
        net->input_->data_[idx] = train[train_idx]->image[idx];
      }
    }
    int idx_target = train[train_idx]->label;

    net->Forward();

    cost += SoftmaxLoss(net, idx_target);

    net->Backward();

    LearnRmsprop(net, learning_rate);

    bool new_line = false;
    int output_each = 1000;
    int time_each = 1000;
    int validate_each = 10000;

    int epoch_num_curr = step / train.size();
    if (epoch_num != epoch_num_curr)
    {
      epoch_num = epoch_num_curr;
      learning_rate /= 3;
      printf("learning rate: %.6f\n", learning_rate);
    }

    if (step % output_each == 0 && step != 0)
    {
      printf("%.3f epoch| cost: %.3f", 1.0f * step / train.size(),
             cost / output_each);
      cost = 0.0;
      new_line = true;
    }
    if (step % validate_each == 0 && step != 0)
    {
      float acc = 0;
      for (int test_idx = 0; test_idx < test.size(); ++test_idx)
      {
        for (int i = 0; i < 784; ++i)
        {
          net->input_->data_[i] = test[test_idx]->image[i];
        }
        int idx_gt = test[test_idx]->label;

        net->Forward();

        int idx_pred = MaxIdx(net->output_->data_);
        if (idx_gt == idx_pred)
        {
          acc += 1;
        }
      }
      acc /= test.size();

      printf("| test acc: %.3f", acc);
      new_line = true;
    }
    if (step % time_each == 0 && step != 0)
    {
      float time_curr = float(clock() - begin_time) / CLOCKS_PER_SEC;
      printf("| time: %.3f s", time_curr);
      begin_time = clock();
      new_line = true;
    }
    if (new_line)
    {
      printf("\n");
    }
  }

  return 0;
}
