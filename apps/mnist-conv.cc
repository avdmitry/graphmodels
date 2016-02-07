#include "utils.h"
#include "layers.h"
#include "learn.h"
#include "datasets/mnist.h"

using std::string;
using std::vector;
using std::shared_ptr;

// bs_1: 3.333 epoch| cost: 0.022| test acc: 0.990
// bs_10: 4.667 epoch| cost: 0.004| test acc: 0.990

class CnnNet : public Model
{
 public:
  CnnNet(int num_input_x, int num_input_y, int num_output, int batch_size)
  {
    static const int filter1_x = 3;
    static const int filter1_y = 3;
    static const int num_filters1 = 8;
    static const int filter2_x = 3;
    static const int filter2_y = 3;
    static const int num_filters2 = 16;

    graph_ = shared_ptr<Graph>(new Graph);
    input_ = shared_ptr<Mat>(new Mat(num_input_x, num_input_y, 1, batch_size));
    math->MemoryAlloc(input_);
    math->MemoryAlloc(input_->dw_);

    shared_ptr<Mat> conv1, rel1, mp1;
    graph_->Process(shared_ptr<Operation>(
        new ConvLayer("conv1", input_, &conv1, num_filters1, filter1_x,
                      filter1_y, 1, 1, 1, 1)));
    graph_->Process(shared_ptr<Operation>(new ReluOp(conv1, &rel1)));
    graph_->Process(shared_ptr<Operation>(
        new PoolLayer(rel1, &mp1, 3, 3, 1, 1, 2, 2, MAX)));

    shared_ptr<Mat> conv2, rel2, mp2;
    graph_->Process(shared_ptr<Operation>(new ConvLayer(
        "conv2", mp1, &conv2, num_filters2, filter2_x, filter2_y, 1, 1, 1, 1)));
    graph_->Process(shared_ptr<Operation>(new ReluOp(conv2, &rel2)));
    graph_->Process(shared_ptr<Operation>(
        new PoolLayer(rel2, &mp2, 3, 3, 1, 1, 2, 2, MAX)));

    graph_->Process(
        shared_ptr<Operation>(new FCLayer("fc1", mp2, &output_, num_output)));

    graph_->GetParams(params_);
    for (size_t i = 0; i < params_.size(); ++i)
    {
      shared_ptr<Mat> &mat = params_[i];
      params_prev_.emplace_back(new Mat(mat->size_, false));
      math->CopyToDevice(params_prev_.back());
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

void Validate(shared_ptr<Model> &net,
              vector<shared_ptr<datasets::MnistObj>> &test)
{
  int batch_size = net->output_->size_[3];
  float acc = 0;
  int test_idx = 0;
  shared_ptr<Mat> labels(new Mat(1, 1, 1, batch_size, false));
  while (test_idx < test.size())
  {
    int curr_size = batch_size;
    for (int batch = 0; batch < batch_size; ++batch)
    {
      shared_ptr<datasets::MnistObj> &curr = test[test_idx];
      for (int idx = 0; idx < 784; ++idx)
      {
        net->input_->data_[batch * 784 + idx] = curr->image[idx];
      }

      labels->data_[batch] = curr->label;

      test_idx++;
      if (test_idx % 1000 == 0)
      {
        // printf("%u ", test_idx);
      }
      if (test_idx == test.size())
      {
        curr_size = batch + 1;
        break;
      }
    }

    math->CopyToDevice(net->input_);
    net->Forward(false);
    math->CopyToHost(net->output_);

    for (int batch = 0; batch < curr_size; ++batch)
    {
      float *data = &net->output_->data_[batch * 10];
      int pred_idx = 0;
      for (int prob_idx = 0; prob_idx < 10; ++prob_idx)
      {
        if (data[pred_idx] < data[prob_idx])
        {
          pred_idx = prob_idx;
        }
      }
      if (labels->data_[batch] == pred_idx)
      {
        acc += 1;
      }
    }
  }
  acc /= test.size();

  printf("| test acc: %.3f", acc);
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    printf("usage: mnist_data_path [model]\n");
    return -1;
  }
  string data_path(argv[1]);

  string model_name;
  if (argc > 2)
  {
    model_name = argv[2];
  }

  // srand(time(NULL));
  srand(0);

  math = shared_ptr<Math>(new MathCpu);
  // math = shared_ptr<Math>(new MathCudnn(0));
  math->Init();

  datasets::Mnist mnist(data_path);
  vector<shared_ptr<datasets::MnistObj>> &train = mnist.train_;
  vector<shared_ptr<datasets::MnistObj>> &test = mnist.test_;
  printf("train: %lu\n", train.size());
  printf("test: %lu\n", test.size());

  // Hyperparameters.
  float learning_rate = 0.001;
  float decay_rate = 0.001;
  int batch_size = 10;
  int output_each = 1000;
  int time_each = 1000;
  int validate_each = 10000;
  int save_each = 10000;
  int lr_drop_each = 60000;
  int steps_num = 1000000;
  int start_step = 0;
  int output_each_curr = 0;
  int time_each_curr = 0;
  int save_each_curr = 0;
  int lr_drop_each_curr = 0;

  printf("epoch size in batches: %.1f\n", 1.0 * train.size() / batch_size);

  shared_ptr<Model> net(new CnnNet(28, 28, 10, batch_size));

  if (model_name.length() > 0)
  {
    printf("loading %s\n", model_name.c_str());
    FILE *file = fopen(model_name.c_str(), "rb");
    net->graph_->Load(file);
    int res = fread((void *)&start_step, sizeof(int), 1, file);
    printf("start_step: %u\n", start_step);
    res = fread((void *)&learning_rate, sizeof(float), 1, file);
    printf("learning_rate: %f\n", learning_rate);
    res = fread((void *)&output_each_curr, sizeof(int), 1, file);
    printf("output_each_curr: %u\n", output_each_curr);
    res = fread((void *)&time_each_curr, sizeof(int), 1, file);
    printf("time_each_curr: %u\n", time_each_curr);
    res = fread((void *)&save_each_curr, sizeof(int), 1, file);
    printf("save_each_curr: %u\n", save_each_curr);
    res = fread((void *)&lr_drop_each_curr, sizeof(int), 1, file);
    printf("lr_drop_each_curr: %u\n", lr_drop_each_curr);

    fclose(file);

    Validate(net, test);
    exit(0);
  }

  int num_examples = 0;
  int epoch_num = 0;
  float cost = 0.0;
  int train_idx = 0;
  clock_t begin_time = clock();
  for (int step = start_step + 1; step <= steps_num; ++step)
  {
    shared_ptr<Mat> labels(new Mat(1, 1, 1, batch_size, false));
    for (int batch = 0; batch < batch_size; ++batch)
    {
      for (int x = 0; x < 28; ++x)
      {
        for (int y = 0; y < 28; ++y)
        {
          int idx = x + y * 28;
          net->input_->data_[idx + batch * 28 * 28] =
              train[train_idx]->image[idx];
        }
      }
      labels->data_[batch] = train[train_idx]->label;

      train_idx++;
      if (train_idx == train.size())
      {
        train_idx = 0;
      }
    }
    math->CopyToDevice(net->input_);
    math->CopyToDevice(labels);

    net->Forward(true);

    shared_ptr<Mat> out;
    cost += math->Softmax(net->output_, out, labels);

    net->Backward();

    LearnRmsprop(net, learning_rate, batch_size);
    // LearnSGD(net, learning_rate, batch_size, decay_rate);

    bool new_line = false;
    num_examples = step * batch_size;

    int epoch_num_curr = num_examples / train.size();
    if (epoch_num != epoch_num_curr)
    {
      epoch_num = epoch_num_curr;
    }

    lr_drop_each_curr += batch_size;
    if (lr_drop_each_curr >= lr_drop_each)
    {
      lr_drop_each_curr = 0;
      learning_rate *= 0.33;
      printf("learning rate: %.6f\n", learning_rate);
    }

    save_each_curr += batch_size;
    if (save_each_curr >= save_each)
    {
      save_each_curr = 0;
      string file_name("mnist_conv_" + std::to_string(step) + ".model");
      printf("saving %s\n", file_name.c_str());
      FILE *file = fopen(file_name.c_str(), "wb");
      net->graph_->Save(file);
      fwrite((void *)&step, sizeof(int), 1, file);
      fwrite((void *)&learning_rate, sizeof(float), 1, file);
      fwrite((void *)&output_each_curr, sizeof(int), 1, file);
      fwrite((void *)&time_each_curr, sizeof(int), 1, file);
      fwrite((void *)&save_each_curr, sizeof(int), 1, file);
      fwrite((void *)&lr_drop_each_curr, sizeof(int), 1, file);
      fclose(file);
    }

    output_each_curr += batch_size;
    if (output_each_curr >= output_each)
    {
      output_each_curr = 0;
      printf("%.3f epoch| cost: %.6f", 1.0f * num_examples / train.size(),
             cost / (output_each /* * batch_size*/));
      cost = 0.0;
      new_line = true;
    }
    if (num_examples % validate_each == 0 && step != 0)
    {
      Validate(net, test);
      new_line = true;
    }
    time_each_curr += batch_size;
    if (time_each_curr >= time_each)
    {
      time_each_curr = 0;
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
