#include "utils.h"
#include "layers.h"
#include "datasets/mnist.h"

using std::string;
using std::vector;
using std::shared_ptr;

class FcNet : public Model
{
 public:
  FcNet(int num_input, int num_output, int batch_size)
  {
    static const int num_hidden_units = 256;

    graph_ = shared_ptr<Graph>(new Graph);
    input_ = shared_ptr<Mat>(new Mat(num_input, 1, 1, batch_size));
    math->MemoryAlloc(input_);
    math->MemoryAlloc(input_->dw_);

    shared_ptr<Mat> fc1, rel1;
    graph_->Process(shared_ptr<Operation>(
        new FCLayer("fc1", input_, &fc1, num_hidden_units)));
    graph_->Process(shared_ptr<Operation>(new ReluOp(fc1, &rel1)));

    shared_ptr<Mat> fc2, rel2;
    graph_->Process(shared_ptr<Operation>(
        new FCLayer("fc2", rel1, &fc2, num_hidden_units)));
    graph_->Process(shared_ptr<Operation>(new ReluOp(fc2, &rel2)));

    graph_->Process(
        shared_ptr<Operation>(new FCLayer("fc3", rel2, &output_, num_output)));

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

  printf("test acc: %.3f\n", acc);
}

void ValidateSnn(shared_ptr<Model> &net,
                 vector<shared_ptr<datasets::MnistObj>> &test)
{
  vector<shared_ptr<Mat>> params;
  net->graph_->GetParamsForward(params);
  vector<shared_ptr<Mat>> potentials;
  for (int layer = 0; layer < params.size() / 2; ++layer)
  {
    shared_ptr<Mat> &filters = params[2 * layer];
    shared_ptr<Mat> &biases = params[2 * layer + 1];
    math->CopyToHost(filters);
    math->CopyToHost(biases);
    potentials.emplace_back(new Mat(1, filters->size_[1], 1, 1, false));
  }

  float acc = 0;

  printf("Processing:");
  for (int test_idx = 0; test_idx < test.size(); ++test_idx)
  {
    if (test_idx % 1000 == 0)
    {
      printf(" %u%%", (int)(100.0 * test_idx / test.size()));
      fflush(stdout);
    }

    for (int layer = 0; layer < params.size() / 2; ++layer)
    {
      shared_ptr<Mat> &potential = potentials[layer];
      for (int idx = 0; idx < potential->data_.size(); ++idx)
      {
        potential->data_[idx] = 0;
      }
    }

    static const int kDuration = 15;
    static const float kThreshold = 1.0f;
    static const int kNumClasses = 10;
    static const int kImageSize = 784;

    vector<int> output_spikes(kNumClasses);
    for (int duration = 0; duration < kDuration; ++duration)
    {
      vector<int> input_spikes(kImageSize);
      for (int idx = 0; idx < kImageSize; ++idx)
      {
        // Test should be in [0..1].
        if (Random01() <= test[test_idx]->image[idx])
        {
          input_spikes[idx] = 1;
        }
        else
        {
          input_spikes[idx] = 0;
        }
      }

      for (int layer = 0; layer < params.size() / 2; ++layer)
      {
        shared_ptr<Mat> &potential = potentials[layer];
        shared_ptr<Mat> &filters = params[2 * layer];
        shared_ptr<Mat> &biases = params[2 * layer + 1];
        // fc: in out 1 1
        int num_in = filters->size_[0];
        int num_out = filters->size_[1];
        vector<int> curr_spikes(num_out);
        for (int i = 0; i < num_out; ++i)
        {
          float result = 0.0;  // biases->data_[i];
          for (int j = 0; j < num_in; ++j)
          {
            int filters_idx = num_out * j + i;
            // int filters_idx = num_in * i + j;
            result += input_spikes[j] * filters->data_[filters_idx];
          }
          potential->data_[i] += result;
          if (potential->data_[i] >= kThreshold)
          {
            curr_spikes[i] = 1;
            potential->data_[i] = 0;
          }
          else
          {
            curr_spikes[i] = 0;
          }
        }
        input_spikes = curr_spikes;
      }

      for (int i = 0; i < kNumClasses; ++i)
      {
        output_spikes[i] += input_spikes[i];
      }
    }

    float max_value = output_spikes[0];
    int max_idx = 0;
    for (int i = 1; i < output_spikes.size(); ++i)
    {
      float value = output_spikes[i];
      if (value > max_value)
      {
        max_idx = i;
        max_value = value;
      }
    }

    int idx_pred = max_idx;
    int idx_gt = test[test_idx]->label;
    if (idx_gt == idx_pred)
    {
      acc += 1;
    }
  }
  printf("\n");
  acc /= test.size();

  printf("snn acc: %.3f\n", acc);
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    printf("usage: mnist_data_path model\n");
    return -1;
  }
  string data_path(argv[1]);
  string model_name(argv[2]);

  // srand(time(NULL));
  srand(0);

  // math = shared_ptr<Math>(new MathCpu);
  math = shared_ptr<Math>(new MathCudnn(0));
  math->Init();

  datasets::Mnist mnist(data_path);
  vector<shared_ptr<datasets::MnistObj>> &test = mnist.test_;
  printf("test: %lu\n", test.size());

  // Hyperparameters.
  float learning_rate = 0.0005;
  int batch_size = 10;
  int start_step = 0;
  int output_each_curr = 0;
  int time_each_curr = 0;
  int save_each_curr = 0;
  int lr_drop_each_curr = 0;

  shared_ptr<Model> net(new FcNet(784, 10, batch_size));

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
    ValidateSnn(net, test);
  }

  return 0;
}
