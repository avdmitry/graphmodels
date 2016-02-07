#ifndef AGENT_H
#define AGENT_H

#include "utils.h"
#include "layers.h"
#include "learn.h"

class Net : public Model
{
 public:
  Net(int ns, int na)
  {
    int num_hidden_units = 100;
    w1_ = RandMat(num_hidden_units, ns, -0.01, 0.01);
    b1_ = std::shared_ptr<Mat>(new Mat(num_hidden_units, 1));
    w2_ = RandMat(na, num_hidden_units, -0.01, 0.01);
    b2_ = std::shared_ptr<Mat>(new Mat(na, 1));

    input_ = std::shared_ptr<Mat>(new Mat(ns, 1));

    GetParameters(params_);
    for (size_t i = 0; i < params_.size(); ++i)
    {
      std::shared_ptr<Mat> &mat = params_[i];
      params_prev_.emplace_back(new Mat(mat->size_));
    }
  }

  void Create(int idx)
  {
    (void)idx;

    graph_ = std::shared_ptr<Graph>(new Graph);

    std::shared_ptr<Mat> mul1, a1mat, h1mat, mul2;
    graph_->Process(std::shared_ptr<Operation>(new MulOp(w1_, input_, &mul1)));
    graph_->Process(std::shared_ptr<Operation>(new AddOp(mul1, b1_, &a1mat)));
    graph_->Process(std::shared_ptr<Operation>(new TanhOp(a1mat, &h1mat)));

    graph_->Process(std::shared_ptr<Operation>(new MulOp(w2_, h1mat, &mul2)));
    graph_->Process(std::shared_ptr<Operation>(new AddOp(mul2, b2_, &output_)));
  }

  void ClearPrevState()
  {
  }

  void GetParameters(std::vector<std::shared_ptr<Mat>> &params)
  {
    params.emplace_back(w1_);
    params.emplace_back(b1_);
    params.emplace_back(w2_);
    params.emplace_back(b2_);
  }

  std::shared_ptr<Mat> w1_, w2_, b1_, b2_;
};

class Observation
{
 public:
  Observation(std::shared_ptr<Mat> &s0, std::shared_ptr<Mat> &s1, int a0,
              float r0)
      : s0_(s0), s1_(s1), a0_(a0), r0_(r0)
  {
  }
  ~Observation()
  {
  }

  std::shared_ptr<Mat> s0_, s1_;
  int a0_;
  float r0_;
};

class Agent
{
 public:
  Agent()
  {
  }
  virtual ~Agent()
  {
  }
};

class DQNAgent : public Agent
{
 public:
  DQNAgent(int ns, int na)
  {
    gamma_ = 0.9;
    epsilon_ = 0.2;
    remember_each_ = 10;
    refreshing_steps_per_new_ = 20;
    tderror_clamp_ = 1.0;

    // Init history, experience replay memory.
    history_.resize(5000);
    // Where to insert.
    history_idx_ = 0;
    // Current length.
    history_length_ = 0;

    // Left, right, up, down, nothing.
    na_ = na;

    // ns: x,y,vx,vy, puck dx,dy.
    net_ = std::shared_ptr<Net>(new Net(ns, na));
    net_->Create(0);

    step_ = 0;
    r0_ = 0;
  }
  ~DQNAgent()
  {
  }

  int Act(std::shared_ptr<Mat> &state)
  {
    int a;
    // Epsilon greedy policy.
    if (Random01() < epsilon_)
    {
      a = Randi(0, na_ - 1);
    }
    else
    {
      // Greedy wrt Q function.
      *net_->input_ = *state;
      net_->Forward(false);
      a = MaxIdx(net_->output_);
    }

    // Shift state memory.
    s0_ = s1_;
    a0_ = a1_;
    s1_ = state;
    a1_ = a;

    return a;
  }

  float LearnFromObservation(std::shared_ptr<Observation> &observation)
  {
    // Want: Q(s,a) = r + gamma * max_a' Q(s',a').

    // Compute the target Q value.
    *net_->input_ = *observation->s1_;
    net_->Forward(false);
    std::shared_ptr<Mat> &out = net_->output_;
    float qmax = observation->r0_ + gamma_ * out->data_[MaxIdx(out)];

    // Predict.
    *net_->input_ = *observation->s0_;
    net_->Forward(true);

    std::shared_ptr<Mat> &pred = net_->output_;
    float tderror = pred->data_[observation->a0_] - qmax;

    // Huber loss to robustify.
    if (tderror > tderror_clamp_)
    {
      tderror = tderror_clamp_;
    }
    if (tderror < -tderror_clamp_)
    {
      tderror = -tderror_clamp_;
    }

    pred->dw_->data_[observation->a0_] = tderror;

    net_->Backward();

    LearnSGD(net_, 0.01, 1, 0.0);

    return tderror;
  }

  // Perform an update on Q function.
  void Learn(float r1)
  {
    // Learn from observation to get a sense of how "surprising" it is.
    std::shared_ptr<Observation> observation(
        new Observation(s0_, s1_, a0_, r0_));
    LearnFromObservation(observation);

    // Decide if we should keep this experience in the replay.
    if (step_ % remember_each_ == 0)
    {
      history_[history_idx_] = observation;
      history_idx_ += 1;
      if (history_length_ < history_.size())
      {
        history_length_ += 1;
      }
      if (history_idx_ >= history_.size())
      {
        history_idx_ = 0;
      }
    }
    step_ += 1;

    // Sample some additional experience from replay memory and learn from it.
    if (history_length_ > 10 * refreshing_steps_per_new_)
    {
      for (int k = 0; k < refreshing_steps_per_new_; k++)
      {
        int ri = Randi(0, history_length_ - 1);  // Priority sweeps?
        LearnFromObservation(history_[ri]);
      }
    }

    // Store for next update.
    r0_ = r1;
  }

  std::shared_ptr<Mat> s0_, s1_;
  int a0_, a1_;
  float r0_;

  int na_;
  std::shared_ptr<Model> net_;

  // Number of time steps before we add another experience to replay memory.
  int remember_each_;
  int refreshing_steps_per_new_;
  std::vector<std::shared_ptr<Observation>> history_;
  int history_idx_, history_length_;
  int step_;

  float gamma_;          // future reward discount factor, [0, 1)
  float epsilon_;        // for epsilon-greedy policy, [0, 1)
  float tderror_clamp_;  // for robustness
};

#endif
