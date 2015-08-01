#ifndef AGENT_H
#define AGENT_H

#include "utils.h"
#include "layers.h"

void UpdateMat(std::shared_ptr<Mat> &mat, float alpha)
{
  // Updates in place.
  for (int i = 0, n = mat->n_ * mat->d_; i < n; i++)
  {
    if (mat->dw_[i] != 0)
    {
      mat->w_[i] += -alpha * mat->dw_[i];
      mat->dw_[i] = 0;
    }
  }
}

class Net
{
 public:
  Net(int ns, int na)
  {
    int num_hidden_units = 100;  // Numer of hidden units.
    w1_ = RandMat(num_hidden_units, ns, -0.01, 0.01);
    b1_ = std::shared_ptr<Mat>(new Mat(num_hidden_units, 1));
    w2_ = RandMat(na, num_hidden_units, -0.01, 0.01);
    b2_ = std::shared_ptr<Mat>(new Mat(na, 1));
  }

  std::shared_ptr<Mat> Forward(std::shared_ptr<Mat> &s, bool needs_backprop)
  {
    graph_ = std::shared_ptr<Graph>(new Graph(needs_backprop));

    std::shared_ptr<Mat> mul1;
    graph_->Process(std::shared_ptr<Object>(new MulOp(w1_, s, &mul1)));
    std::shared_ptr<Mat> a1mat;
    graph_->Process(std::shared_ptr<Object>(new AddOp(mul1, b1_, &a1mat)));
    std::shared_ptr<Mat> h1mat;
    graph_->Process(std::shared_ptr<Object>(new TanhOp(a1mat, &h1mat)));

    std::shared_ptr<Mat> mul2;
    graph_->Process(std::shared_ptr<Object>(new MulOp(w2_, h1mat, &mul2)));

    std::shared_ptr<Mat> a2mat;
    graph_->Process(std::shared_ptr<Object>(new AddOp(mul2, b2_, &a2mat)));

    graph_->Forward();

    return a2mat;
  }

  void Backward()
  {
    graph_->Backward();
  }

  void UpdateNet(float alpha)
  {
    UpdateMat(w1_, alpha);
    UpdateMat(b1_, alpha);
    UpdateMat(w2_, alpha);
    UpdateMat(b2_, alpha);
  }

  std::shared_ptr<Mat> w1_, w2_, b1_, b2_;
  std::shared_ptr<Graph> graph_;
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
    Reset(ns, na);
  }
  ~DQNAgent()
  {
  }

  void Reset(int ns, int na)
  {
    gamma_ = 0.9;
    epsilon_ = 0.2;
    alpha_ = 0.01;
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

    step_ = 0;
    r0_ = 0;
  }

  int Act(std::shared_ptr<Mat> &state)
  {
    // Convert to a Mat column vector.

    int a;
    // Epsilon greedy policy.
    if (Random01() < epsilon_)
    {
      a = Randi(0, na_ - 1);
    }
    else
    {
      // Greedy wrt Q function.
      std::shared_ptr<Mat> amat = net_->Forward(state, false);
      a = MaxIdx(amat->w_);
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
    std::shared_ptr<Mat> tmat = net_->Forward(observation->s1_, false);
    float qmax = observation->r0_ + gamma_ * tmat->w_[MaxIdx(tmat->w_)];

    // Predict.
    std::shared_ptr<Mat> pred = net_->Forward(observation->s0_, true);

    float tderror = pred->w_[observation->a0_] - qmax;
    // Huber loss to robustify.
    if (tderror > tderror_clamp_)
    {
      tderror = tderror_clamp_;
    }
    if (tderror < -tderror_clamp_)
    {
      tderror = -tderror_clamp_;
    }
    pred->dw_[observation->a0_] = tderror;

    net_->Backward();

    // Update net.
    net_->UpdateNet(alpha_);

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
    r0_ = r1;  // Store for next update.
  }

  std::shared_ptr<Mat> s0_, s1_;
  int a0_, a1_;
  float r0_;

  int na_;
  std::shared_ptr<Net> net_;

  // Number of time steps before we add another experience to replay memory.
  int remember_each_;
  int refreshing_steps_per_new_;
  std::vector<std::shared_ptr<Observation>> history_;
  int history_idx_, history_length_;
  int step_;

  float gamma_;          // future reward discount factor, [0, 1)
  float epsilon_;        // for epsilon-greedy policy, [0, 1)
  float alpha_;          // value function learning rate
  float tderror_clamp_;  // for robustness
};

#endif
