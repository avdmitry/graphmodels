#include "puck_world.h"

using std::string;
using std::vector;
using std::shared_ptr;

PuckWorld::PuckWorld()
{
  ppx_ = Random01();  // x,y
  ppy_ = Random01();
  pvx_ = Random01() * 0.05 - 0.025;  // velocity
  pvy_ = Random01() * 0.05 - 0.025;
  tx_ = Random01();  // target
  ty_ = Random01();
  tx2_ = Random01();  // target
  ty2_ = Random01();

  self_rad_ = 0.05;
  bad_rad_ = 0.25;
  step_ = 0;
}

shared_ptr<Mat> PuckWorld::GetState()
{
  shared_ptr<Mat> state(new Mat(GetNumStates(), 1));
  state->data_[0] = ppx_ - 0.5;
  state->data_[1] = ppy_ - 0.5;
  state->data_[2] = pvx_ * 10;
  state->data_[3] = pvy_ * 10;
  state->data_[4] = tx_ - ppx_;
  state->data_[5] = ty_ - ppy_;
  state->data_[6] = tx2_ - ppx_;
  state->data_[7] = ty2_ - ppy_;
  return state;
}

int PuckWorld::SampleNextState(int a)
{
  // World dynamics.
  ppx_ += pvx_;  // newton
  ppy_ += pvy_;
  pvx_ *= 0.95;  // damping
  pvy_ *= 0.95;

  // Agent action influences velocity.
  float accel = 0.002;
  switch (a)
  {
    case 0:
      pvx_ -= accel;
      break;
    case 1:
      pvx_ += accel;
      break;
    case 2:
      pvy_ -= accel;
      break;
    case 3:
      pvy_ += accel;
      break;
  }

  // Handle boundary conditions and bounce.
  if (ppx_ < self_rad_)
  {
    pvx_ *= -0.5;
    ppx_ = self_rad_;
  }
  if (ppx_ > 1 - self_rad_)
  {
    pvx_ *= -0.5;
    ppx_ = 1 - self_rad_;
  }
  if (ppy_ < self_rad_)
  {
    pvy_ *= -0.5;
    ppy_ = self_rad_;
  }
  if (ppy_ > 1 - self_rad_)
  {
    pvy_ *= -0.5;
    ppy_ = 1 - self_rad_;
  }

  step_ += 1;
  if (step_ % 100 == 0)
  {
    tx_ = Random01();  // Reset the target location.
    ty_ = Random01();
  }

  // Compute distances.
  float dx = ppx_ - tx_;
  float dy = ppy_ - ty_;
  float d1 = sqrt(dx * dx + dy * dy);

  dx = ppx_ - tx2_;
  dy = ppy_ - ty2_;
  float d2 = sqrt(dx * dx + dy * dy);

  float dxnorm = dx / d2;
  float dynorm = dy / d2;
  float speed = 0.001;
  tx2_ += speed * dxnorm;
  ty2_ += speed * dynorm;

  // Compute reward.
  float r = -d1;  // Want to go close to green.
  if (d2 < bad_rad_)
  {
    // But if we're too close to red that's bad.
    r += 2 * (d2 - bad_rad_) / bad_rad_;
  }

  // if (a == 4) { r += 0.05; } // Give bonus for gliding with no force.

  return r;
}
