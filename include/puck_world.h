#ifndef PUCK_WORLD_H
#define PUCK_WORLD_H

#include "utils.h"

class PuckWorld
{
 public:
  PuckWorld();

  int GetNumStates()
  {
    return 8;  // x,y,vx,vy, dx,dy
  }

  int GetMaxNumActions()
  {
    return 5;  // left, right, up, down, nothing
  }

  std::shared_ptr<Mat> GetState();

  int SampleNextState(int a);

  int step_;
  float self_rad_, bad_rad_;
  float ppx_, ppy_, pvx_, pvy_, tx_, ty_, tx2_, ty2_;
};

#endif
