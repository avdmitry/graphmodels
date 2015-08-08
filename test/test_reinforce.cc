#include "utils.h"
#include "agent.h"
#include "circle_world.h"

using namespace std;

int main(int argc, char *argv[])
{
  (void)argc;
  (void)argv;

  // srand(time(NULL));
  srand(0);

  //math = shared_ptr<Math>(new MathCuda(0));
  math = shared_ptr<Math>(new MathCpu);
  math->Init();

  shared_ptr<CircleWorld> env(new CircleWorld);
  shared_ptr<DQNAgent> agent(
      new DQNAgent(env->GetNumStates(), env->GetMaxNumActions()));

  vector<float> expected = {-1.430, -1.017, -0.646, -1.044};
  float reward = 0;
  static const int kCompareAfter = 1000;
  int i = 0;
  for (int step = 0; step < kCompareAfter * expected.size(); ++step)
  {
    shared_ptr<MatWdw> state = env->GetState();
    int action = agent->Act(state);
    int r = env->SampleNextState(action);
    if (step > 0)
    {
      agent->Learn(r);
    }

    reward += r;
    if (step % kCompareAfter == 0 && step != 0)
    {
      float curr_reward = reward / kCompareAfter;
      if (curr_reward != expected[i])
      {
        printf("test failed on %u step: got: %f, expect: %f\n", i, curr_reward,
               expected[i]);
        exit(-1);
      }
      reward = 0;
      i += 1;
    }
  }

  math->Deinit();

  printf("test passed\n");

  return 0;
}
