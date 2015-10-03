#include "utils.h"
#include "agent.h"
#include "circle_world.h"

using std::string;
using std::vector;
using std::shared_ptr;

int main(int argc, char *argv[])
{
  (void)argc;
  (void)argv;

  // srand(time(NULL));
  srand(0);

  math = shared_ptr<Math>(new MathCpu);
  math->Init();

  shared_ptr<CircleWorld> env(new CircleWorld);
  shared_ptr<DQNAgent> agent(
      new DQNAgent(env->GetNumStates(), env->GetMaxNumActions()));

  float reward = 0;
  for (int step = 0; step < 1000000; ++step)
  {
    shared_ptr<MatWdw> state = env->GetState();
    int action = agent->Act(state);
    int r = env->SampleNextState(action);
    if (step > 0)
    {
      agent->Learn(r);
    }

    reward += r;
    if (step % 1000 == 0 && step != 0)
    {
      printf("reward: %.3f\n", reward / 1000);
      reward = 0;
    }
  }

  return 0;
}
