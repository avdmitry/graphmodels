#include "utils.h"
#include "agent.h"
#include "puck_world.h"

using std::string;
using std::vector;
using std::shared_ptr;

int main(int argc, char *argv[])
{
  (void)argc;
  (void)argv;

  // srand(time(NULL));
  srand(0);

  // math = shared_ptr<Math>(new MathCuda(0));
  // math = shared_ptr<Math>(new MathBlas);
  math = shared_ptr<Math>(new MathCpu);
  math->Init();

  shared_ptr<PuckWorld> env(new PuckWorld);
  shared_ptr<DQNAgent> agent(
      new DQNAgent(env->GetNumStates(), env->GetMaxNumActions()));

  vector<string> expected = {"-1.430", "-1.017", "-0.646", "-1.044"};
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
      char curr_reward_str[10];
      sprintf(curr_reward_str, "%.3f", curr_reward);
      if (expected[i] != curr_reward_str)
      {
        printf("test failed on %u step: got: %s, expect: %s\n", i,
               curr_reward_str, expected[i].c_str());
        exit(-1);
      }
      reward = 0;
      i += 1;
    }
  }

  printf("test passed\n");

  return 0;
}
