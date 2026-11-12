# Learning path: from tabular RL to training a humanoid

For someone who knows Q-learning and SARSA but not deep RL or JAX. Roughly one
week at one to two days a week.

1. **Policy gradients.** [Spinning Up, Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html),
   parts 1 to 3. The key shift from SARSA: instead of learning action values and
   acting greedily, we learn the policy directly, which works for continuous
   actions like joint targets.
2. **PPO.** [Spinning Up, PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
   This is the algorithm in `scripts/train.py`. Make sure you can explain the
   clipped objective, the value function and GAE.
3. **PPO in code.** Read [CleanRL's ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)
   line by line, then run it on a Gymnasium MuJoCo task such as HalfCheetah.
   Change one hyperparameter and see what happens to the reward curve.
4. **JAX.** The [JAX quickstart](https://docs.jax.dev/en/latest/quickstart.html):
   `jit`, `grad` and `vmap` are most of what Playground relies on. `vmap` is how
   thousands of robots are simulated in parallel on one GPU.
5. **Playground.** Run `configs/smoke_test.yaml`, then read
   `mujoco_playground/_src/locomotion/t1/joystick.py` in the installed package
   (`uv run python -c "import mujoco_playground, os; print(os.path.dirname(mujoco_playground.__file__))"`).
   For each reward term in `_get_reward`, write one sentence on what behaviour it
   encourages or punishes.

Papers worth reading once the above makes sense:

- [MuJoCo Playground](https://arxiv.org/abs/2502.08844), the framework we use.
- [Learning to walk in minutes using massively parallel deep RL](https://arxiv.org/abs/2109.11978),
  the training recipe behind Playground's locomotion tasks.
- [Learning agile soccer skills for a bipedal robot with deep RL](https://arxiv.org/abs/2304.13653),
  humanoid soccer from simulation to real robots.
