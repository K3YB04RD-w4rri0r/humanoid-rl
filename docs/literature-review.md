# Literature review

References checked against arXiv and publisher pages on 24 November 2026.

## RL foundations

**Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv 2017.** [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

PPO. Its clipped surrogate objective allows several epochs of minibatch updates on each batch of data.

**Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016.** [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)

The advantage estimator inside PPO; λ trades bias against variance. Its experiments include a simulated biped learning to stand up from the ground.

**Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*, ICML 2018.** [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)

An off-policy alternative to PPO that reuses past samples. Brax, which Playground trains with, also implements SAC, so a PPO vs SAC comparison needs no new code.

**Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning*, RSS 2018.** [arXiv:1710.06542](https://arxiv.org/abs/1710.06542)

**Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017.** [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)

Randomize the simulator enough and the real world looks like one more variation. The paper randomizes rendering for an object detector; locomotion work randomizes physics instead, and Playground's T1 task has its own randomization code (`randomize.py`).

**Henderson et al., *Deep Reinforcement Learning that Matters*, AAAI 2018.** [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)

Shows how much deep RL results vary from run to run, which makes comparisons hard to trust, and proposes reporting guidelines. Basis for how we report the reproduced baseline (goal 2): several seeds, with the spread.


## Simulators and frameworks

- [MuJoCo Playground v0.2.0](https://github.com/google-deepmind/mujoco_playground/tree/v0.2.0)
- [Brax](https://github.com/google/brax)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [mjlab](https://github.com/mujocolab/mjlab)
- [Booster Gym](https://github.com/BoosterRobotics/booster_gym)


