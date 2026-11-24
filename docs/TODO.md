# Need to decide what to add to the literature review


## Interesting Reads

**Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*, CoRL 2021.** [arXiv:2109.11978](https://arxiv.org/abs/2109.11978)

Thousands of robots on one GPU, PPO adapted to very large batches, and a game-inspired terrain curriculum. Its analysis of batch size and other PPO settings in this regime applies to tuning the T1 baseline, which trains the same way.

**Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain*, Science Robotics 2020.** [arXiv:2010.11251](https://arxiv.org/abs/2010.11251)

Teacher–student training: a teacher with privileged information (terrain, contacts) guides a student that uses only onboard sensors. The UT Austin striker paper below uses the same recipe for kicking.

**Radosavovic et al., *Real-World Humanoid Locomotion with Reinforcement Learning*, Science Robotics 2024.** [arXiv:2303.03381](https://arxiv.org/abs/2303.03381)

A causal transformer over the history of observations and actions, trained with RL in randomized simulation and deployed zero-shot on the Digit humanoid. The history lets the policy adapt to conditions it cannot observe directly.

**Gu et al., *Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer*, ICRA Workshop on Agile Robotics 2024.** [arXiv:2404.05695](https://arxiv.org/abs/2404.05695)

An open humanoid framework that trains in Isaac Gym, then verifies policies in MuJoCo (sim-to-sim). The same check applies to our MJX-trained policies.


## Walking and natural motion

**Peng et al., *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control*, ACM TOG 2021.** [arXiv:2104.02180](https://arxiv.org/abs/2104.02180)

A discriminator trained on motion clips rewards the policy for moving like the data, replacing hand-tuned style rewards. Booster Lab and the RoboCup-winning T1 soccer controller both use it, which makes it our candidate method for a natural K1 gait.

**Chen et al., *Booster Lab: A Data-Centric Pipeline for Learning Deployable Humanoid Locomotion Policies*, arXiv 2026.** [arXiv:2606.27813](https://arxiv.org/abs/2606.27813)

Booster's newest pipeline: curated motion data, a simulator calibrated to the real robot, AMP training in Isaac Lab, and cross-simulator checks. The only paper here with K1 results (preliminary). Moving it from T1 to K1 meant replacing the morphology model, joint limits, actuator parameters, retargeting config and deployment interface, the same list our port faces.

## Getting up

**Huang et al., *Learning Humanoid Standing-up Control across Diverse Postures*, RSS 2025.** [arXiv:2502.08378](https://arxiv.org/abs/2502.08378)

HoST learns to stand up from scratch, from many fallen postures, with several critics and a terrain curriculum, then runs on a real Unitree G1. Smoothness regularization and a speed bound stop the motion from becoming oscillatory or violent.

**He et al., *Learning Getting-Up Policies for Real-World Humanoid Robots*, RSS 2025.** [arXiv:2502.12152](https://arxiv.org/abs/2502.12152)

HumanUP works in two stages: first find any motion that gets up, with loose limits on smoothness, speed and torque; then refine it to be slow, smooth and robust to start pose and terrain. Unlike walking, getting up needs accurate collision geometry.

## Kicking and soccer

**Haarnoja et al., *Learning Agile Soccer Skills for a Bipedal Robot with Deep Reinforcement Learning*, Science Robotics 2024.** [arXiv:2304.13653](https://arxiv.org/abs/2304.13653)

The closest precedent for the whole project. DeepMind trained a get-up skill and a scoring skill separately, then distilled them into one 1v1 soccer policy using self-play, on small Robotis OP3 humanoids simulated in MuJoCo. The same train-skills-then-combine structure fits our walk, get-up and kick goals.

**Wang et al., *Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots*, Science Robotics 2026.** [arXiv:2511.03996](https://arxiv.org/abs/2511.03996)

One RL controller on the Booster T1 that searches for, chases and kicks the ball using onboard vision alone. It uses AMP for natural motion, and in training a simulated perception system adds noise and missed detections. The Tsinghua Hephaestus team used it to win the RoboCup 2025 Adult-Size league.

**Xu et al., *Learning Agile Striker Skills for Humanoid Soccer Robots from Noisy Sensory Input*, ICRA 2026.** [arXiv:2512.06571](https://arxiv.org/abs/2512.06571)

UT Austin's four-stage teacher–student pipeline for kicking on a Booster T1: ball chasing, directional kicking, distillation into a student with noisy sensing, then refinement with constrained RL. It reuses Booster Gym's observations, so it fits next to our T1 setup. Its ablations show that the constrained RL, the noise modeling and the adaptation stage are all needed.