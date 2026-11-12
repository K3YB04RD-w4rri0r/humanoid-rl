"""Fast checks that the pinned stack loads and steps the T1 environment."""

import jax
import jax.numpy as jp
from mujoco_playground import registry


def test_t1_reset_and_step():
    env = registry.load("T1JoystickFlatTerrain")
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state = jax.jit(env.step)(state, jp.zeros(env.action_size))
    assert not bool(jp.any(jp.isnan(state.data.qpos)))
    assert "command" in state.info
