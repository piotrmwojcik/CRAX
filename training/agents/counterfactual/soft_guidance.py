"""Numerical building blocks for Jing et al. (AAAI 2020), Eqs. (3)--(9).

We use squared Gaussian-kernel MMD on normalized finite-horizon occupancies.
The trust-region solution below solves Eq. (5) in span(H^-1 g, H^-1 b),
which is equivalent to solving its two-multiplier dual (reward ascent signs).
"""
import jax
import jax.numpy as jnp
import numpy as np


def gaussian_kernel(x, y, bandwidth):
    distance = jnp.maximum(jnp.sum(x*x, axis=1)[:, None]
                           + jnp.sum(y*y, axis=1)[None, :] - 2*x@y.T, 0.)
    return jnp.exp(-distance / (2 * bandwidth**2))


def weighted_mmd(weights, agent_kernel, cross_kernel, expert_kernel):
    weights = weights / jnp.sum(weights)
    return weights @ agent_kernel @ weights - 2*jnp.dot(weights, cross_kernel.mean(1)) + expert_kernel.mean()


def conjugate_gradient(matvec, rhs, iterations=10, tolerance=1e-10):
    x = jnp.zeros_like(rhs)
    residual, direction = rhs, rhs
    squared = jnp.vdot(residual, residual)
    for _ in range(iterations):
        if float(squared) < tolerance:
            break
        product = matvec(direction)
        curvature = jnp.vdot(direction, product)
        if not np.isfinite(float(curvature)) or float(curvature) <= 0:
            break
        alpha = squared / curvature
        x += alpha * direction
        residual -= alpha * product
        next_squared = jnp.vdot(residual, residual)
        direction = residual + next_squared / squared * direction
        squared = next_squared
    return x


def constrained_step(g, b, inverse_g, inverse_b, slack, delta):
    """Maximize g.x subject to b.x <= slack and x.H.x/2 <= delta.

    If infeasible, return the natural-gradient discrepancy recovery step.
    """
    q = max(float(jnp.vdot(g, inverse_g)), 0.)
    s = max(float(jnp.vdot(b, inverse_b)), 0.)
    r = float(jnp.vdot(b, inverse_g))
    zero = jnp.zeros_like(g)
    if s < 1e-12:
        if slack < 0:
            return zero, 'infeasible_zero_gradient'
        return (np.sqrt(2*delta/q)*inverse_g if q > 1e-12 else zero), 'reward'
    radius = np.sqrt(2*delta*s)
    if slack < -radius:
        return -np.sqrt(2*delta/s)*inverse_b, 'recovery'
    reward_step = np.sqrt(2*delta/q)*inverse_g if q > 1e-12 else zero
    if float(jnp.vdot(b, reward_step)) <= slack:
        return reward_step, 'reward'
    tangent = inverse_g - (r/s)*inverse_b
    tangent_norm = max(q-r*r/s, 0.)
    remaining = max(2*delta-slack*slack/s, 0.)
    step = (slack/s)*inverse_b
    if tangent_norm > 1e-12:
        step += np.sqrt(remaining/tangent_norm)*tangent
    return step, 'constrained'


def discounted_returns(rewards, valid, gamma):
    def body(carry, item):
        reward, mask = item
        carry = (reward + gamma*carry)*mask
        return carry, carry
    return jax.lax.scan(body, jnp.zeros(rewards.shape[1]), (rewards, valid), reverse=True)[1]
