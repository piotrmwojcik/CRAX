"""Tests for simulator-evaluated projected action gradient descent."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from scripts.counterfactual_actions import search_action


class ActionSearchTest(unittest.TestCase):
    def test_gradient_through_dynamic_simulator_loop(self):
        def evaluate(action):
            _, cost = jax.lax.while_loop(
                lambda carry: carry[0] < 3,
                lambda carry: (carry[0] + 1, carry[1] + action[0]),
                (jnp.array(0), jnp.array(1.0)),
            )
            return cost, jnp.array(10.0)
        _, history, status = search_action(
            evaluate, np.zeros(1), 1., 10., 0.01, 0.2,
            gradient_steps=1, learning_rate=0.01, cost_penalty=2.)
        self.assertEqual(status, 'completed')
        # Float32 simulator outputs introduce finite-difference roundoff.
        np.testing.assert_allclose(history[1]['action'], [-0.06], atol=2e-6)

    def test_known_minimal_action_change(self):
        best, history, status = search_action(
            lambda a: (1 + a[0], jnp.array(10.0)), np.zeros(2), 1., 10., 0.01, 0.2,
            gradient_steps=100, learning_rate=0.01, cost_penalty=2.)
        self.assertIsNotNone(best)
        self.assertLessEqual(best['cost'], 0.8)
        self.assertAlmostEqual(best['action_change_l2'], 0.2, delta=0.02)
        self.assertLess(abs(best['action'][1]), 1e-6)
        self.assertEqual(len(history), 101)
        self.assertEqual(status, 'completed')
        # Exact first gradient: cost hinge contributes +2 along coordinate 0.
        np.testing.assert_allclose(history[1]['action'], [-0.02, 0], atol=1e-7)

    def test_reward_constraint_prevents_cheaper_action(self):
        best, _, _ = search_action(
            lambda a: (1 + a[0], 10 + a[0]), np.zeros(2), 1., 10., 0.01, 0.2,
            gradient_steps=20)
        self.assertIsNone(best)

    def test_action_bounds_are_respected(self):
        best, history, _ = search_action(
            lambda a: (2 + a[0], jnp.array(10.0)), np.array([0.9]), 2.9, 10., 0.01, 3.,
            gradient_steps=5, learning_rate=100.)
        self.assertIsNone(best)
        self.assertTrue(all(np.all(np.abs(row['action']) <= 1) for row in history))

    def test_nonfinite_gradient_stops_explicitly(self):
        best, history, status = search_action(
            lambda a: (1 + jnp.sqrt(a[0]), jnp.array(10.0)), np.zeros(1), 1., 10., 0.01, 0.2,
            gradient_steps=5)
        self.assertIsNone(best)
        self.assertEqual(len(history), 1)
        self.assertEqual(status, 'stopped_nonfinite_gradient_or_loss')


if __name__ == '__main__':
    unittest.main()
