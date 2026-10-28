"""Geometry checks for replayable Safe Pathway cost measurements."""

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jp
import numpy as np

from crax import base
from crax.envs.safe_pathway import SafePathway


class PathwayCostTest(unittest.TestCase):
    def measurements(self, feet, rectangular=False):
        env = SimpleNamespace(
            _foot_link_ids=list(range(len(feet))),
            _hazard_height=0.01,
            _hazard_radii=jp.array([1.0]),
            _hazard_half_extents=jp.array([[1.0, 1.0]]),
            _hazard_is_rect=jp.array([rectangular]),
        )
        state = SimpleNamespace(x=base.Transform.create(pos=jp.array(feet)))
        return SafePathway._calculate_hazard_penetration(
            env, state, jp.zeros((1, 3))
        )

    def test_edge_halfway_and_center(self):
        for rectangular in (False, True):
            for distance, expected in ((1.0, 0.0), (0.5, 0.5), (0.0, 1.0)):
                penetration = self.measurements([[distance, 0.0, 0.0]], rectangular)
                np.testing.assert_allclose(penetration, [expected], atol=1.1e-4)
                np.testing.assert_allclose(penetration ** 2, [expected ** 2], atol=2.1e-4)

    def test_airborne_and_outside_feet_do_not_contribute(self):
        for rectangular in (False, True):
            np.testing.assert_array_equal(
                self.measurements([[0.0, 0.0, 0.5], [2.0, 0.0, 0.0]], rectangular),
                [0.0],
            )

    def test_deepest_foot_is_used_instead_of_summing_feet(self):
        for rectangular in (False, True):
            penetration = self.measurements([[0.25, 0.0, 0.0], [0.5, 0.0, 0.0]], rectangular)
            np.testing.assert_allclose(penetration, [0.75], atol=1e-6)

    def test_no_hazards(self):
        result = jax.jit(lambda: SafePathway._calculate_hazard_penetration(
            None, None, jp.zeros((0, 3))
        ))()
        self.assertEqual(result.shape, (0,))
        self.assertEqual(float(jp.sum(result ** 2)), 0.0)


if __name__ == '__main__':
    unittest.main()
