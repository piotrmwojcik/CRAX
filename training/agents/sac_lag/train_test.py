"""Tests for SAC-Lag training helpers."""

from absl.testing import absltest
import jax.numpy as jnp

from training.agents.sac_lag.train import update_observed_episode_length


class ObservedEpisodeLengthTest(absltest.TestCase):
    def test_ema_updates_from_completed_episodes(self):
        observed = update_observed_episode_length(
            previous_length=jnp.asarray(2000.0),
            completed_length_sum=jnp.asarray(850.0),
            completed_episode_count=jnp.asarray(1.0),
            ema=0.9,
        )
        self.assertAlmostEqual(float(observed), 1885.0)

    def test_ema_uses_mean_of_all_newly_completed_episodes(self):
        observed = update_observed_episode_length(
            previous_length=jnp.asarray(1000.0),
            completed_length_sum=jnp.asarray(1500.0),
            completed_episode_count=jnp.asarray(2.0),
            ema=0.5,
        )
        self.assertAlmostEqual(float(observed), 875.0)

    def test_ema_is_unchanged_when_no_episode_finished(self):
        observed = update_observed_episode_length(
            previous_length=jnp.asarray(900.0),
            completed_length_sum=jnp.asarray(0.0),
            completed_episode_count=jnp.asarray(0.0),
            ema=0.9,
        )
        self.assertAlmostEqual(float(observed), 900.0)


if __name__ == '__main__':
    absltest.main()
