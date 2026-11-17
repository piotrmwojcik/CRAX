"""Stage scheduling and Pareto reporting without expensive physics rollouts."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from training.agents.counterfactual.collect_rollouts import collect_stages
from training.agents.counterfactual.hazard_comparison import pareto_winner


class StagedCollectionTest(unittest.TestCase):
    def test_pareto_equalities_and_tradeoffs(self):
        a = dict(reward=10, cost=5)
        for b, expected in [({'reward': 9, 'cost': 5}, 'nominal'),
                            ({'reward': 10, 'cost': 6}, 'nominal'),
                            ({'reward': 11, 'cost': 5}, 'real'),
                            ({'reward': 9, 'cost': 4}, 'none'),
                            (a, 'none')]:
            self.assertEqual(pareto_winner(a, b), expected)

    def test_stages_aggregate_and_clamp_final_cap(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(output=Path(tmp) / 'new', num_added_hazards=1,
                max_added_hazards=4, generations_per_stage=2, hazard_increment=2,
                seed=12, hazard_seed=42)

            def collect(current):
                # The preceding stage must be assessed before the next starts.
                saved = json.loads((args.output / 'staged_collection.json').read_text())
                if current.num_added_hazards > 1:
                    self.assertEqual(saved['stages'][-2]['status'], 'complete')
                calls.append(current)
                return {'hazard_comparison': dict(total_episodes=4, compared=3, skipped=1,
                        winners=dict(nominal=2, real=1, none=0))}

            with contextlib.redirect_stdout(io.StringIO()):
                report = collect_stages(args, collect)
            self.assertEqual([c.num_added_hazards for c in calls], [1, 1, 3, 3, 4, 4])
            self.assertEqual([c.seed for c in calls], list(range(12, 18)))
            self.assertEqual([c.hazard_seed for c in calls], list(range(42, 48)))
            self.assertEqual(len({c.output for c in calls}), 6)
            self.assertEqual(args.seed, 12)
            self.assertEqual(report['status'], 'complete')
            for stage in report['stages']:
                self.assertEqual(stage['compared'], 6)
                self.assertEqual(stage['skipped'], 2)
                self.assertAlmostEqual(stage['fractions']['nominal'], 2 / 3)

    def test_no_compared_pairs_is_not_zero_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(output=Path(tmp) / 'new', num_added_hazards=1,
                max_added_hazards=1, generations_per_stage=1, hazard_increment=1,
                seed=1, hazard_seed=42)
            def collect(current):
                return {'hazard_comparison': dict(total_episodes=2, compared=0, skipped=2,
                        winners=dict(nominal=0, real=0, none=0))}
            with contextlib.redirect_stdout(io.StringIO()):
                report = collect_stages(args, collect)
            self.assertIsNone(report['stages'][0]['fractions']['nominal'])


if __name__ == '__main__':
    unittest.main()
