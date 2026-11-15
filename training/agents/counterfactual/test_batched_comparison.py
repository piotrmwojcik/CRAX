"""Checks for branch vectorization and placement-cache compatibility."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from collections import namedtuple
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from crax.envs.base import State
from training.agents.counterfactual.hazard_comparison import make_branch_runner
from training.agents.counterfactual.edit_lidar import cached_placements

Physics = namedtuple('Physics','q qd mocap_pos')
Physics.replace = Physics._replace

class BranchEnv:
    _hazard_half_extents=jnp.ones((2,2))
    _hazard_is_rect=jnp.zeros(2,dtype=bool)
    _hazard_radii=jnp.ones(2)
    _hazard_mocap_ids=[0,1]
    _cost_scaler=1.
    def pipeline_init(self,q,qd):
        return Physics(q,qd,jnp.zeros((2,3)))
    def _get_obs(self,ps,positions):
        return ps.q
    def reset(self,key):
        return State(pipeline_state=self.pipeline_init(jnp.zeros(1),jnp.zeros(1)),obs=jnp.zeros(1),
            reward=jnp.array(0.),done=jnp.array(0.),metrics={},
            info={'hazard_positions':jnp.zeros((2,3)),'hazard_penetration':jnp.zeros(2),'cost':jnp.array(0.)})
    def step(self,state,action):
        q=state.pipeline_state.q+1
        cost=self._hazard_half_extents[-1,0]*q[0]
        return state.replace(pipeline_state=state.pipeline_state._replace(q=q),obs=q,reward=action[0],
            done=(q[0]>=3).astype(float),info=dict(state.info,cost=cost,hazard_penetration=jnp.array([0.,cost])))

class BatchTests(unittest.TestCase):
    def test_vmap_matches_serial_with_distinct_widths_and_termination(self):
        distribution=SimpleNamespace(create_dist=lambda logits:SimpleNamespace(loc=logits,scale=jnp.ones_like(logits)),postprocess=lambda x:x)
        network=SimpleNamespace(policy_network=SimpleNamespace(apply=lambda n,p,x:x*.1),parametric_action_distribution=distribution)
        _,run=make_branch_runner(BranchEnv(),network,(None,None),4,1,1.)
        inputs=(jnp.array([[0.],[1.],[2.]]),jnp.zeros((3,1)),jnp.zeros((3,2,3)),
                jnp.array([[.1],[.2],[.3]]),jnp.arange(12,dtype=float).reshape(3,4,1)*.01)
        batch=jax.jit(jax.vmap(run))(*inputs)
        for i in range(3):
            serial=run(*(x[i] for x in inputs))
            for key in serial:
                np.testing.assert_allclose(batch[key][i],serial[key],rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(batch['valid'].sum(1),[3,2,1])

    def test_cache_reuses_compared_and_skipped_and_rejects_incomplete(self):
        manifest=dict(configuration=dict(num_added_hazards=3,hazard_seed=42,num_envs=2),total_completed_episodes=2)
        rows=[dict(env_index=0,episode_index=0,status='compared',positions=[[1,0,.01]],half_lengths=[.2]),
              dict(env_index=1,episode_index=0,status='skipped',reason='no safe gap')]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); log=path/'hazard_comparisons.jsonl'
            log.write_text('\n'.join(map(json.dumps,rows)))
            placements,records=cached_placements(path,manifest,3,42)
            np.testing.assert_array_equal(placements[0,0],[[1,0,.01]])
            self.assertEqual(records[1]['status'],'unchanged')
            self.assertIsNone(cached_placements(path,manifest,2,42))
            self.assertIsNone(cached_placements(path,manifest,3,43))
            log.write_text(json.dumps(rows[0]))
            self.assertIsNone(cached_placements(path,manifest,3,42))
            log.write_text('\n'.join(map(json.dumps,[rows[0],rows[0]])))
            self.assertIsNone(cached_placements(path,manifest,3,42))

if __name__=='__main__': unittest.main()
