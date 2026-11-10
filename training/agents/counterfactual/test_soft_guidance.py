"""Numerical checks for the constrained update and on-policy data path."""
import unittest
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize
from training.agents.counterfactual.soft_guidance import (
    constrained_step, conjugate_gradient, gaussian_kernel, weighted_mmd, discounted_returns)
from training.agents.counterfactual.finetune_soft_guidance import make_rollout, make_update, evaluate_policy
from training.agents.counterfactual.test_collect_rollouts import ToyEnv
from training.agents.ppo.networks import make_ppo_networks
from crax.envs.wrappers import training as wrappers


class SoftGuidanceTest(unittest.TestCase):
    def test_evaluation_is_repeatable_and_uses_episode_totals(self):
        net=make_ppo_networks(2,1,policy_hidden_layer_sizes=(8,),value_hidden_layer_sizes=(8,))
        actor=net.policy_network.init(jax.random.PRNGKey(1))
        rollout=make_rollout(wrappers.wrap(ToyEnv(),episode_length=5),net,None,5)
        first=evaluate_policy(rollout,actor,10000,4)
        second=evaluate_policy(rollout,actor,10000,4)
        first.pop('eval/seconds'); second.pop('eval/seconds')
        self.assertEqual(first,second)
        self.assertEqual(first['eval/episode_length'],3.)
        self.assertEqual(first['eval/episodes'],4)
        self.assertGreater(first['eval/cost'],0.)

    def test_constrained_solution_matches_numerical_optimizer(self):
        h = np.array([[2., .2],[.2,1.]])
        g,b = np.array([1.,2.]),np.array([1.,1.])
        for slack in (.8,.1,-.1):
            step,mode = constrained_step(jnp.array(g),jnp.array(b),
                jnp.array(np.linalg.solve(h,g)),jnp.array(np.linalg.solve(h,b)),slack,.2)
            oracle = minimize(lambda x:-g@x,np.zeros(2),constraints=[
                {'type':'ineq','fun':lambda x:slack-b@x},
                {'type':'ineq','fun':lambda x:.2-x@h@x/2}],method='SLSQP',tol=1e-10)
            self.assertTrue(oracle.success)
            self.assertLessEqual(float(b@step),slack+1e-6)
            self.assertLessEqual(float(step@h@step/2),.2+1e-6)
            self.assertAlmostEqual(float(g@step),float(g@oracle.x),places=5)
        step,mode = constrained_step(jnp.array(g),jnp.array(b),
            jnp.array(np.linalg.solve(h,g)),jnp.array(np.linalg.solve(h,b)),-10.,.2)
        self.assertEqual(mode,'recovery')
        self.assertLess(float(b@step),0)
        self.assertAlmostEqual(float(step@h@step/2),.2,places=5)

    def test_mmd_and_cg(self):
        x=jnp.array([[0.],[1.],[2.]])
        k=gaussian_kernel(x,x,1.)
        self.assertAlmostEqual(float(weighted_mmd(jnp.ones(3),k,k,k)),0.,places=6)
        h=jnp.array([[3.,1.],[1.,2.]])
        rhs=jnp.array([1.,3.])
        np.testing.assert_allclose(conjugate_gradient(lambda v:h@v,rhs),np.linalg.solve(h,rhs),atol=1e-6)

    def test_rollout_mask_returns_and_policy_derivatives(self):
        net=make_ppo_networks(2,1,policy_hidden_layer_sizes=(8,),value_hidden_layer_sizes=(8,))
        actor=net.policy_network.init(jax.random.PRNGKey(1))
        env=wrappers.wrap(ToyEnv(),episode_length=5)
        data=make_rollout(env,net,None,5)(actor,jax.random.PRNGKey(2),jax.random.split(jax.random.PRNGKey(3),2))
        np.testing.assert_array_equal(data['valid'],[[1,1],[1,1],[1,1],[0,0],[0,0]])
        returns=discounted_returns(data['reward'],data['valid'],.9)
        np.testing.assert_allclose(returns[0],data['reward'][0]+.9*data['reward'][1]+.81*data['reward'][2],atol=1e-6)
        theta,unravel=ravel_pytree(actor)
        measures,grad,bgrad,hvp=make_update(net,None,unravel)
        data=dict(data,advantage=returns,occupancy=data['valid']/6.,kl_obs=data['obs'][0])
        features=jnp.concatenate([data['obs'],data['action']],-1).reshape(-1,3)[:6]
        k=gaussian_kernel(features,features,1.)
        cross=gaussian_kernel(features,features+1.,1.)
        args=(theta,data,jnp.arange(6),(k,cross,k))
        self.assertAlmostEqual(float(measures(theta,*args)[2]),0.,places=6)
        g,b=grad(theta,*args),bgrad(theta,*args)
        self.assertTrue(np.isfinite(g).all() and np.isfinite(b).all())
        self.assertGreater(float(jnp.linalg.norm(g)),0.)
        self.assertGreater(float(jnp.linalg.norm(b)),0.)
        product=hvp(theta,g,*args,.01)
        self.assertGreater(float(jnp.vdot(g,product)),0.)


if __name__ == '__main__':
    unittest.main()
