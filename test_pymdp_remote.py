"""Test pymdp legacy backend on AWS."""
from pymdp.legacy.agent import Agent
from pymdp.legacy import utils
import numpy as np

num_obs = [5]
num_states = [5]
num_controls = [3]

A = utils.random_A_matrix(num_obs, num_states)
B = utils.random_B_matrix(num_states, num_controls)
C = utils.obj_array_zeros(num_obs)
D = utils.obj_array_uniform(num_states)

agent = Agent(A=A, B=B, C=C, D=D, policy_len=2, control_fac_idx=[0])
print("Agent created")

obs = [0]
qs = agent.infer_states(obs)
print("infer_states OK")

q_pi, efe = agent.infer_policies()
print(f"infer_policies OK, num_policies: {len(q_pi)}")

action = agent.sample_action()
print(f"action: {action}")
print("LEGACY PYMDP FULL LOOP SUCCESS")
