from .agent import Agent
from .allocation_critics import *
from .allocation_policies import *
from functools import partial

ALLOC_CRITIC_REGISTRY = {}
ALLOC_CRITIC_REGISTRY['standard'] = StandardAllocCritic
ALLOC_CRITIC_REGISTRY['ppo'] = PPOAllocCritic
ALLOC_CRITIC_REGISTRY['mappo'] = PPOAllocCritic

ALLOC_POLICY_REGISTRY = {}
ALLOC_POLICY_REGISTRY['autoreg'] = AutoregressiveAllocPolicy
ALLOC_POLICY_REGISTRY['autoreg_ppo'] = AutoregressivePPOAllocPolicy
ALLOC_POLICY_REGISTRY['matching_ppo'] = MatchingPPOAllocPolicy
ALLOC_POLICY_REGISTRY['mappo'] = MAPPOAllocPolicy
