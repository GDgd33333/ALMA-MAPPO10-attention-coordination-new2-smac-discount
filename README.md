# IAGA-MAPPO-QMIX

This repository implements **IAGA-MAPPO-QMIX**, a hierarchical multi-agent reinforcement learning framework for large-scale cooperative task allocation and execution.

The method is built on the ALMA/REFIL/PyMARL codebase and extends the original hierarchical task-allocation framework with:

- a high-level MAPPO task allocator;
- an Intention-Aware Graph Attention module, abbreviated as **IAGA**;
- a low-level QMIX executor conditioned on assigned subtasks;
- additional allocation-quality and IAGA diagnostic metrics;
- extended SaveTheCity and SMAC multi-army scenarios.

## Method Overview

The framework decomposes cooperative multi-agent decision making into two levels.

At the high level, a parameter-shared MAPPO actor generates an agent-wise task allocation distribution. The actor first encodes agent entities and task entities, computes base agent-task matching logits, and then uses IAGA to refine the logits with graph-structured coordination information.

At the low level, a QMIX executor learns subtask-conditioned action policies. Once each agent receives an assigned subtask from the high-level allocator, the low-level agent network selects primitive actions under that subtask condition.

The overall execution flow is:

```text
Entities and masks
      |
      v
High-level agent/task encoding
      |
      v
Base allocation logits L_base
      |
      v
IAGA graph refinement
      |
      v
Refined allocation logits L_refined
      |
      v
Assigned subtasks for each agent
      |
      v
Low-level QMIX action selection
      |
      v
Environment step and reward
```

## IAGA Module

IAGA refines the high-level allocation logits by explicitly modeling three types of relationships:

1. **Agent-Agent graph**: captures potential allocation conflicts or cooperation among agents based on their soft task intentions.
2. **Task-Task graph**: captures task-side relationships, including task representation and task load information.
3. **Agent-Task graph**: produces edge-level allocation corrections for each agent-task pair.

The refinement is implemented as a bounded residual update:

```text
L_refined = L_base + alpha * tanh(Delta L)
```

where `alpha` is controlled by `hier_agent.iga_delta_scale`.

This design keeps the base MAPPO actor intact while injecting relation-aware coordination information into the final allocation distribution.

## CTDE and Parameter Sharing

This project follows a CTDE-style training paradigm:

- **Centralized training**:
  - the high-level MAPPO critic evaluates the team-level high-level allocation;
  - the low-level QMIX mixer uses global state information to train joint action-value estimates.
- **Agent-wise execution**:
  - the high-level actor outputs each agent's task distribution in parallel;
  - the low-level agent network selects each agent's primitive action under its assigned subtask.

The high-level actor is parameter-shared. Agent identity is injected by adding a learnable agent-ID embedding to the encoded agent representation:

```text
H_i = AgentEncoder(e_i) + ID_i
```

The low-level agent network is also parameter-shared, which follows the standard QMIX-style implementation. Parameter sharing does not mean centralized action enumeration: each agent still receives its own input and produces its own task/action decision.

## Important Training Details

Current high-level MAPPO settings include:

- GAE for high-level advantage estimation;
- advantage normalization for actor updates;
- PPO clipped objective;
- entropy regularization;
- gradient clipping;
- a centralized high-level critic.

Current value-normalization behavior:

- Low-level QMIX uses PopArt through the mixer when `popart: True`.
- High-level MAPPO currently uses advantage normalization, but its PPO critic directly fits raw high-level returns. It does not currently apply PopArt or ValueNorm to the high-level value target.

## Repository Structure

```text
.
├── src/
│   ├── main.py
│   ├── config/
│   │   ├── algs/
│   │   │   ├── qmix_atten.yaml
│   │   │   ├── qmix_atten_mappo_alloc.yaml
│   │   │   └── qmix_atten_mappo_alloc_smac.yaml
│   │   └── envs/
│   │       ├── ff.yaml
│   │       └── sc2multiarmy.yaml
│   ├── controllers/
│   ├── learners/
│   ├── modules/
│   │   ├── agents/
│   │   │   ├── allocation_policies.py
│   │   │   ├── allocation_critics.py
│   │   │   └── intention_graph_attention.py
│   │   └── mixers/
│   └── envs/
│       ├── firefighters/
│       └── starcraft2/
├── results/
├── 改动及方案说明.md
├── IA-GA指标添加评估与说明.md
└── TensorBoard指标说明与趋势分析.md
```

## Main Configurations

### SaveTheCity

Use:

```text
--env-config=ff
--config=qmix_atten_mappo_alloc
```

Recommended high-level setting:

```text
--agent.subtask_cond=mask
--hier_agent.task_allocation=mappo
--hier_agent.alloc_policy=mappo
--hier_agent.alloc_critic=ppo
--hier_agent.use_iga=True
--hier_agent.action_length=5
```

### SMAC Multi-Army

Use:

```text
--env-config=sc2multiarmy
--config=qmix_atten_mappo_alloc_smac
```

Recommended high-level setting:

```text
--agent.subtask_cond=mask
--hier_agent.task_allocation=mappo
--hier_agent.alloc_policy=mappo
--hier_agent.alloc_critic=ppo
--hier_agent.use_iga=True
--hier_agent.action_length=3
```

## Supported Scenarios

### SaveTheCity

The project supports the original SaveTheCity scenarios and the extended larger-scale scenarios, including:

```text
30-30
35-35
38-38
40-40
45-45
50-50
```

For the newly checked scenario registrations, the intended large-scale setting follows the original SaveTheCity convention where the number of buildings is `n_agents + 1`.

### SMAC Multi-Army

The following multi-army scenarios are registered in `src/envs/starcraft2/custom_scenarios.py`:

```text
6-8sz_maxsize4_maxarmies3_symmetric
6-8sz_maxsize4_maxarmies3_unitdisadvantage
6-8MMM_maxsize4_maxarmies3_symmetric
6-8MMM_maxsize4_maxarmies3_unitdisadvantage
8-10MMM_maxsize4_maxarmies4_unitdisadvantage
10-12MMM_maxsize4_maxarmies4_unitdisadvantage
16-18MMM_maxsize4_maxarmies5_unitdisadvantage
6-8csz_maxsize4_maxarmies3_unitdisadvantage
```

In these SMAC multi-army tasks, a high-level task corresponds to an enemy army or enemy squad. The environment builds `entity2task_mask` and `task_mask` so that the allocator can assign agents to enemy-army subtasks.

## Running Experiments

### SaveTheCity Example

```bash
cd /home/gud/ALMA-MAPPO10-attention-coordination/src

CUDA_VISIBLE_DEVICES=0 python main.py \
  --env-config=ff \
  --config=qmix_atten_mappo_alloc \
  --agent.subtask_cond=mask \
  --scenario=30-30 \
  --hier_agent.task_allocation=mappo \
  --hier_agent.alloc_policy=mappo \
  --hier_agent.alloc_critic=ppo \
  --hier_agent.use_iga=True \
  --hier_agent.action_length=5 \
  --epsilon_anneal_time=2000000 \
  --use_tensorboard=True \
  --save_model=True \
  --save_model_interval=1000000 \
  2>&1 | tee iga_save_30_30.log
```

### SMAC Multi-Army Example

Before running SMAC, set `SC2PATH` to a valid StarCraft II installation. On the current machine, the repaired installation path is:

```bash
export SC2PATH=/home/gud/sc2_fixed/StarCraftII
```

If websocket connection errors occur, clear proxy variables before launching SC2:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy WS_PROXY WSS_PROXY ws_proxy wss_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

Example command:

```bash
cd /home/gud/ALMA-MAPPO10-attention-coordination/src
export SC2PATH=/home/gud/sc2_fixed/StarCraftII
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy WS_PROXY WSS_PROXY ws_proxy wss_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

CUDA_VISIBLE_DEVICES=0 python main.py \
  --env-config=sc2multiarmy \
  --config=qmix_atten_mappo_alloc_smac \
  --agent.subtask_cond=mask \
  --scenario=6-8MMM_maxsize4_maxarmies3_unitdisadvantage \
  --batch_size_run=4 \
  --hier_agent.task_allocation=mappo \
  --hier_agent.alloc_policy=mappo \
  --hier_agent.alloc_critic=ppo \
  --hier_agent.use_agent_id=True \
  --hier_agent.use_last_task=False \
  --t_max=50000000 \
  --lr=0.00025 \
  --hier_agent.action_length=3 \
  --hier_agent.gamma_high=0.95 \
  --hier_agent.use_gae=True \
  --hier_agent.gae_lambda=0.95 \
  --hier_agent.ppo_clip=0.1 \
  --hier_agent.ppo_epochs=2 \
  --hier_agent.ppo_entropy_coef=0.005 \
  --hier_agent.ppo_update_interval=5 \
  --hier_agent.alloc_pi_lr=0.00025 \
  --hier_agent.alloc_q_lr=0.00025 \
  --hier_agent.use_iga=True \
  --epsilon_anneal_time=2000000 \
  --use_tensorboard=True \
  --save_model=True \
  --save_model_interval=1000000 \
  2>&1 | tee 6-8MMM_unitdisadvantage.log
```

Do not leave a blank space after a line-continuation backslash. For example, use:

```bash
--scenario=6-8MMM_maxsize4_maxarmies3_unitdisadvantage \
```

not:

```bash
--scenario=6-8MMM_maxsize4_maxarmies3_unitdisadvantage \ 
```

## TensorBoard

TensorBoard logs are saved under:

```text
results/tb_logs
```

Launch TensorBoard with:

```bash
tensorboard --logdir /home/gud/ALMA-MAPPO10-attention-coordination/results/tb_logs
```

Useful metric groups include:

- `test/*`: evaluation performance, including success/win-related metrics;
- `losses/*`: low-level and high-level losses;
- `alloc_metrics/*`: MAPPO allocation learning statistics;
- `alloc_iga/*`: IAGA refinement and attention statistics;
- `alloc_quality/*`: hard allocation quality, coverage, load, and coordination metrics.

For detailed metric explanations, see:

```text
TensorBoard指标说明与趋势分析.md
IA-GA指标添加评估与说明.md
```

## Useful Ablations

To disable IAGA and keep the high-level MAPPO allocation framework:

```bash
--hier_agent.use_iga=False
```

To disable a specific graph inside IAGA:

```bash
--hier_agent.iga_use_agent_graph=False
--hier_agent.iga_use_task_graph=False
--hier_agent.iga_use_agent_task_graph=False
```

To keep all IAGA graph modules while hiding the base policy's probabilistic
task intention from the refiner:

```bash
--hier_agent.iga_use_probabilistic_intention=False
```

This controlled ablation replaces each agent's learned task distribution with
a uniform distribution over its currently valid tasks and zeros the direct
base-logit feature supplied to the Agent-Task relation head. The real base
logits are still used in the final residual update, so the base actor and the
graph architecture remain intact.

`iga_use_agent_task_graph=False` is a controlled Agent-Task relation
ablation rather than another alias for `use_iga=False`. It keeps the
Agent-Agent and Task-Task graph encoders and the bounded residual update, but
replaces explicit per-(agent, task) edge features with an agent-wise
non-relational MLP. The MLP consumes the agent graph context, a pooled task
graph context, and the complete base-logit row, then predicts one residual
task vector for that agent.

To adjust IAGA correction strength:

```bash
--hier_agent.iga_delta_scale=0.05
--hier_agent.iga_delta_scale=0.1
--hier_agent.iga_delta_scale=0.2
```

To use top-k graph sparsification:

```bash
--hier_agent.iga_topk_agents=3
--hier_agent.iga_topk_tasks=3
```

The default setting uses full graphs:

```text
iga_topk_agents = 0
iga_topk_tasks = 0
```

## Notes on Theoretical Analysis

The project is suitable for the following theoretical or structural analyses:

1. **Allocation complexity**: centralized enumeration over joint task assignments scales as `O(M^N)`, while the proposed agent-wise allocation matrix has output size `O(NM)`. IAGA adds polynomial graph reasoning cost, approximately `O(N^2 d + M^2 d + NM d)` under full graphs.
2. **Bounded residual refinement**: because `L_refined = L_base + alpha * tanh(Delta L)`, each logit correction is bounded by `alpha`, which helps preserve the base actor while injecting coordination information.
3. **CTDE-compatible coordination**: IAGA uses relational information to coordinate allocation but still produces agent-wise task distributions rather than enumerating a centralized joint allocation.

For claims such as "IAGA helps early-stage learning", it is safer to present this as a mechanism analysis supported by diagnostic metrics, rather than as a strict convergence proof. A full convergence guarantee for deep MAPPO with neural graph refinement would require strong assumptions that are usually unrealistic in this setting.

## Lineage and Citation

This repository is based on ALMA:

```bibtex
@inproceedings{iqbal2022alma,
title={ALMA: Hierarchical Learning for Composite Multi-Agent Tasks},
author={Shariq Iqbal and Robby Costales and Fei Sha},
booktitle={Advances in Neural Information Processing Systems},
year={2022},
url={https://openreview.net/forum?id=JUXn1vXcrLA}
}
```

ALMA is built on the public code release for REFIL, which is built on PyMARL.

