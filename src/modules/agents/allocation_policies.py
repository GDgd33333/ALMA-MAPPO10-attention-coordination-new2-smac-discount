import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import RelaxedOneHotCategorical
from torch.distributions import Categorical
from ..layers import EntityAttentionLayer
from .allocation_common import groupmask2attnmask, TaskEmbedder, COUNT_NORM_FACTOR
from .intention_graph_attention import IntentionGraphAttentionRefinement

class AutoregressiveAllocPolicy(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args

        self.args = args
        self.pi_ag_attn = self.args.hier_agent['pi_ag_attn']
        self.pi_pointer_net = self.args.hier_agent['pi_pointer_net']
        self.subtask_mask = self.args.hier_agent['subtask_mask']
        self.sel_task_upd = self.args.hier_agent['sel_task_upd']
        self.pi_autoreg = self.args.hier_agent['pi_autoreg']

        embd_upd_in_shape = args.attn_embed_dim * 2
        if not self.pi_pointer_net:
            input_shape += args.n_tasks
            embd_upd_in_shape += args.n_tasks
        if not self.sel_task_upd:
            embd_upd_in_shape += args.attn_embed_dim
        self.fc1 = nn.Linear(input_shape, args.attn_embed_dim)
        self.task_embed = TaskEmbedder(args.attn_embed_dim, args)
        self.attn = EntityAttentionLayer(args.attn_embed_dim,
                                         args.attn_embed_dim,
                                         args.attn_embed_dim, args)
        if self.pi_autoreg:
            self.embed_update = nn.Linear(embd_upd_in_shape, args.attn_embed_dim)
            if self.pi_pointer_net:
                self.count_embed = nn.Linear(2, args.attn_embed_dim)
        elif self.pi_pointer_net:
            # can only embed nonagent entity counts since agent allocs are decided all at once
            self.count_embed = nn.Linear(1, args.attn_embed_dim)
        if not self.pi_pointer_net:
            self.out_fc = nn.Linear(args.attn_embed_dim * 2, args.n_tasks)
        self.register_buffer('sample_temp',
                             th.scalar_tensor(1))  # TODO: anneal or try other values?
        self.register_buffer('scale_factor',
                             th.scalar_tensor(args.attn_embed_dim).sqrt())

    @property
    def device(self):
        return self.fc1.weight.device

    def _autoreg_forward(self, task_embeds, task_nonag_counts, agent_embeds, task_mask, entity_mask, avail_actions,
                         calc_stats=False, test_mode=False, repeat_fn=lambda x: x):
        nt = self.args.n_tasks
        bs, na, _ = agent_embeds.shape
        stats = {}
        task_embeds = repeat_fn(task_embeds)
        task_mask = repeat_fn(task_mask)
        if task_nonag_counts is not None:
            task_nonag_counts = repeat_fn(task_nonag_counts)
        prop_bs = task_mask.shape[0]
        allocs = repeat_fn(th.zeros((bs, na, nt), device=agent_embeds.device))
        all_log_pi = th.zeros_like(allocs)

        task_ag_counts = th.zeros_like(allocs[:, 0])

        agent_mask = entity_mask[:, :na]

        prev_alloc_mask = th.zeros_like(task_mask)

        for ai in range(self.args.n_agents):
            # compute pointer-net logits (scale as in dot-product attention)
            curr_agent_embed = agent_embeds[:, [ai]]
            curr_agent_embed = repeat_fn(curr_agent_embed)

            if self.pi_pointer_net:
                count_embeds = self.count_embed(th.stack([task_nonag_counts, task_ag_counts], dim=-1))
                logits = th.bmm(curr_agent_embed, (task_embeds + count_embeds).transpose(1, 2)).squeeze(1) / self.scale_factor
            else:
                # curr_agent_embed.shape = (bs, 1, hd), task_embeds.shape = (bs, hd)
                logit_ins = th.cat([curr_agent_embed.squeeze(1), task_embeds], dim=1)
                logits = self.out_fc(F.relu(logit_ins))

            # mask inactive tasks s.t. softmax is 0
            curr_mask = task_mask.clone()
            masked_logits = logits.masked_fill(curr_mask.bool(), th.finfo(logits.dtype).min)
            # mask for inactive agents
            curr_agent_mask = repeat_fn(1 - agent_mask[:, [ai]].float())
            dist = RelaxedOneHotCategorical(self.sample_temp, logits=masked_logits)
            # sample action
            soft_ac = dist.rsample()
            if calc_stats:
                # NOTE: we can use dist.logits as log prob as pytorch
                # Categorical distribution normalizes logits such that
                # th.exp(dist.logits) is the probability
                all_log_pi[:, ai] = dist.logits
                ag_log_pi = dist.logits.gather(1, soft_ac.argmax(dim=1, keepdim=True))
                stats['log_pi'] = stats.get('log_pi', 0) + ag_log_pi * curr_agent_mask
                stats['best_prob'] = stats.get('best_prob', 0) + dist.probs.max(dim=1, keepdim=True)[0] * curr_agent_mask
                entropy = -(dist.logits * dist.probs).sum(dim=1, keepdim=True)
                stats['entropy'] = stats.get('entropy', 0) + entropy * curr_agent_mask
            # make one-hot sample that acts like a continuous sample in the backward pass
            onehot_ac = F.one_hot(soft_ac.argmax(dim=1), num_classes=nt).float()
            hard_ac = onehot_ac - soft_ac.detach() + soft_ac
            hard_ac = hard_ac * curr_agent_mask
            prev_alloc_mask += hard_ac.detach().to(th.uint8)
            task_ag_counts += hard_ac.detach() * COUNT_NORM_FACTOR
            allocs[:, ai] = hard_ac
            # update embedding of selected task to incorporate new agent (only if agent is active)
            if self.pi_pointer_net:
                if self.sel_task_upd:
                    # only update selected task embeddings
                    embed_upd_in = th.cat([task_embeds, curr_agent_embed.repeat(1, nt, 1)], dim=2)
                    task_embeds = task_embeds + self.embed_update(F.relu(embed_upd_in)) * hard_ac.detach().unsqueeze(2) * curr_agent_mask.unsqueeze(2)
                else:
                    # update all task embeddings (use copy of selected task to condition on the previous agents' allocations)
                    sel_task_embed = (task_embeds * hard_ac.detach().unsqueeze(2)).sum(dim=1, keepdim=True)
                    embed_upd_in = th.cat([task_embeds, curr_agent_embed.repeat(1, nt, 1), sel_task_embed.repeat(1, nt, 1)], dim=2)
                    task_embeds = task_embeds + self.embed_update(F.relu(embed_upd_in)) * curr_agent_mask.unsqueeze(2)
            else:
                embed_upd_in = th.cat([task_embeds, curr_agent_embed.squeeze(1), hard_ac.detach()], dim=1)
                task_embeds = task_embeds + self.embed_update(F.relu(embed_upd_in)) * curr_agent_mask
        if calc_stats:
            stats['all_log_pi'] = all_log_pi
        return allocs, stats

    def _standard_forward(self, task_embeds, task_nonag_counts, agent_embeds, task_mask, entity_mask,
                         calc_stats=False, test_mode=False, repeat_fn=lambda x: x):
        nt = self.args.n_tasks
        bs, na, _ = agent_embeds.shape
        stats = {}
        task_embeds = repeat_fn(task_embeds) + repeat_fn(self.count_embed(task_nonag_counts.unsqueeze(-1)))
        task_mask = repeat_fn(task_mask)
        agent_embeds = repeat_fn(agent_embeds)
        allocs = repeat_fn(th.zeros((bs, na, nt), device=agent_embeds.device))

        if self.pi_pointer_net:
            logits = th.bmm(agent_embeds, task_embeds.transpose(1, 2)) / self.scale_factor
        else:
            raise NotImplementedError
            # curr_agent_embed.shape = (bs, na, hd), task_embeds.shape(bs, nt, hd)
            logit_ins = th.cat([agent_embeds.unsqueeze(2).repeat(1, 1, nt, 1),
                                task_embeds.unsqueeze(1).repeat(1, na, 1, 1)], dim=3)
            logits = self.out_fc(F.relu(logit_ins)).squeeze(3)

        # mask inactive tasks s.t. softmax is 0
        masked_logits = logits.masked_fill(task_mask.unsqueeze(1).bool(), th.finfo(logits.dtype).min)
        # mask for inactive agents
        agent_mask = repeat_fn(1 - entity_mask[:, :na].float())
        dist = RelaxedOneHotCategorical(self.sample_temp, logits=masked_logits)
        # sample action
        soft_ac = dist.rsample()
        if calc_stats:
            # NOTE: we can use dist.logits as log prob as pytorch
            # Categorical distribution normalizes logits such that
            # th.exp(dist.logits) is the probability
            stats['all_log_pi'] = dist.logits
            ag_log_pi = dist.logits.gather(2, soft_ac.argmax(dim=2, keepdim=True)).squeeze(2)
            stats['log_pi'] = (ag_log_pi * agent_mask).sum(dim=1, keepdim=True)
            stats['best_prob'] = (dist.probs.max(dim=2)[0] * agent_mask).sum(dim=1, keepdim=True)
            entropy = -(dist.logits * dist.probs).sum(dim=2)
            stats['entropy'] = (entropy * agent_mask).sum(dim=1, keepdim=True)
        # make one-hot sample that acts like a continuous sample in the backward pass
        onehot_ac = F.one_hot(soft_ac.argmax(dim=2), num_classes=nt).float()
        allocs = onehot_ac - soft_ac.detach() + soft_ac
        allocs = allocs * agent_mask.unsqueeze(2)
        return allocs, stats

    def forward(self, batch, calc_stats=False, test_mode=False, n_proposals=-1):
        # copy entity2task mask and zero out assignments
        entities = batch['entities']
        entity_mask = batch['entity_mask']
        entity2task_mask = batch['entity2task_mask']
        avail_actions = batch['avail_actions']

        nag = self.args.n_agents
        entity2task = 1 - entity2task_mask.float()
        last_alloc = batch['last_alloc']
        entity2task[:, :nag] = last_alloc

        # observe which task agents were assigned to in previous step + which
        # task non-agent entities belong to
        if not self.pi_pointer_net:
            entities = th.cat([entities, entity2task], dim=-1)
        x1 = self.fc1(entities)
        if self.pi_pointer_net:
            x1 += self.task_embed(entity2task)

        # compute attention for non-agent entities and get embedding for each task
        nonagent_x1 = x1[:, nag:]
        if self.pi_pointer_net and self.subtask_mask:
            nonagent_attn_mask = groupmask2attnmask(
                entity2task_mask[:, nag:])
        else:
            nonagent_attn_mask = groupmask2attnmask(
                entity_mask[:, nag:])
        nonagent_mask = entity_mask[:, nag:]
        nonagent_x2 = self.attn(F.relu(nonagent_x1), pre_mask=nonagent_attn_mask,
                                post_mask=nonagent_mask)
        if self.pi_pointer_net:
            nonagent_entity2task = entity2task[:, nag:]  # (bs, n_nonagent, nt)
            # sum up embeddings of non-agent entities belonging to each task
            task_x2 = th.bmm(nonagent_entity2task.transpose(1, 2), nonagent_x2)
            # count nonagent entities present in each task
            task_nonag_cnt = nonagent_entity2task.sum(dim=1) * COUNT_NORM_FACTOR
        else:
            task_x2 = nonagent_x2.mean(dim=1)
            task_nonag_cnt = None

        # get agent embeddings
        agent_x1 = x1[:, :nag]
        if self.pi_ag_attn:
            ag_mask = entity_mask[:, :nag]
            active_mask = groupmask2attnmask(ag_mask)
            inverse_causal_mask = th.diag(
                th.ones(nag, device=ag_mask.device)
            ).cumsum(dim=1).transpose(0,1).to(th.uint8)
            ag_attn_mask = (active_mask + inverse_causal_mask).min(th.ones_like(active_mask))
            agent_embeds = agent_x1 + self.attn(
                F.relu(agent_x1), pre_mask=ag_attn_mask, post_mask=ag_mask)
        else:
            agent_embeds = agent_x1

        repeat = 1
        if n_proposals > 0:
            repeat = n_proposals
        repeat_fn = lambda x: x.repeat_interleave(repeat, dim=0)

        if self.pi_autoreg:
            allocs, stats = self._autoreg_forward(
                task_x2, task_nonag_cnt, agent_embeds, batch['task_mask'], entity_mask, avail_actions,
                calc_stats=calc_stats, test_mode=test_mode, repeat_fn=repeat_fn)
        else:
            allocs, stats = self._standard_forward(
                task_x2, task_nonag_cnt, agent_embeds, batch['task_mask'], entity_mask,
                calc_stats=calc_stats, test_mode=test_mode, repeat_fn=repeat_fn)

        if n_proposals > 1:
            allocs = allocs.reshape(-1, n_proposals, nag, self.args.n_tasks)
        if calc_stats:
            stats['best_prob'] = stats['best_prob'] / repeat_fn(1 - entity_mask[:, :nag].float()).sum(dim=1, keepdim=True)
            for k, v in stats.items():
                stats[k] = v.reshape(-1, n_proposals, *v.shape[1:])
            return allocs, stats
        return allocs


class MatchingPPOAllocPolicy(nn.Module):
    """
    直接式 matching PPO selector。

    每个高层决策点并行构造：
    - agent embeddings: [bs, n_agents, d]
    - task embeddings:  [bs, n_tasks, d]
    - matching scores:  [bs, n_agents, n_tasks]

    然后每个 agent 只根据自己的 task score 构造 categorical 分布：
    - 训练时 sample
    - 测试时 argmax

    joint logprob 定义为所有 agent logprob 的和。
    """

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_tasks = args.n_tasks
        self.entity_dim = input_shape

        hidden = args.attn_embed_dim
        self.global_enc = nn.Sequential(
            nn.Linear(self.entity_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.task_enc = nn.Sequential(
            nn.Linear(self.entity_dim + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.agent_enc = nn.Sequential(
            nn.Linear(self.entity_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.global_to_task = nn.Linear(hidden, hidden)
        self.global_to_agent = nn.Linear(hidden, hidden)
        self.register_buffer(
            "scale_factor",
            th.scalar_tensor(float(hidden)).sqrt()
        )

    def _build_features(self, batch):
        """
        从 batch 中构造高层 PPO 需要的三类输入特征：
        1. global_state: 全局状态摘要
        2. task_feats: 每个 task 的特征
        3. agent_feats: 每个 agent 的特征
        """

        # 所有实体的特征，形状通常是 (bs, n_entities, entity_dim)
        entities = batch["entities"]

        # entity_mask：1 表示无效实体，0 表示有效实体
        # 转成 float 方便后面做乘法
        entity_mask = batch["entity_mask"].float()

        # entity2task_mask 原来一般是“1 表示不属于该 task”
        # 所以这里取反，得到 entity2task：
        # 1 表示属于该 task，0 表示不属于
        entity2task = 1 - batch["entity2task_mask"].float()

        # 前 n_agents 个实体默认视为 agent 自身
        ag_entities = entities[:, :self.n_agents]

        # 前 n_agents 个实体对应的 mask
        ag_mask = entity_mask[:, :self.n_agents]

        # 后面的实体默认视为非 agent 实体（环境中的目标、建筑、敌人等）
        nonag_entities = entities[:, self.n_agents:]

        # 非 agent 实体对应的 mask
        nonag_mask = entity_mask[:, self.n_agents:]

        # 非 agent 实体到 task 的归属矩阵
        # 形状通常是 (bs, n_nonag_entities, n_tasks)
        nonag_assign = entity2task[:, self.n_agents:]

        # active_nonag = 1 表示该非 agent 实体有效
        # 因为 nonag_mask 中 1 表示无效，所以这里用 1 - mask
        active_nonag = 1.0 - nonag_mask

        # 把无效实体从 task 聚合里去掉
        # active_nonag.unsqueeze(-1) 变成 (bs, n_nonag, 1)
        # 这样就能和 nonag_assign 对齐相乘
        weighted_assign = nonag_assign * active_nonag.unsqueeze(-1)

        # 统计每个 task 当前关联了多少有效非 agent 实体
        # sum(dim=1) 是沿着实体维求和，得到 (bs, n_tasks)
        # clamp_min(1.0) 是为了防止后面除以 0
        task_counts = weighted_assign.sum(dim=1).clamp_min(1.0)

        # 用 batch matrix multiply 聚合每个 task 的实体特征
        # weighted_assign.transpose(1, 2): (bs, n_tasks, n_nonag)
        # nonag_entities:                  (bs, n_nonag, entity_dim)
        # 结果:                            (bs, n_tasks, entity_dim)
        # 再除以 task_counts，得到 task 的平均实体特征
        task_feats = th.bmm(weighted_assign.transpose(1, 2), nonag_entities) / task_counts.unsqueeze(-1)

        # task_mask：1 表示无效 task，0 表示有效 task
        # 所以 1 - task_mask = 1 表示有效 task
        task_valid = 1.0 - batch["task_mask"].float()

        # 这里给每个 task 再拼 3 个额外特征：
        # 1. 有效性标记 task_valid
        # 2. 归一化后的 task_counts
        # 3. 常数 bias = 1
        #
        # 最终 task_feats 维度从 entity_dim 变成 entity_dim + 3
        task_feats = th.cat(
            [
                task_feats,                                      # 聚合后的 task 实体特征
                task_valid.unsqueeze(-1),                        # 当前 task 是否有效
                (task_counts * COUNT_NORM_FACTOR).unsqueeze(-1), # task 中有效实体数量（归一化）
                th.ones_like(task_valid.unsqueeze(-1)),          # 常数偏置项
            ],
            dim=-1,  # 在最后一维拼接
        )

        # 对所有实体求全局活跃标记
        active_entities = 1.0 - entity_mask

        # 统计当前样本里有多少有效实体
        # keepdim=True 让它保持二维，便于后面广播
        global_denom = active_entities.sum(dim=1, keepdim=True).clamp_min(1.0)

        # 用所有有效实体的平均特征作为 global_state
        # 先把无效实体特征乘成 0，再沿实体维求和，然后除以有效实体数
        global_state = (entities * active_entities.unsqueeze(-1)).sum(dim=1) / global_denom

        # 再给 global_state 拼一个“当前有效实体比例/规模特征”
        # global_denom / entities.shape[1] 表示有效实体占总实体数的比例
        global_state = th.cat([global_state, global_denom / entities.shape[1]], dim=-1)

        # agent_feats = agent 本身特征 + 是否有效标记
        # ag_mask 中 1 表示无效，所以 1-ag_mask 表示有效
        agent_feats = th.cat([ag_entities, (1.0 - ag_mask).unsqueeze(-1)], dim=-1)

        # 返回高层 PPO 用到的三类特征
        return global_state, task_feats, agent_feats

    def _compute_matching_scores(self, batch):
        global_state, task_feats, agent_feats = self._build_features(batch)
        global_embed = self.global_enc(global_state)

        task_embed = self.task_enc(task_feats)
        task_embed = task_embed + self.global_to_task(global_embed).unsqueeze(1)

        agent_embed = self.agent_enc(agent_feats)
        agent_embed = agent_embed + self.global_to_agent(global_embed).unsqueeze(1)

        scores = th.bmm(agent_embed, task_embed.transpose(1, 2)) / self.scale_factor
        agent_active = 1.0 - batch["entity_mask"][:, :self.n_agents].float()
        return scores, agent_active

    def sample_allocation(self, batch, test_mode=False):
        """
        并行计算所有 agent-task score，并为每个 agent 独立采样 task。
        """
        device = batch["entities"].device
        scores, agent_active = self._compute_matching_scores(batch)
        bs = scores.shape[0]
        task_mask = batch["task_mask"].bool()
        masked_scores = scores.masked_fill(task_mask.unsqueeze(1), th.finfo(scores.dtype).min)
        dist = Categorical(logits=masked_scores)
        if test_mode:
            action_seq = masked_scores.argmax(dim=-1)
        else:
            action_seq = dist.sample()

        log_prob_seq = dist.log_prob(action_seq) * agent_active
        entropy_seq = dist.entropy() * agent_active
        alloc = F.one_hot(action_seq, num_classes=self.n_tasks).float()
        alloc = alloc * agent_active.unsqueeze(-1)

        stats = {
            "action_seq": action_seq,
            "log_prob_seq": log_prob_seq,
            "log_prob": log_prob_seq.sum(dim=1, keepdim=True),
            "entropy": entropy_seq.sum(dim=1, keepdim=True),
        }
        return alloc.to(device), stats

    def evaluate_actions(self, batch, action_seq):
        """
        并行重算 rollout 动作在当前策略下的 joint logprob，不回放自回归序列。
        """
        scores, agent_active = self._compute_matching_scores(batch)
        task_mask = batch["task_mask"].bool()
        masked_scores = scores.masked_fill(task_mask.unsqueeze(1), th.finfo(scores.dtype).min)
        dist = Categorical(logits=masked_scores)
        action_seq = action_seq.long()
        log_prob_seq = dist.log_prob(action_seq) * agent_active
        entropy_seq = dist.entropy() * agent_active
        alloc = F.one_hot(action_seq, num_classes=self.n_tasks).float()
        alloc = alloc * agent_active.unsqueeze(-1)
        return {
            "alloc": alloc,
            "log_prob_seq": log_prob_seq,
            "log_prob": log_prob_seq.sum(dim=1, keepdim=True),
            "entropy": entropy_seq.sum(dim=1, keepdim=True),
        }


AutoregressivePPOAllocPolicy = MatchingPPOAllocPolicy


class MAPPOAllocPolicy(nn.Module):
    """
    MAPPO-style decentralized high-level allocator.

    The actor is shared across agents, but each agent's task distribution is
    computed from that agent's own entity row plus task features. Unlike
    MatchingPPOAllocPolicy, this actor does not inject a pooled global embedding
    into every agent/task representation.
    """

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_tasks = args.n_tasks
        self.entity_dim = input_shape

        hidden = args.attn_embed_dim
        self.use_agent_id = bool(args.hier_agent.get("use_agent_id", True))
        self.use_iga = bool(args.hier_agent.get("use_iga", False))

        self.agent_enc = nn.Sequential(
            nn.Linear(self.entity_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.task_enc = nn.Sequential(
            nn.Linear(self.entity_dim + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        if self.use_agent_id:
            self.agent_id_embed = nn.Embedding(self.n_agents, hidden)
        self.register_buffer(
            "scale_factor",
            th.scalar_tensor(float(hidden)).sqrt()
        )
        if self.use_iga:
            self.iga_refiner = IntentionGraphAttentionRefinement(
                args,
                agent_dim=hidden,
                task_dim=hidden,
                n_agents=self.n_agents,
                n_tasks=self.n_tasks,
            )

    def _build_features(self, batch):
        entities = batch["entities"]
        entity_mask = batch["entity_mask"].float()
        entity2task = 1 - batch["entity2task_mask"].float()

        ag_entities = entities[:, :self.n_agents]
        ag_mask = entity_mask[:, :self.n_agents]
        agent_feats = th.cat([ag_entities, (1.0 - ag_mask).unsqueeze(-1)], dim=-1)

        nonag_entities = entities[:, self.n_agents:]
        nonag_mask = entity_mask[:, self.n_agents:]
        nonag_assign = entity2task[:, self.n_agents:]
        active_nonag = 1.0 - nonag_mask
        weighted_assign = nonag_assign * active_nonag.unsqueeze(-1)
        task_counts = weighted_assign.sum(dim=1).clamp_min(1.0)
        task_feats = th.bmm(weighted_assign.transpose(1, 2), nonag_entities) / task_counts.unsqueeze(-1)

        task_valid = 1.0 - batch["task_mask"].float()
        task_feats = th.cat(
            [
                task_feats,
                task_valid.unsqueeze(-1),
                (task_counts * COUNT_NORM_FACTOR).unsqueeze(-1),
                th.ones_like(task_valid.unsqueeze(-1)),
            ],
            dim=-1,
        )
        return agent_feats, task_feats

    def _compute_base_logits(self, batch):
        agent_feats, task_feats = self._build_features(batch)
        agent_embed = self.agent_enc(agent_feats)
        if self.use_agent_id:
            agent_ids = th.arange(self.n_agents, device=agent_embed.device)
            agent_embed = agent_embed + self.agent_id_embed(agent_ids).unsqueeze(0)
        task_embed = self.task_enc(task_feats)
        logits = th.bmm(agent_embed, task_embed.transpose(1, 2)) / self.scale_factor
        agent_active = 1.0 - batch["entity_mask"][:, :self.n_agents].float()
        return logits, agent_embed, task_embed, agent_active

    def _compute_logits(self, batch):
        base_logits, agent_embed, task_embed, agent_active = self._compute_base_logits(batch)
        iga_stats = {}
        if self.use_iga:
            logits, iga_stats = self.iga_refiner(
                base_logits=base_logits,
                agent_embed=agent_embed,
                task_embed=task_embed,
                agent_active=agent_active,
                task_mask=batch["task_mask"],
            )
        else:
            logits = base_logits
        return logits, base_logits, agent_active, iga_stats

    def _masked_dist(self, batch):
        logits, base_logits, agent_active, iga_stats = self._compute_logits(batch)
        task_mask = batch["task_mask"].bool().unsqueeze(1)
        masked_logits = logits.masked_fill(task_mask, th.finfo(logits.dtype).min)
        masked_base_logits = base_logits.masked_fill(task_mask, th.finfo(base_logits.dtype).min)
        return Categorical(logits=masked_logits), masked_logits, masked_base_logits, agent_active, iga_stats

    def _hard_assignment_stats(self, action_seq, agent_active, task_mask, priority_logits=None):
        active_actions = F.one_hot(action_seq, num_classes=self.n_tasks).float()
        active_actions = active_actions * agent_active.unsqueeze(-1)
        valid_task = ~task_mask.bool()
        task_valid = valid_task.float()
        counts = active_actions.sum(dim=1) * task_valid
        coverage = (counts > 0).float().sum(dim=-1)
        max_load = counts.max(dim=-1)[0]
        valid_task_count = task_valid.sum(dim=-1).clamp_min(1.0)
        active_agent_count = agent_active.sum(dim=-1).clamp_min(1.0)
        normalized_coverage = coverage / valid_task_count
        normalized_max_load = max_load / active_agent_count

        over_threshold = int(self.args.hier_agent.get("iga_over_alloc_threshold", 2))
        over_alloc = ((counts > over_threshold) & valid_task).float().sum(dim=-1) / valid_task_count

        load_mean = counts.sum(dim=-1, keepdim=True) / valid_task_count.unsqueeze(-1)
        load_var = (((counts - load_mean) * task_valid) ** 2).sum(dim=-1) / valid_task_count
        hard_load_std = th.sqrt(load_var + 1e-8)

        load_prob = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        load_prob = load_prob * task_valid
        load_entropy = -(load_prob * load_prob.clamp_min(1e-8).log()).sum(dim=-1)
        max_entropy = valid_task_count.clamp_min(2.0).log()
        normalized_load_entropy = load_entropy / max_entropy.clamp_min(1e-8)

        important_topk = int(self.args.hier_agent.get("iga_important_topk", 3))
        if priority_logits is None:
            task_priority = counts
        else:
            priority_probs = F.softmax(priority_logits, dim=-1).masked_fill(~valid_task.unsqueeze(1), 0.0)
            task_priority = (priority_probs * agent_active.unsqueeze(-1)).sum(dim=1)
            task_priority = task_priority / active_agent_count.unsqueeze(-1)
        task_priority = task_priority.masked_fill(~valid_task, -1e10)
        k = max(1, min(important_topk, self.n_tasks))
        important_idx = task_priority.topk(k=k, dim=-1).indices
        important_mask = th.zeros_like(valid_task, dtype=th.bool)
        important_mask.scatter_(dim=1, index=important_idx, value=True)
        important_mask = important_mask & valid_task
        important_count = important_mask.float().sum(dim=-1).clamp_min(1.0)
        covered_mask = counts > 0
        important_coverage = ((important_mask & covered_mask).float().sum(dim=-1) / important_count)
        important_miss_rate = ((important_mask & ~covered_mask).float().sum(dim=-1) / important_count)

        coordination_score = normalized_coverage - normalized_max_load

        stats = {
            "iga_hard_max_load": max_load.mean().detach(),
            "iga_hard_coverage": coverage.mean().detach(),
            "iga_over_allocation_ratio": over_alloc.mean().detach(),
            "iga_normalized_hard_coverage": normalized_coverage.mean().detach(),
            "iga_normalized_hard_max_load": normalized_max_load.mean().detach(),
            "iga_hard_load_std": hard_load_std.mean().detach(),
            "iga_normalized_hard_load_entropy": normalized_load_entropy.mean().detach(),
            "iga_important_task_coverage": important_coverage.mean().detach(),
            "iga_important_task_miss_rate": important_miss_rate.mean().detach(),
            "iga_coordination_score": coordination_score.mean().detach(),
        }
        return stats

    def _counterfactual_assignment_stats(
            self, masked_logits, masked_base_logits, agent_active, task_mask):
        """Compare greedy base/refined allocations at exactly the same state."""
        with th.no_grad():
            base_actions = masked_base_logits.argmax(dim=-1)
            refined_actions = masked_logits.argmax(dim=-1)
            base_stats = self._hard_assignment_stats(
                base_actions, agent_active, task_mask,
                priority_logits=masked_base_logits,
            )
            refined_stats = self._hard_assignment_stats(
                refined_actions, agent_active, task_mask,
                priority_logits=masked_base_logits,
            )
            metric_sources = {
                "coverage": "iga_normalized_hard_coverage",
                "max_load": "iga_hard_max_load",
                "load_std": "iga_hard_load_std",
                "over_allocation": "iga_over_allocation_ratio",
                "coordination_score": "iga_coordination_score",
                "important_coverage": "iga_important_task_coverage",
            }
            stats = {}
            for metric_name, source_key in metric_sources.items():
                base_value = base_stats[source_key]
                refined_value = refined_stats[source_key]
                prefix = "alloc_counterfactual/"
                stats[prefix + "base_{}".format(metric_name)] = base_value.detach()
                stats[prefix + "refined_{}".format(metric_name)] = refined_value.detach()
                # Delta is always refined minus base. Negative is better for
                # max_load, load_std, and over_allocation.
                stats[prefix + "{}_delta".format(metric_name)] = (
                    refined_value - base_value
                ).detach()
            return stats

    def sample_allocation(self, batch, test_mode=False):
        dist, masked_logits, masked_base_logits, agent_active, iga_stats = self._masked_dist(batch)
        if test_mode:
            action_seq = masked_logits.argmax(dim=-1)
        else:
            action_seq = dist.sample()

        log_probs = dist.log_prob(action_seq) * agent_active
        entropy = dist.entropy() * agent_active
        alloc = F.one_hot(action_seq, num_classes=self.n_tasks).float()
        alloc = alloc * agent_active.unsqueeze(-1)

        stats = {
            "action_seq": action_seq,
            "actions": action_seq.unsqueeze(-1),
            "log_probs": log_probs.unsqueeze(-1),
            "log_prob_seq": log_probs,
            "joint_log_prob": log_probs.sum(dim=1, keepdim=True),
            "log_prob": log_probs.sum(dim=1, keepdim=True),
            "entropy": entropy.unsqueeze(-1),
            "joint_entropy": entropy.sum(dim=1, keepdim=True),
            "agent_mask": agent_active.unsqueeze(-1),
        }
        stats.update(iga_stats)
        stats.update(self._hard_assignment_stats(
            action_seq, agent_active, batch["task_mask"], priority_logits=masked_base_logits
        ))
        stats.update(self._counterfactual_assignment_stats(
            masked_logits, masked_base_logits, agent_active, batch["task_mask"]
        ))
        return alloc.to(batch["entities"].device), stats

    def evaluate_actions(self, batch, action_seq):
        dist, masked_logits, masked_base_logits, agent_active, iga_stats = self._masked_dist(batch)
        if action_seq.dim() == 3:
            action_seq = action_seq.squeeze(-1)
        action_seq = action_seq.long()

        log_probs = dist.log_prob(action_seq) * agent_active
        entropy = dist.entropy() * agent_active
        alloc = F.one_hot(action_seq, num_classes=self.n_tasks).float()
        alloc = alloc * agent_active.unsqueeze(-1)
        rets = {
            "alloc": alloc,
            "log_probs": log_probs.unsqueeze(-1),
            "log_prob_seq": log_probs,
            "joint_log_prob": log_probs.sum(dim=1, keepdim=True),
            "log_prob": log_probs.sum(dim=1, keepdim=True),
            "entropy": entropy.unsqueeze(-1),
            "joint_entropy": entropy.sum(dim=1, keepdim=True),
            "agent_mask": agent_active.unsqueeze(-1),
        }
        rets.update(iga_stats)
        rets.update(self._hard_assignment_stats(
            action_seq, agent_active, batch["task_mask"], priority_logits=masked_base_logits
        ))
        rets.update(self._counterfactual_assignment_stats(
            masked_logits, masked_base_logits, agent_active, batch["task_mask"]
        ))
        return rets
