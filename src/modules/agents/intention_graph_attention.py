import torch as th
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(logits, mask=None, dim=-1):
    if mask is None:
        return F.softmax(logits, dim=dim)
    masked_logits = logits.masked_fill(mask.bool(), -1e10)
    probs = F.softmax(masked_logits, dim=dim)
    return probs.masked_fill(mask.bool(), 0.0)


class IntentionGraphAttentionRefinement(nn.Module):
    """
    Intention-aware graph refinement for high-level MAPPO allocation logits.

    The module keeps the original actor intact: it consumes base logits and
    actor embeddings, then returns refined logits used by sampling/evaluation.
    """

    def __init__(self, args, agent_dim, task_dim, n_agents, n_tasks):
        super().__init__()
        self.args = args
        self.n_agents = n_agents
        self.n_tasks = n_tasks
        cfg = args.hier_agent

        self.hidden_dim = int(cfg.get("iga_hidden_dim", 64))
        self.num_heads = int(cfg.get("iga_num_heads", 2))
        self.delta_scale = float(cfg.get("iga_delta_scale", 0.1))
        self.center_delta = bool(cfg.get("iga_center_delta", True))
        self.use_agent_graph = bool(cfg.get("iga_use_agent_graph", True))
        self.use_task_graph = bool(cfg.get("iga_use_task_graph", True))
        self.use_agent_task_graph = bool(cfg.get("iga_use_agent_task_graph", True))
        self.use_task_load = bool(cfg.get("iga_use_task_load", True))
        self.dropout = float(cfg.get("iga_dropout", 0.0))
        self.debug_checks = bool(cfg.get("iga_debug_checks", False))
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("iga_hidden_dim must be divisible by iga_num_heads")

        self.agent_proj = nn.Linear(agent_dim, self.hidden_dim)
        self.task_proj = nn.Linear(task_dim, self.hidden_dim)
        self.intent_proj = nn.Linear(n_tasks, self.hidden_dim)
        self.agent_input_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        if self.use_task_load:
            self.task_load_proj = nn.Linear(2, self.hidden_dim)

        if self.use_agent_graph:
            self.agent_attn = nn.MultiheadAttention(
                self.hidden_dim, self.num_heads, dropout=self.dropout, batch_first=True
            )
        if self.use_task_graph:
            self.task_attn = nn.MultiheadAttention(
                self.hidden_dim, self.num_heads, dropout=self.dropout, batch_first=True
            )
        self.agent_ln = nn.LayerNorm(self.hidden_dim)
        self.task_ln = nn.LayerNorm(self.hidden_dim)

        if self.use_agent_task_graph:
            pair_dim = self.hidden_dim * 4 + 2 + 1
            self.agent_task_mlp = nn.Sequential(
                nn.Linear(pair_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
        else:
            # Controlled Agent-Task relation ablation. This head consumes one
            # whole task-logit row at a time instead of constructing an
            # explicit feature for every (agent, task) edge. Agent and task
            # graph information is retained through agent_context and the
            # pooled task_context, respectively.
            nonrel_input_dim = self.hidden_dim * 2 + self.n_tasks
            self.nonrelational_residual_mlp = nn.Sequential(
                nn.Linear(nonrel_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.n_tasks),
            )

    def _attn_stats(self, prefix, attn):
        if attn is None:
            zero = th.tensor(0.0, device=self.agent_proj.weight.device)
            return {
                "{}_attn_entropy".format(prefix): zero.detach(),
                "{}_attn_max".format(prefix): zero.detach(),
            }
        attn = attn.clamp_min(0.0)
        entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1).mean()
        return {
            "{}_attn_entropy".format(prefix): entropy.detach(),
            "{}_attn_max".format(prefix): attn.max().detach(),
        }

    def _top12_gap(self, logits, valid_mask=None):
        if logits.shape[-1] < 2:
            return th.zeros((), device=logits.device, dtype=logits.dtype)
        if valid_mask is None:
            values = th.topk(logits, k=2, dim=-1).values
            return (values[..., 0] - values[..., 1]).mean()
        valid_counts = valid_mask.float().sum(dim=-1)
        can_top2 = valid_counts >= 2
        if not can_top2.any():
            return th.zeros((), device=logits.device, dtype=logits.dtype)
        values = th.topk(logits, k=2, dim=-1).values
        gaps = values[..., 0] - values[..., 1]
        return gaps[can_top2].mean()

    def _policy_shift_stats(self, masked_base_logits, refined_logits, pair_task_mask, agent_active):
        with th.no_grad():
            if pair_task_mask is None:
                valid = th.ones_like(masked_base_logits, dtype=th.bool)
            else:
                valid = ~pair_task_mask
            valid_agent = valid.any(dim=-1) & agent_active.bool()
            if not valid_agent.any():
                zero = th.zeros((), device=masked_base_logits.device, dtype=masked_base_logits.dtype)
                return {
                    "iga_argmax_change_rate": zero,
                    "iga_prob_l1_change": zero,
                    "iga_prob_kl_change": zero,
                    "iga_base_policy_entropy": zero,
                    "iga_refined_policy_entropy": zero,
                    "iga_entropy_change": zero,
                }

            base_prob = F.softmax(masked_base_logits, dim=-1).masked_fill(~valid, 0.0)
            refined_prob = F.softmax(refined_logits, dim=-1).masked_fill(~valid, 0.0)

            base_argmax = masked_base_logits.argmax(dim=-1)
            refined_argmax = refined_logits.argmax(dim=-1)
            argmax_change = ((base_argmax != refined_argmax) & valid_agent).float()
            argmax_change_rate = argmax_change.sum() / valid_agent.float().sum().clamp_min(1.0)

            prob_l1 = (base_prob - refined_prob).abs().sum(dim=-1)
            prob_l1_change = prob_l1[valid_agent].mean()

            eps = 1e-8
            prob_kl = refined_prob * (
                refined_prob.clamp_min(eps).log() - base_prob.clamp_min(eps).log()
            )
            prob_kl_change = prob_kl.sum(dim=-1)[valid_agent].mean()

            base_entropy = -(base_prob * base_prob.clamp_min(eps).log()).sum(dim=-1)
            refined_entropy = -(refined_prob * refined_prob.clamp_min(eps).log()).sum(dim=-1)
            base_entropy_mean = base_entropy[valid_agent].mean()
            refined_entropy_mean = refined_entropy[valid_agent].mean()

            return {
                "iga_argmax_change_rate": argmax_change_rate.detach(),
                "iga_prob_l1_change": prob_l1_change.detach(),
                "iga_prob_kl_change": prob_kl_change.detach(),
                "iga_base_policy_entropy": base_entropy_mean.detach(),
                "iga_refined_policy_entropy": refined_entropy_mean.detach(),
                "iga_entropy_change": (refined_entropy_mean - base_entropy_mean).detach(),
            }

    def _task_load_stats(self, task_load_flat, pair_task_mask):
        with th.no_grad():
            valid_task = (
                th.ones_like(task_load_flat, dtype=th.bool)
                if pair_task_mask is None else (~pair_task_mask).any(dim=1)
            )
            valid_task_float = valid_task.float()
            valid_task_count = valid_task_float.sum(dim=-1).clamp_min(1.0)
            masked_load = task_load_flat * valid_task_float

            per_sample_max = masked_load.max(dim=-1).values
            load_sum = masked_load.sum(dim=-1).clamp_min(1e-8)
            load_mean = load_sum / valid_task_count
            load_var = (((masked_load - load_mean.unsqueeze(-1)) * valid_task_float) ** 2).sum(dim=-1)
            load_var = load_var / valid_task_count

            if task_load_flat.shape[-1] >= 2:
                top2 = masked_load.masked_fill(~valid_task, -1e10).topk(k=2, dim=-1).values
                can_top2 = valid_task_count >= 2
                top2_gap = th.where(
                    can_top2,
                    top2[:, 0] - top2[:, 1],
                    th.zeros_like(per_sample_max),
                )
            else:
                top2_gap = th.zeros_like(per_sample_max)

            load_prob = masked_load / load_sum.unsqueeze(-1)
            load_entropy = -(load_prob * load_prob.clamp_min(1e-8).log()).sum(dim=-1)
            max_entropy = valid_task_count.clamp_min(2.0).log()
            normalized_entropy = load_entropy / max_entropy.clamp_min(1e-8)

            active_agent_count = task_load_flat.sum(dim=-1).clamp_min(1.0)
            max_ratio = per_sample_max / active_agent_count

            return {
                "iga_task_load_max_mean": per_sample_max.mean().detach(),
                "iga_task_load_max_ratio": max_ratio.mean().detach(),
                "iga_task_load_top2_gap": top2_gap.mean().detach(),
                "iga_task_load_entropy": normalized_entropy.mean().detach(),
                "iga_task_load_std_mean": th.sqrt(load_var + 1e-8).mean().detach(),
            }

    def forward(self, base_logits, agent_embed, task_embed, agent_active=None, task_mask=None):
        if self.debug_checks:
            assert not th.isnan(base_logits).any()

        if agent_active is None:
            agent_active = th.ones(
                base_logits.shape[:2], device=base_logits.device, dtype=base_logits.dtype
            )
        else:
            agent_active = agent_active.to(device=base_logits.device, dtype=base_logits.dtype)
        agent_padding_mask = ~agent_active.bool()
        all_agents_masked = agent_padding_mask.all(dim=1)
        safe_agent_padding_mask = agent_padding_mask.clone()
        if all_agents_masked.any():
            # MultiheadAttention cannot handle a row where every key is masked.
            safe_agent_padding_mask[all_agents_masked, 0] = False

        bool_task_mask = task_mask.bool() if task_mask is not None else None
        if bool_task_mask is not None and bool_task_mask.dim() == 2:
            pair_task_mask = bool_task_mask.unsqueeze(1).expand_as(base_logits)
            task_padding_mask = bool_task_mask
        elif bool_task_mask is not None:
            pair_task_mask = bool_task_mask
            # A task is unavailable to the task graph when no agent can select it.
            task_padding_mask = pair_task_mask.all(dim=1)
        else:
            pair_task_mask = None
            task_padding_mask = th.zeros(
                base_logits.shape[0], base_logits.shape[-1],
                device=base_logits.device, dtype=th.bool,
            )
        all_tasks_masked = task_padding_mask.all(dim=1)
        safe_task_padding_mask = task_padding_mask.clone()
        if all_tasks_masked.any():
            # MultiheadAttention cannot handle a row where every key is masked.
            # Its result is cleared below, so temporarily exposing one key is safe.
            safe_task_padding_mask[all_tasks_masked, 0] = False

        masked_base_logits = (
            base_logits.masked_fill(pair_task_mask, -1e10)
            if pair_task_mask is not None else base_logits
        )
        agent_intention = masked_softmax(base_logits, pair_task_mask, dim=-1)
        agent_intention = agent_intention * agent_active.unsqueeze(-1)

        agent_proj = self.agent_proj(agent_embed)
        task_proj = self.task_proj(task_embed)
        intent_embed = self.intent_proj(agent_intention)
        agent_graph_input = self.agent_input_proj(th.cat([agent_proj, intent_embed], dim=-1))

        agent_attn_weights = None
        if self.use_agent_graph:
            agent_ctx, agent_attn_weights = self.agent_attn(
                agent_graph_input, agent_graph_input, agent_graph_input,
                key_padding_mask=safe_agent_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            agent_context = self.agent_ln(agent_graph_input + agent_ctx)
            agent_context = agent_context * agent_active.unsqueeze(-1)
        else:
            agent_context = self.agent_ln(agent_graph_input) * agent_active.unsqueeze(-1)

        valid_intention = agent_intention * agent_active.unsqueeze(-1)
        task_load = valid_intention.sum(dim=1).unsqueeze(-1)
        active_agent_count = agent_active.sum(dim=1, keepdim=True).clamp_min(1.0)
        task_load_norm = task_load / active_agent_count.unsqueeze(-1)
        task_load_feat = th.cat([task_load, task_load_norm], dim=-1)
        if self.use_task_load:
            task_graph_input = task_proj + self.task_load_proj(task_load_feat)
        else:
            task_graph_input = task_proj

        task_attn_weights = None
        if self.use_task_graph:
            task_ctx, task_attn_weights = self.task_attn(
                task_graph_input, task_graph_input, task_graph_input,
                key_padding_mask=safe_task_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            task_context = self.task_ln(task_graph_input + task_ctx)
        else:
            task_context = self.task_ln(task_graph_input)
        task_context = task_context.masked_fill(task_padding_mask.unsqueeze(-1), 0.0)

        if self.use_agent_task_graph:
            bs, na, nt = base_logits.shape
            agent_pair = agent_proj.unsqueeze(2).expand(bs, na, nt, self.hidden_dim)
            agent_ctx_pair = agent_context.unsqueeze(2).expand(bs, na, nt, self.hidden_dim)
            task_pair = task_proj.unsqueeze(1).expand(bs, na, nt, self.hidden_dim)
            task_ctx_pair = task_context.unsqueeze(1).expand(bs, na, nt, self.hidden_dim)
            if self.use_task_load:
                load_pair = task_load_feat.unsqueeze(1).expand(bs, na, nt, 2)
            else:
                # Preserve the relational head shape for a controlled
                # ablation, but remove all task-load information.
                load_pair = base_logits.new_zeros(bs, na, nt, 2)
            pair_feat = th.cat(
                [
                    agent_pair,
                    agent_ctx_pair,
                    task_pair,
                    task_ctx_pair,
                    load_pair,
                    base_logits.unsqueeze(-1),
                ],
                dim=-1,
            )
            delta_logits = self.agent_task_mlp(pair_feat).squeeze(-1)
        else:
            # Keep both homogeneous graphs, but remove explicit per-edge
            # Agent-Task feature construction. Predict a complete residual
            # task vector for each agent from unary/global context instead.
            valid_tasks = (~task_padding_mask).to(task_context.dtype)
            valid_task_count = valid_tasks.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled_task_context = (
                task_context * valid_tasks.unsqueeze(-1)
            ).sum(dim=1) / valid_task_count
            pooled_task_context = pooled_task_context.unsqueeze(1).expand(
                -1, base_logits.shape[1], -1
            )
            base_logit_features = (
                base_logits.masked_fill(pair_task_mask, 0.0)
                if pair_task_mask is not None else base_logits
            )
            nonrelational_features = th.cat(
                [agent_context, pooled_task_context, base_logit_features],
                dim=-1,
            )
            delta_logits = self.nonrelational_residual_mlp(nonrelational_features)
            delta_logits = delta_logits * agent_active.unsqueeze(-1)

        valid = None if pair_task_mask is None else (~pair_task_mask).float()
        if valid is not None:
            delta_logits = delta_logits * valid
        if self.center_delta:
            if valid is None:
                centered_delta = delta_logits - delta_logits.mean(dim=-1, keepdim=True)
            else:
                valid_count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
                delta_mean = (delta_logits * valid).sum(dim=-1, keepdim=True) / valid_count
                centered_delta = (delta_logits - delta_mean) * valid
        else:
            centered_delta = delta_logits if valid is None else delta_logits * valid

        bounded_delta = th.tanh(centered_delta)
        refined_logits = base_logits + self.delta_scale * bounded_delta
        if pair_task_mask is not None:
            refined_logits = refined_logits.masked_fill(pair_task_mask, -1e10)

        task_load_flat = task_load.squeeze(-1)
        stats = {
            "iga_delta_abs_mean": bounded_delta.abs().mean().detach(),
            "iga_delta_std": bounded_delta.std().detach(),
            "iga_delta_task_std_mean": bounded_delta.std(dim=-1).mean().detach(),
            "iga_task_load_max": task_load.max().detach(),
            "iga_task_load_std": task_load_flat.std(dim=-1).mean().detach(),
            "iga_base_top12_gap_mean": self._top12_gap(masked_base_logits, None if pair_task_mask is None else ~pair_task_mask).detach(),
            "iga_refined_top12_gap_mean": self._top12_gap(refined_logits, None if pair_task_mask is None else ~pair_task_mask).detach(),
        }
        stats["iga_gap_ratio"] = (
            stats["iga_refined_top12_gap_mean"] / (stats["iga_base_top12_gap_mean"] + 1e-8)
        ).detach()
        stats.update(self._attn_stats("iga_agent", agent_attn_weights))
        stats.update(self._attn_stats("iga_task", task_attn_weights))
        stats.update(self._policy_shift_stats(masked_base_logits, refined_logits, pair_task_mask, agent_active))
        stats.update(self._task_load_stats(task_load_flat, pair_task_mask))

        if self.debug_checks:
            assert not th.isnan(refined_logits).any()
            assert not th.isinf(refined_logits).any()
            if pair_task_mask is not None:
                assert (refined_logits[pair_task_mask] == -1e10).all()
                assert (agent_intention[pair_task_mask].abs() < 1e-5).all()

        return refined_logits, stats
