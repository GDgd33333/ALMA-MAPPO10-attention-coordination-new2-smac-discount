import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ..layers import EntityAttentionLayer
from .allocation_common import groupmask2attnmask, TaskEmbedder, CountEmbedder
from utils.rl_utils import ExponentialMeanStd


class StandardAllocCritic(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args

        self.args = args
        self.in_fc_ent = nn.Linear(input_shape, args.alloc_embed_dim)
        self.in_fc_alloc = TaskEmbedder(args.alloc_embed_dim, args)
        self.attn = EntityAttentionLayer(args.alloc_embed_dim,
                                         args.alloc_embed_dim,
                                         args.alloc_embed_dim, args,
                                         n_heads=args.alloc_n_heads,
                                         use_layernorm=False)
        self.count_embed = CountEmbedder(args.alloc_embed_dim, args)
        self.out_dim = 1
        self.out_fc = nn.Linear(args.alloc_embed_dim, self.out_dim)

        self.use_popart = self.args.popart
        if self.use_popart:
            self.targ_rms = ExponentialMeanStd(alpha=0.01)
            self.popart_weight = nn.parameter.Parameter(
                th.ones(1, self.out_dim))
            self.popart_bias = nn.parameter.Parameter(
                th.zeros(1, self.out_dim))

    def load_state_dict(self, state_dict):
        if self.use_popart:
            targ_rms_state_dict, state_dict = state_dict
            self.targ_rms.load_state_dict(targ_rms_state_dict)
        super().load_state_dict(state_dict)

    def state_dict(self):
        if self.use_popart:
            return self.targ_rms.state_dict(), super().state_dict()
        return super().state_dict()

    def popart_update(self, targets, mask):
        assert self.use_popart
        if self.targ_rms.mean is not None:
            old_mean = self.targ_rms.mean.clone()
            old_var = self.targ_rms.var.clone()
            self.targ_rms.update(targets, mask)
        else:
            self.targ_rms.update(targets, mask)
            old_mean = self.targ_rms.mean.clone()
            old_var = self.targ_rms.var.clone()
        sd_ratio = old_var.sqrt() / self.targ_rms.var.sqrt()
        self.popart_weight.data.mul_(sd_ratio)
        self.popart_bias.data.mul_(sd_ratio).add_((old_mean - self.targ_rms.mean) / self.targ_rms.var.sqrt())
        return (targets - self.targ_rms.mean) / self.targ_rms.var.sqrt()

    def denormalize(self, q):
        if self.targ_rms.mean is None or not self.use_popart:
            return q
        return q * self.targ_rms.var.sqrt() + self.targ_rms.mean

    def forward(self, batch, override_alloc=None, test_mode=False, calc_stats=False):
        entities = batch['entities']
        bs = entities.shape[0]
        entity_mask = batch['entity_mask']
        attn_mask = groupmask2attnmask(entity_mask)
        entity2task = 1 - batch['entity2task_mask'].float()
        multi_eval = False
        repeat_fn = lambda x: x
        if override_alloc is not None:
            if len(override_alloc.shape) == 4:
                multi_eval = True
                bs, np, na, nt = override_alloc.shape
                override_alloc = override_alloc.reshape(bs * np, na, nt)
                repeat_fn = lambda x: x.repeat_interleave(np, dim=0)
            entity2task = repeat_fn(entity2task)
            entity2task[:, :self.args.n_agents] = override_alloc
        x1_ent = self.in_fc_ent(entities)
        x1_alloc = self.in_fc_alloc(entity2task)
        x1_count = self.count_embed(entity2task)
        x1 = F.relu(repeat_fn(x1_ent) + x1_alloc + x1_count)
        x2 = F.relu(self.attn(x1, pre_mask=repeat_fn(attn_mask),
                              post_mask=repeat_fn(entity_mask[:, :self.args.n_agents])))
        out_shape = (bs, self.out_dim)
        if multi_eval:
            out_shape = (bs, np, self.out_dim)
        out = self.out_fc(x2.mean(dim=1)).reshape(*out_shape)
        if self.use_popart:
            if multi_eval:
                out = out * self.popart_weight.unsqueeze(1) + self.popart_bias.unsqueeze(1)
            else:
                out = out * self.popart_weight + self.popart_bias
        if calc_stats:
            return out, {}
        return out


class PPOAllocCritic(nn.Module):
    """
    高层 PPO allocator 对应的 centralized critic（集中式价值网络）。

    它的作用不是输出具体分配动作，而是评估高层状态价值。
    默认输出 V(s)；仅当 critic_condition_on_alloc=True 时输出 C(s, Z)。

    可以把它理解成：
    - Actor 负责“怎么分”
    - Critic 负责“这次分得好不好”
    """

    def __init__(self, input_shape, args):
        # 调用父类 nn.Module 的初始化函数
        super().__init__()

        # 保存配置参数，后面 forward 和特征构造都要用到
        self.args = args

        # 智能体数量
        self.n_agents = args.n_agents

        # task 数量
        self.n_tasks = args.n_tasks

        # 单个 entity 的原始特征维度
        self.entity_dim = input_shape

        # False estimates a state value V(s). True preserves the legacy
        # allocation-conditioned value C(s, Z) for controlled A/B runs.
        self.critic_condition_on_alloc = self.args.hier_agent.get(
            "critic_condition_on_alloc", False
        )

        # critic 的隐藏层维度
        # 这里使用 alloc_embed_dim 作为 critic 网络的隐藏维度
        hidden = args.alloc_embed_dim

        # 每个 task 的特征维度 = 原始聚合实体特征维度 + 3 个附加特征
        # 这 3 个附加特征在 _build_features 里构造：
        # 1) task_valid
        # 2) task_count 的归一化值
        # 3) 常数 bias
        task_feat_dim = self.entity_dim + 3

        # critic 最终输入维度由 2 个必选部分和 1 个可选部分拼接而成：
        #
        # 1. global_state
        #    维度 = entity_dim + 1
        #    其中 +1 是额外拼接的“活跃实体比例/规模信息”
        #
        # 2. 所有 task 的特征
        #    维度 = n_tasks * task_feat_dim
        #
        # 3. 当前 allocation（one-hot 展平，仅旧模式启用）
        #    维度 = n_agents * n_tasks
        #
        # 所以总输入维度就是下面这个 critic_in_dim
        critic_in_dim = (
            (self.entity_dim + 1)
            + self.n_tasks * task_feat_dim
        )
        if self.critic_condition_on_alloc:
            critic_in_dim += self.n_agents * self.n_tasks
        self.critic_in_dim = critic_in_dim

        # 定义 critic 主网络
        # 输入：拼接后的大向量 x
        # 输出：一个标量 value
        #
        # 这就是一个标准 MLP critic
        self.net = nn.Sequential(
            nn.Linear(critic_in_dim, hidden),  # 第一层：输入映射到 hidden
            nn.ReLU(),                         # 激活函数
            nn.Linear(hidden, hidden),         # 第二层 hidden
            nn.ReLU(),                         # 激活函数
            nn.Linear(hidden, 1),              # 输出一个标量价值
        )

    def _build_features(self, batch):
        """
        从 batch 中构造 critic 用的状态特征。

        和 policy 不同，这里 critic 不需要当前单个 agent 的特征，
        它更关心的是：
        - 当前全局状态 global_state
        - 每个 task 的摘要特征 task_feats

        因为 critic 要评估的是“整张 allocation”整体值不值，
        所以它天然更偏向 centralized/global 视角。
        """

        # 所有实体特征
        # 形状通常是 (bs, n_entities, entity_dim)
        entities = batch["entities"]

        # entity_mask：
        # 1 表示无效实体
        # 0 表示有效实体
        # 转成 float 方便做数值运算
        entity_mask = batch["entity_mask"].float()

        # entity2task_mask 原始语义通常是：
        # 1 表示“不属于这个 task”
        # 0 表示“属于这个 task”
        #
        # 所以这里取反以后：
        # 1 表示“属于这个 task”
        # 0 表示“不属于这个 task”
        entity2task = 1 - batch["entity2task_mask"].float()

        # 取出所有非 agent 实体
        # 默认前 n_agents 个实体是 agent，后面的是 task 相关实体/环境实体
        nonag_entities = entities[:, self.n_agents:]

        # 非 agent 实体对应的 mask
        nonag_mask = entity_mask[:, self.n_agents:]

        # 非 agent 实体到 task 的归属矩阵
        # 形状大致是 (bs, n_nonag_entities, n_tasks)
        nonag_assign = entity2task[:, self.n_agents:]

        # active_nonag：
        # 1 表示该非 agent 实体有效
        # 0 表示无效
        active_nonag = 1.0 - nonag_mask

        # 把实体是否有效也乘进 task 归属权重里
        # 这样无效实体就不会参与 task 聚合
        weighted_assign = nonag_assign * active_nonag.unsqueeze(-1)

        # 统计每个 task 关联了多少个有效非 agent 实体
        # sum(dim=1) 沿实体维求和，得到形状 (bs, n_tasks)
        #
        # clamp_min(1.0) 的作用是防止后面除以 0
        # 也就是说，如果某个 task 当前没有任何实体，也强行把分母当成 1
        task_counts = weighted_assign.sum(dim=1).clamp_min(1.0)

        # 聚合每个 task 的实体特征：
        # weighted_assign.transpose(1, 2): (bs, n_tasks, n_nonag)
        # nonag_entities:                  (bs, n_nonag, entity_dim)
        # bmm 后得到：                     (bs, n_tasks, entity_dim)
        #
        # 再除以 task_counts，得到每个 task 的平均实体特征
        task_feats = th.bmm(weighted_assign.transpose(1, 2), nonag_entities) / task_counts.unsqueeze(-1)

        # task_valid：
        # 原 task_mask 里 1 表示无效 task，0 表示有效
        # 所以取反后 1 表示有效 task
        task_valid = 1.0 - batch["task_mask"].float()

        # 给每个 task 再拼上 3 个附加特征：
        #
        # 1. task_valid：当前 task 是否有效
        # 2. (task_counts * 0.2)：task 关联实体数量的缩放值
        # 3. 常数 bias = 1
        #
        # 最终 task_feats 维度从 entity_dim 变成 entity_dim + 3
        task_feats = th.cat(
            [
                task_feats,                        # task 聚合后的实体特征
                task_valid.unsqueeze(-1),          # 有效性标记
                (task_counts * 0.2).unsqueeze(-1), # task 实体数的缩放值
                th.ones_like(task_valid.unsqueeze(-1)),  # 常数偏置项
            ],
            dim=-1,  # 在最后一维拼接
        )

        # 所有实体的“是否活跃”标记
        active_entities = 1.0 - entity_mask

        # 当前有效实体总数
        # keepdim=True 保留维度，方便后面广播
        global_denom = active_entities.sum(dim=1, keepdim=True).clamp_min(1.0)

        # 对所有有效实体做平均池化，得到全局状态摘要
        # 这是一个 very simple 的 global pooling
        global_state = (entities * active_entities.unsqueeze(-1)).sum(dim=1) / global_denom

        # 再额外拼接一个“当前有效实体比例/规模”特征
        # global_denom / entities.shape[1] 表示当前有效实体占总实体数的比例
        global_state = th.cat([global_state, global_denom / entities.shape[1]], dim=-1)

        # 返回 critic 所需的两类输入特征
        return global_state, task_feats

    def forward(self, batch, override_alloc=None, test_mode=False, calc_stats=False):
        """
        critic 前向计算。

        输入：
        - batch: 当前高层决策点的 batch
        - override_alloc: 仅 allocation-conditioned 模式使用；state-only
                          模式会完全忽略它
        - test_mode: 这里实际上没用到，只是接口兼容
        - calc_stats: 是否额外返回中间统计量

        输出：
        - value: state-only 模式为 V(s)，旧模式为 C(s,Z)，形状 (bs, 1)
        """

        # 先构造 critic 的状态特征
        global_state, task_feats = self._build_features(batch)

        parts = [global_state, task_feats.flatten(1)]
        if self.critic_condition_on_alloc:
            if override_alloc is None:
                alloc = 1 - batch["entity2task_mask"][:, :self.n_agents].float()
            else:
                alloc = override_alloc
            parts.append(alloc.flatten(1))

        # In state-only mode override_alloc and the agents' allocation rows in
        # entity2task_mask are deliberately not read by the critic.
        x = th.cat(parts, dim=-1)

        # 用 MLP 输出 value
        # 结果形状是 (bs, 1)
        value = self.net(x)

        # 如果需要额外统计量，就一起返回
        if calc_stats:
            return value, {}

        # 否则只返回 value
        return value
