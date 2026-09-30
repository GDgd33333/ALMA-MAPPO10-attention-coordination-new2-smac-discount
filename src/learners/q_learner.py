import copy
from components.episode_buffer import EpisodeBatch
from functools import partial
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.flex_qmix import FlexQMixer, LinearFlexQMixer
from components.action_selectors import parse_avail_actions
import torch as th
import torch.nn.functional as F
import torch.distributions as D
from torch.optim import RMSprop, Adam


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.use_copa = self.args.hier_agent['copa']

        self.params = list(mac.parameters())
        if self.use_copa:
            self.params += list(self.mac.coach.parameters())
            if self.args.hier_agent['copa_vi_loss']:
                self.params += list(self.mac.copa_recog.parameters())

        self.last_target_update_episode = 0
        self.last_alloc_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "flex_qmix":
                assert args.entity_scheme, "FlexQMixer only available with entity scheme"
                self.mixer = FlexQMixer(args)
            elif args.mixer == "lin_flex_qmix":
                assert args.entity_scheme, "FlexQMixer only available with entity scheme"
                self.mixer = LinearFlexQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        low_lr = args.lr
        self.optimiser = RMSprop(params=self.params, lr=low_lr, alpha=args.optim_alpha, eps=args.optim_eps,
                                 weight_decay=args.weight_decay)

        alloc_pi_lr = None
        alloc_q_lr = None
        if self.args.hier_agent["task_allocation"] in ["aql", "ppo", "mappo"]:
            alloc_pi_lr = self.args.hier_agent.get("alloc_pi_lr", None)
            alloc_q_lr = self.args.hier_agent.get("alloc_q_lr", None)
            if alloc_pi_lr is None:
                alloc_pi_lr = args.lr
            else:
                alloc_pi_lr = float(alloc_pi_lr)
            if alloc_q_lr is None:
                alloc_q_lr = args.lr
            else:
                alloc_q_lr = float(alloc_q_lr)

            self.alloc_pi_params = list(mac.alloc_pi_params())
            if self.args.hier_agent["alloc_opt"] == "rmsprop":
                OptClass = partial(RMSprop, alpha=args.optim_alpha)
            elif self.args.hier_agent["alloc_opt"] == "adam":
                OptClass = Adam
            else:
                raise Exception("Optimizer not recognized")
            self.alloc_pi_optimiser = OptClass(
                params=self.alloc_pi_params, lr=alloc_pi_lr, eps=args.optim_eps,
                weight_decay=args.weight_decay)
            self.alloc_q_params = list(mac.alloc_q_params())
            self.alloc_q_optimiser = OptClass(
                params=self.alloc_q_params, lr=alloc_q_lr, eps=args.optim_eps,
                weight_decay=args.alloc_q_weight_decay)

            if self.args.hier_agent["task_allocation"] in ["ppo", "mappo"]:
                critic_mode = (
                    "allocation-conditioned C(s,Z)"
                    if self.mac.alloc_critic.critic_condition_on_alloc
                    else "state-only V(s)"
                )
                self.logger.console_logger.info(
                    "High-level critic mode: {} (input_dim={})".format(
                        critic_mode, self.mac.alloc_critic.critic_in_dim
                    )
                )

        # Print LR config once at startup to verify decoupled settings.
        self.logger.console_logger.info("[LR CONFIG]")
        self.logger.console_logger.info("low-level lr = {}".format(low_lr))
        if self.args.hier_agent["task_allocation"] in ["aql", "ppo", "mappo"]:
            self.logger.console_logger.info("alloc_pi_lr = {}".format(alloc_pi_lr))
            self.logger.console_logger.info("alloc_q_lr = {}".format(alloc_q_lr))

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.log_alloc_stats_t = -self.args.learner_log_interval - 1
        self._logged_high_adv_cfg = False

    def _build_highlevel_discounted_returns(self, seg_rewards, segment_discounts, decision_points, seg_terminated):
        returns = th.zeros_like(seg_rewards)
        running_return = th.zeros_like(seg_rewards[:, 0])
        for t in reversed(range(seg_rewards.shape[1])):
            is_decision = decision_points[:, t].float()
            running_return = running_return * (1 - seg_terminated[:, t])
            curr_return = seg_rewards[:, t] + segment_discounts[:, t] * running_return
            running_return = is_decision * curr_return + (1 - is_decision) * running_return
            returns[:, t] = running_return * is_decision
        return returns

    def _build_highlevel_gae(self, seg_rewards, seg_values, segment_discounts, decision_points, seg_terminated, gae_lambda):
        advantages = th.zeros_like(seg_rewards)
        returns = th.zeros_like(seg_rewards)
        running_adv = th.zeros_like(seg_rewards[:, 0])
        next_value = th.zeros_like(seg_rewards[:, 0])
        for t in reversed(range(seg_rewards.shape[1])):
            is_decision = decision_points[:, t].float()
            v_t = seg_values[:, t]
            duration_discount = segment_discounts[:, t]
            delta_t = seg_rewards[:, t] + duration_discount * next_value * (1 - seg_terminated[:, t]) - v_t
            curr_adv = delta_t + duration_discount * gae_lambda * (1 - seg_terminated[:, t]) * running_adv
            running_adv = is_decision * curr_adv + (1 - is_decision) * running_adv
            advantages[:, t] = running_adv * is_decision
            returns[:, t] = (advantages[:, t] + v_t) * is_decision
            next_value = is_decision * v_t + (1 - is_decision) * next_value
        return advantages, returns

    def _get_mixer_ins(self, batch):
        if not self.args.entity_scheme:
            return (batch["state"][:, :-1],
                    batch["state"][:, 1:])
        else:
            entities = []
            bs, max_t, ne, ed = batch["entities"].shape
            entities.append(batch["entities"])
            if self.args.entity_last_action:
                last_actions = th.zeros(bs, max_t, ne, self.args.n_actions,
                                        device=batch.device,
                                        dtype=batch["entities"].dtype)
                last_actions[:, 1:, :self.args.n_agents] = batch["actions_onehot"][:, :-1]
                entities.append(last_actions)

            entities = th.cat(entities, dim=3)
            mix_ins = {"entities": entities[:, :-1],
                       "entity_mask": batch["entity_mask"][:, :-1]}
            targ_mix_ins = {"entities": entities[:, 1:],
                            "entity_mask": batch["entity_mask"][:, 1:]}
            if self.args.multi_task:
                # use same subtask assignments for prediction and target
                mix_ins["entity2task_mask"] = batch["entity2task_mask"][:, :-1]
                targ_mix_ins["entity2task_mask"] = batch["entity2task_mask"][:, :-1]
            return mix_ins, targ_mix_ins

    def _make_meta_batch(self, batch: EpisodeBatch):
        reward = batch['reward']
        terminated = batch['terminated'].float()
        reset = batch['reset'].float()
        mask = batch['filled'].float()
        allocs = 1 - batch['entity2task_mask'][:, :, :self.args.n_agents].float()
        mask[:, 1:] = mask[:, 1:] * (1 - reset[:, :-1])
        # Preserve the valid low-level transitions before timeout masking.
        transition_mask = mask.clone()
        bs, ts, _ = mask.shape
        t_added = batch['t_added'].reshape(bs, 1, 1).repeat(1, ts, 1)

        timeout = reset - terminated

        decision_points = batch['hier_decision'].float()

        seg_rewards = th.zeros_like(reward)
        cuml_rewards = th.zeros_like(reward[:, 0])
        seg_lengths = th.zeros_like(reward)
        cuml_lengths = th.zeros_like(reward[:, 0])
        seg_terminated = th.zeros_like(terminated)
        cuml_terminated = th.zeros_like(terminated[:, 0])
        cuml_timeout = th.zeros_like(timeout[:, 0])
        # gamma_high is the per-environment-step discount used by the
        # high-level SMDP return. A segment of length L bootstraps with
        # gamma_high ** L rather than one fixed discount per segment.
        gamma_high = float(self.args.hier_agent.get("gamma_high", 0.99))

        for t in reversed(range(reward.shape[1])):
            # Discount rewards inside each high-level segment:
            # r_t + gamma_high*r_{t+1} + ... + gamma_high**(L-1)*r_{t+L-1}.
            cuml_rewards = (
                reward[:, t] * transition_mask[:, t]
                + gamma_high * cuml_rewards
            )
            cuml_lengths += transition_mask[:, t]
            seg_rewards[:, t] = cuml_rewards
            seg_lengths[:, t] = cuml_lengths
            cuml_rewards *= 1 - decision_points[:, t]
            cuml_lengths *= 1 - decision_points[:, t]

            # track whether env terminated between decision points
            cuml_terminated = cuml_terminated.max(terminated[:, t])
            seg_terminated[:, t] = cuml_terminated
            cuml_terminated *= 1 - decision_points[:, t]

            # mask out decision point if a env timeout happens (since we can't bootstrap from next decision point)
            cuml_timeout = cuml_timeout.max(timeout[:, t])
            mask[:, t] *= (1 - cuml_timeout)
            cuml_timeout *= 1 - decision_points[:, t]

        # Duration-aware bootstrap discount for the SMDP transition.
        segment_discounts = th.pow(
            seg_rewards.new_tensor(gamma_high), seg_lengths
        )
        seg_advantages = None
        seg_returns = seg_rewards.clone()
        use_gae = self.args.hier_agent.get("use_gae", False)
        use_discounted_return = self.args.hier_agent.get("use_discounted_return", False)
        if use_gae and 'alloc_value' in batch:
            gae_lambda = float(self.args.hier_agent.get("gae_lambda", 0.95))
            seg_values = batch['alloc_value'].clone()
            seg_advantages, seg_returns = self._build_highlevel_gae(
                seg_rewards=seg_rewards,
                seg_values=seg_values,
                segment_discounts=segment_discounts,
                decision_points=decision_points,
                seg_terminated=seg_terminated,
                gae_lambda=gae_lambda,
            )
        elif use_discounted_return:
            seg_returns = self._build_highlevel_discounted_returns(
                seg_rewards=seg_rewards,
                segment_discounts=segment_discounts,
                decision_points=decision_points,
                seg_terminated=seg_terminated,
            )

        last_alloc = th.zeros_like(allocs)
        was_reset = th.zeros_like(reset[:, [0]])
        for t in range(1, reward.shape[1]):
            # make sure that last_alloc doesn't copy final assignment from previous episode
            was_reset = (was_reset + reset[:, [t - 1]]).min(th.ones_like(was_reset))
            last_alloc[:, t] = allocs[:, t - 1] * (1 - was_reset)
            was_reset *= (1 - decision_points[:, [t]])

        # mask out last decision point in each trajectory if not terminal state (since we can't bootstrap)
        bs, ts, _ = decision_points.shape
        last_dp_ind = (
            decision_points * th.arange(
                ts, dtype=decision_points.dtype,
                device=decision_points.device).reshape(1, ts, 1)
        ).squeeze().argmax(dim=1)
        mask[th.arange(bs), last_dp_ind] *= seg_terminated[th.arange(bs), last_dp_ind]

        entity2task_mask = batch['entity2task_mask'].clone()

        d_inds = (decision_points == 1).reshape(bs, ts)
        max_bs = self.args.hier_agent['max_bs']
        meta_batch = {
            'reward': seg_rewards[d_inds][:max_bs],
            'segment_length': seg_lengths[d_inds][:max_bs],
            'segment_discount': segment_discounts[d_inds][:max_bs],
            'return': seg_returns[d_inds][:max_bs],
            'terminated': seg_terminated[d_inds][:max_bs],
            'mask': mask[d_inds][:max_bs],
            'entities': batch['entities'][d_inds][:max_bs],
            'obs_mask': batch['obs_mask'][d_inds][:max_bs],
            'entity_mask': batch['entity_mask'][d_inds][:max_bs],
            'entity2task_mask': entity2task_mask[d_inds][:max_bs],
            'task_mask': batch['task_mask'][d_inds][:max_bs],
            'avail_actions': batch['avail_actions'][d_inds][:max_bs],
            'last_alloc': last_alloc[d_inds][:max_bs],
            't_added': t_added[d_inds][:max_bs],
        }
        if 'alloc_logprob' in batch:
            meta_batch['alloc_logprob'] = batch['alloc_logprob'][d_inds][:max_bs]
        if 'alloc_actions' in batch:
            meta_batch['alloc_actions'] = batch['alloc_actions'][d_inds][:max_bs]
        if 'alloc_entropy' in batch:
            meta_batch['alloc_entropy'] = batch['alloc_entropy'][d_inds][:max_bs]
        if 'alloc_agent_mask' in batch:
            meta_batch['alloc_agent_mask'] = batch['alloc_agent_mask'][d_inds][:max_bs]
        if 'alloc_value' in batch:
            meta_batch['alloc_value'] = batch['alloc_value'][d_inds][:max_bs]
        if seg_advantages is not None:
            meta_batch['advantage'] = seg_advantages[d_inds][:max_bs]
        return meta_batch

    def alloc_train_ppo(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """
        高层 PPO allocator 的训练函数。

        作用：
        1. 从完整 episode batch 中抽取“高层决策点”对应的 meta_batch
        2. 读取 rollout 时存下来的旧 log_prob、旧 value、旧 allocation
        3. 用当前策略重新评估这些旧动作，得到 new_log_prob / entropy / value_pred
        4. 按 PPO 的 clipped objective 更新高层 actor
        5. 用 MSE 更新高层 critic
        6. 记录训练统计量

        参数：
        - batch: 一个完整的 episode batch（通常包含低层每一步的数据）
        - t_env: 当前全局训练步数，用于日志记录
        - episode_num: 当前 episode 编号（这里没有直接用，但接口保留）
        """

        # ------------------------------------------
        # 第一步：从完整的 low-level batch 中提取高层决策点对应的数据
        # ------------------------------------------
        # 这个函数通常会把“每 K 步一次”的高层决策点筛出来，
        # 形成一个只用于高层训练的 meta_batch。
        meta_batch = self._make_meta_batch(batch)

        # 如果当前没有任何高层决策样本，就直接返回空字典
        # 这种情况可能发生在：episode 太短、没有触发高层决策点等
        if meta_batch["reward"].shape[0] == 0:
            return {}

        # ------------------------------------------
        # 第二步：取出高层 PPO 训练所需的基本字段
        # ------------------------------------------

        # 高层 reward，通常是“两个高层决策点之间累计的环境奖励”
        # 形状一般是 (n_meta_steps, 1) 或 (bs_meta, 1)
        rewards = meta_batch["reward"]

        # mask 用于表示哪些高层决策样本是有效的
        # 比如 episode 结束后的 padding 样本通常要 mask 掉
        mask = meta_batch["mask"]

        # 把 mask 扩展成和 rewards 一样的形状，方便后面逐元素乘
        # 例如 rewards 是 (N, 1)，那 active_mask 也会是 (N, 1)
        active_mask = mask.expand_as(rewards)

        use_gae = self.args.hier_agent.get("use_gae", False)
        use_discounted_return = self.args.hier_agent.get("use_discounted_return", False)
        gamma_high = float(self.args.hier_agent.get("gamma_high", 0.99))
        gae_lambda = float(self.args.hier_agent.get("gae_lambda", 0.95))
        if not self._logged_high_adv_cfg:
            self.logger.console_logger.info(
                "[HIGH PPO ADV CONFIG] use_gae={} use_discounted_return={} gamma_high_per_env_step={} gae_lambda={}".format(
                    use_gae, use_discounted_return, gamma_high, gae_lambda
                )
            )
            self._logged_high_adv_cfg = True
        returns = meta_batch.get("return", rewards).detach()

        # ------------------------------------------
        # 第三步：读取 rollout 时行为策略留下来的旧统计量
        # ------------------------------------------

        # old_log_prob：
        # rollout 时，高层行为策略（旧策略）对那次 allocation 的总 log_prob
        # detach 的目的是：训练当前策略时，不让梯度回到旧 rollout 图里
        old_log_prob = meta_batch["alloc_logprob"].detach()

        # old_value：
        # rollout 时，critic 对那次高层决策给出的 value 预测
        # 同样 detach，作为旧基线使用
        old_value = meta_batch["alloc_value"].detach()

        # ------------------------------------------
        # 第四步：构造 advantage
        # ------------------------------------------

        # 目前 advantage 的简化写法：
        # advantage = return - old_value
        #
        # 含义：
        # - 如果实际回报比旧 value 高，说明这次分配比预期好，advantage 为正
        # - 如果实际回报比旧 value 低，说明这次分配比预期差，advantage 为负
        if use_gae and "advantage" in meta_batch:
            advantages = meta_batch["advantage"].detach()
        else:
            advantages = returns - old_value

        # advantage 的 mask，和 active_mask 形状对齐
        adv_mask = active_mask.expand_as(advantages)

        # 计算被 mask 后的 advantage 均值
        # 只在有效样本上统计
        adv_mean = (advantages * adv_mask).sum() / adv_mask.sum().clamp_min(1.0)

        # 计算被 mask 后的 advantage 方差
        # 也是只在有效样本上统计
        adv_var = (((advantages - adv_mean) * adv_mask) ** 2).sum() / adv_mask.sum().clamp_min(1.0)

        # 对 advantage 做标准化
        # 这是 PPO 中很常见的做法，有助于训练稳定
        advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)

        # ------------------------------------------
        # 第五步：从 allocation 恢复 action_seq
        # ------------------------------------------

        # meta_batch["entity2task_mask"] 里前 n_agents 行对应 agent 到 task 的 one-hot/反mask关系
        # 原始 mask 语义通常是：1 表示“不属于 task”，0 表示“属于 task”
        # 所以这里取反后得到 one-hot allocation
        alloc_onehot = 1 - meta_batch["entity2task_mask"][:, :self.args.n_agents].float()

        # action_seq 是整数形式的 task id 序列
        # 例如 one-hot [0,0,1,0] -> argmax 后变成 task id = 2
        #
        # 之所以需要 action_seq，是因为 evaluate_actions() 里要逐 agent 回放旧动作
        action_seq = alloc_onehot.argmax(dim=-1)

        # ------------------------------------------
        # 第六步：读取 PPO 的训练超参数
        # ------------------------------------------

        # PPO clip 范围，默认 0.2
        eps_clip = self.args.hier_agent.get("ppo_clip", 0.2)

        # critic loss 的权重系数
        value_coef = self.args.hier_agent.get("ppo_value_coef", 0.5)

        # entropy 正则项的权重系数
        entropy_coef = self.args.hier_agent.get("ppo_entropy_coef", 0.01)

        # 每次高层 PPO 更新要做多少个 epoch
        ppo_epochs = int(self.args.hier_agent.get("ppo_epochs", 4))

        # 训练统计量字典，后面会往里面写 loss、KL、clip fraction 等
        stats = {}

        # ------------------------------------------
        # 第七步：进行多轮 PPO epoch 更新
        # ------------------------------------------
        for _ in range(ppo_epochs):

            # 用当前策略重新评估 rollout 时执行过的旧 action_seq
            # evaluate_actions 的作用是：
            # - 不重新采样动作
            # - 而是按旧 action_seq 回放
            # - 算出当前策略下这些旧动作的 new_log_prob 和 entropy
            pi_eval = self.mac.alloc_policy.evaluate_actions(meta_batch, action_seq)

            # 当前策略对整张 allocation 的总 log_prob
            # 用于和 old_log_prob 做 ratio
            new_log_prob = pi_eval["log_prob"]

            # 当前策略下的 entropy（通常是整张 allocation 序列的总 entropy）
            entropy = pi_eval["entropy"]

            # State-only V(s) is the default. The allocation is supplied only
            # for controlled runs using the legacy allocation-conditioned C(s, Z).
            if self.mac.alloc_critic.critic_condition_on_alloc:
                value_pred = self.mac.alloc_critic(meta_batch, override_alloc=alloc_onehot)
            else:
                value_pred = self.mac.alloc_critic(meta_batch)

            # --------------------------------------
            # 第八步：PPO actor 部分
            # --------------------------------------

            # PPO 核心比值：
            # ratio = π_new(a|s) / π_old(a|s)
            # 因为存的是 log_prob，所以用 exp(new - old)
            ratio = th.exp(new_log_prob - old_log_prob)

            # PPO surrogate objective 第一项
            # 如果 ratio 不大，就直接按 ratio * advantage 来更新
            surr1 = ratio * advantages

            # PPO surrogate objective 第二项
            # 把 ratio 限制在 [1-eps, 1+eps] 区间内
            # 防止策略更新太猛
            surr2 = th.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages

            # actor loss 的逐样本形式
            # PPO 取 min(surr1, surr2)，再取负号做梯度下降
            actor_loss_t = -th.min(surr1, surr2)

            # 只在有效样本上平均 actor loss
            actor_loss = (actor_loss_t * active_mask).sum() / active_mask.sum().clamp_min(1.0)

            # --------------------------------------
            # 第九步：critic 部分
            # --------------------------------------

            # critic 的 MSE 误差
            # 当前实现里 target 就是 returns（简化版本）
            value_err = (value_pred - returns) ** 2

            # 同样只在有效样本上平均
            critic_loss = (value_err * active_mask).sum() / active_mask.sum().clamp_min(1.0)

            # --------------------------------------
            # 第十步：entropy 正则
            # --------------------------------------

            # entropy 也只在有效样本上平均
            entropy_mean = (entropy * active_mask).sum() / active_mask.sum().clamp_min(1.0)

            # 高层 actor 的总损失
            # = actor_loss - entropy_coef * entropy
            # 这里减 entropy 是因为我们希望在优化时鼓励更高熵（保持探索）
            pi_loss = actor_loss - entropy_coef * entropy_mean

            # critic 的总损失
            # = value_coef * critic_loss
            q_loss = value_coef * critic_loss

            # --------------------------------------
            # 第十一步：更新高层 actor 参数
            # --------------------------------------

            # 清空 actor 优化器里的旧梯度
            self.alloc_pi_optimiser.zero_grad()

            # 反向传播 actor loss
            pi_loss.backward()

            # 做梯度裁剪，防止梯度爆炸
            # 返回值是裁剪前的梯度范数
            pi_grad_norm = th.nn.utils.clip_grad_norm_(self.alloc_pi_params, self.args.grad_norm_clip)

            # actor 参数更新一步
            self.alloc_pi_optimiser.step()

            # --------------------------------------
            # 第十二步：更新高层 critic 参数
            # --------------------------------------

            # 清空 critic 优化器里的旧梯度
            self.alloc_q_optimiser.zero_grad()

            # 反向传播 critic loss
            q_loss.backward()

            # critic 梯度裁剪
            q_grad_norm = th.nn.utils.clip_grad_norm_(self.alloc_q_params, self.args.grad_norm_clip)

            # critic 参数更新一步
            self.alloc_q_optimiser.step()

        # ------------------------------------------
        # 第十三步：在 no_grad 下统计一些训练诊断量
        # ------------------------------------------
        with th.no_grad():

            # 近似 KL：
            # 这里用 (old_log_prob - new_log_prob) 的平均来做一个简化近似
            # 用于监控当前策略相对旧策略偏移了多少
            approx_kl = ((old_log_prob - new_log_prob) * active_mask).sum() / active_mask.sum().clamp_min(1.0)

            # clip fraction：
            # 看有多少样本的 ratio 超出了 clip 区间
            # 这是 PPO 一个很常用的训练健康度指标
            clip_frac = ((th.abs(ratio - 1.0) > eps_clip).float() * active_mask).sum() / active_mask.sum().clamp_min(1.0)

            # 记录 actor loss
            stats["losses/alloc_ppo_actor"] = actor_loss.item()

            # 记录 critic loss
            stats["losses/alloc_ppo_critic"] = critic_loss.item()

            # 记录平均 entropy
            stats["losses/alloc_ppo_entropy"] = entropy_mean.item()
            stats["alloc_metrics/return_target_mean"] = ((returns * active_mask).sum() / active_mask.sum().clamp_min(1.0)).item()

            # 记录近似 KL
            stats["alloc_metrics/alloc_kl"] = approx_kl.item()

            # 记录 clip 比例
            stats["alloc_metrics/alloc_clip_frac"] = clip_frac.item()

            # 记录 actor 梯度范数
            stats["train_metrics/alloc_pi_grad_norm"] = pi_grad_norm

            # 记录 critic 梯度范数
            stats["train_metrics/alloc_q_grad_norm"] = q_grad_norm

        # ------------------------------------------
        # 第十四步：按日志间隔写日志
        # ------------------------------------------

        # 只有当距离上次记录已经超过 learner_log_interval 时，才写日志
        if t_env - self.log_alloc_stats_t >= self.args.learner_log_interval:

            # 把 stats 里的每一项写入 logger
            for name, value in stats.items():
                self.logger.log_stat(name, value, t_env)

            # 更新上次记录日志的时间戳
            self.log_alloc_stats_t = t_env

        # 返回本次 PPO 更新的统计结果
        return stats

    def alloc_train_mappo(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """Train MAPPO-style high-level allocator with per-agent PPO ratios."""
        meta_batch = self._make_meta_batch(batch)
        if meta_batch["reward"].shape[0] == 0:
            return {}

        rewards = meta_batch["reward"]
        mask = meta_batch["mask"]
        active_mask = mask.expand_as(rewards)
        returns = meta_batch.get("return", rewards).detach()

        use_gae = self.args.hier_agent.get("use_gae", False)
        use_discounted_return = self.args.hier_agent.get("use_discounted_return", False)
        gamma_high = float(self.args.hier_agent.get("gamma_high", 0.99))
        gae_lambda = float(self.args.hier_agent.get("gae_lambda", 0.95))
        if not self._logged_high_adv_cfg:
            self.logger.console_logger.info(
                "[HIGH MAPPO ADV CONFIG] use_gae={} use_discounted_return={} gamma_high_per_env_step={} gae_lambda={}".format(
                    use_gae, use_discounted_return, gamma_high, gae_lambda
                )
            )
            self._logged_high_adv_cfg = True

        old_log_probs = meta_batch["alloc_logprob"].detach()
        old_value = meta_batch["alloc_value"].detach()
        if use_gae and "advantage" in meta_batch:
            advantages = meta_batch["advantage"].detach()
        else:
            advantages = returns - old_value

        adv_mask = active_mask.expand_as(advantages)
        adv_mean = (advantages * adv_mask).sum() / adv_mask.sum().clamp_min(1.0)
        adv_var = (((advantages - adv_mean) * adv_mask) ** 2).sum() / adv_mask.sum().clamp_min(1.0)
        if self.args.hier_agent.get("normalize_advantages", True):
            advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)

        alloc_actions = meta_batch["alloc_actions"].long()
        if alloc_actions.dim() == 2:
            alloc_actions = alloc_actions.unsqueeze(-1)
        # Reconstruct all fixed agent slots, then remove agents that were
        # already inactive when this high-level decision was made.
        alloc_onehot_unmasked = F.one_hot(
            alloc_actions.squeeze(-1), num_classes=self.args.n_tasks
        ).float()

        if "alloc_agent_mask" in meta_batch:
            agent_mask = meta_batch["alloc_agent_mask"].float()
        else:
            agent_mask = (1.0 - meta_batch["entity_mask"][:, :self.args.n_agents].float()).unsqueeze(-1)
        valid_agent_mask = agent_mask * active_mask.unsqueeze(1)
        alloc_onehot = alloc_onehot_unmasked * agent_mask

        eps_clip = self.args.hier_agent.get("ppo_clip", 0.2)
        value_coef = self.args.hier_agent.get(
            "ppo_value_coef", self.args.hier_agent.get("value_coef", 0.5))
        entropy_coef = self.args.hier_agent.get("ppo_entropy_coef", 0.01)
        ppo_epochs = int(self.args.hier_agent.get("ppo_epochs", 4))

        stats = {}
        for _ in range(ppo_epochs):
            pi_eval = self.mac.alloc_policy.evaluate_actions(meta_batch, alloc_actions)
            new_log_probs = pi_eval["log_probs"]
            entropy = pi_eval["entropy"]
            if self.mac.alloc_critic.critic_condition_on_alloc:
                value_pred = self.mac.alloc_critic(meta_batch, override_alloc=alloc_onehot)
            else:
                value_pred = self.mac.alloc_critic(meta_batch)

            ratio = th.exp(new_log_probs - old_log_probs)
            adv_agent = advantages.unsqueeze(1).expand_as(ratio)
            surr1 = ratio * adv_agent
            surr2 = th.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv_agent
            actor_loss_per_agent = -th.min(surr1, surr2) * valid_agent_mask

            # Standard MAPPO reduction: average over all valid agent decisions.
            # This keeps the actor scale independent of the active-agent count.
            actor_loss = (
                actor_loss_per_agent.sum()
                / valid_agent_mask.sum().clamp_min(1.0)
            )

            value_err = (value_pred - returns) ** 2
            critic_loss = (value_err * active_mask).sum() / active_mask.sum().clamp_min(1.0)

            entropy_mean = (entropy * valid_agent_mask).sum() / valid_agent_mask.sum().clamp_min(1.0)
            pi_loss = actor_loss - entropy_coef * entropy_mean
            q_loss = value_coef * critic_loss

            self.alloc_pi_optimiser.zero_grad()
            pi_loss.backward()
            pi_grad_norm = th.nn.utils.clip_grad_norm_(self.alloc_pi_params, self.args.grad_norm_clip)
            self.alloc_pi_optimiser.step()

            self.alloc_q_optimiser.zero_grad()
            q_loss.backward()
            q_grad_norm = th.nn.utils.clip_grad_norm_(self.alloc_q_params, self.args.grad_norm_clip)
            self.alloc_q_optimiser.step()

        with th.no_grad():
            approx_kl = ((old_log_probs - new_log_probs) * valid_agent_mask).sum() / valid_agent_mask.sum().clamp_min(1.0)
            clip_frac = ((th.abs(ratio - 1.0) > eps_clip).float() * valid_agent_mask).sum() / valid_agent_mask.sum().clamp_min(1.0)
            task_valid = 1.0 - meta_batch["task_mask"].float()
            alloc_counts = alloc_onehot.sum(dim=1)
            selected_task_mask = (alloc_counts > 0).float() * task_valid
            valid_task_count = task_valid.sum(dim=1).clamp_min(1.0)
            covered = selected_task_mask.sum(dim=1)
            duplicate_tasks = ((alloc_counts > 1).float() * task_valid).sum(dim=1)
            duplicate_denom = selected_task_mask.sum(dim=1).clamp_min(1.0)

            segment_lengths = meta_batch["segment_length"][active_mask.bool()]
            if segment_lengths.numel() > 0:
                max_segment_length = int(self.args.hier_agent["action_length"])
                stats["segment_length/mean"] = segment_lengths.mean().item()
                stats["segment_length/std"] = segment_lengths.std(unbiased=False).item()
                stats["segment_length/min"] = segment_lengths.min().item()
                stats["segment_length/max"] = segment_lengths.max().item()
                stats["segment_length/early_end_fraction"] = (
                    segment_lengths < max_segment_length
                ).float().mean().item()
                for length in range(1, max_segment_length + 1):
                    stats["segment_length/length_{}_fraction".format(length)] = (
                        segment_lengths == length
                    ).float().mean().item()

            stats["losses/alloc_mappo_actor"] = actor_loss.item()
            stats["losses/alloc_mappo_critic"] = critic_loss.item()
            stats["losses/alloc_mappo_entropy"] = entropy_mean.item()
            stats["alloc_metrics/adv_std"] = adv_var.sqrt().item()
            stats["alloc_metrics/return_target_mean"] = ((returns * active_mask).sum() / active_mask.sum().clamp_min(1.0)).item()
            stats["alloc_metrics/value_mean"] = ((value_pred * active_mask).sum() / active_mask.sum().clamp_min(1.0)).item()
            stats["alloc_metrics/alloc_kl"] = approx_kl.item()
            stats["alloc_metrics/alloc_clip_frac"] = clip_frac.item()
            stats["alloc_metrics/ratio_max"] = ratio[valid_agent_mask.bool()].max().item() if valid_agent_mask.bool().any() else 0.0
            stats["alloc_metrics/ratio_min"] = ratio[valid_agent_mask.bool()].min().item() if valid_agent_mask.bool().any() else 0.0
            stats["alloc_metrics/task_coverage_ratio"] = (covered / valid_task_count).mean().item()
            stats["alloc_metrics/duplicate_task_ratio"] = (duplicate_tasks / duplicate_denom).mean().item()
            task_load_mean = (alloc_counts * task_valid).sum(dim=1, keepdim=True) / valid_task_count.unsqueeze(1)
            stats["alloc_metrics/task_load_std"] = (((alloc_counts - task_load_mean) * task_valid) ** 2).sum(dim=1).div(valid_task_count).sqrt().mean().item()
            stats["alloc_debug/active_agent_count"] = (
                agent_mask.squeeze(-1).sum(dim=1).mean().item()
            )
            for key, value in pi_eval.items():
                if key.startswith("iga_"):
                    stats["alloc_iga/{}".format(key)] = value.detach().mean().item()
                elif key.startswith("alloc_counterfactual/"):
                    stats[key] = value.detach().mean().item()
            stats["train_metrics/alloc_pi_grad_norm"] = pi_grad_norm
            stats["train_metrics/alloc_q_grad_norm"] = q_grad_norm

        if t_env - self.log_alloc_stats_t >= self.args.learner_log_interval:
            for name, value in stats.items():
                self.logger.log_stat(name, value, t_env)
            self.log_alloc_stats_t = t_env
        return stats

    def alloc_train_aql(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        meta_batch = self._make_meta_batch(batch)
        rewards = meta_batch['reward']
        terminated = meta_batch['terminated']
        mask = meta_batch['mask']
        stats = {}

        # Compute Q-values (evaluate task allocation stored in entity2task_mask)
        alloc_q, q_stats = self.mac.evaluate_allocation(meta_batch, calc_stats=True)

        # Compute proposal allocations (test_mode=True to remove stochasticity in critic, pass in target_mac for stability in bootstrap targets)
        new_alloc, pi_stats = self.mac.compute_allocation(meta_batch, calc_stats=True, test_mode=True, target_mac=self.target_mac)

        # Compute target Q-values
        target_alloc_q = pi_stats['targ_best_prop_values']
        target_alloc_q = self.target_mac.alloc_critic.denormalize(target_alloc_q)

        # Compute TD-loss (don't bootstrap from next state if previous state is
        # terminal)
        targets = (rewards[:-1] + self.args.gamma * (1 - terminated[:-1]) * target_alloc_q[1:]).detach()
        if self.args.popart:
            targets = self.mac.alloc_critic.popart_update(
                targets, mask[:-1])

        td_error = (alloc_q[:-1] - targets.detach())
        td_mask = mask[:-1].expand_as(td_error)
        if self.args.hier_agent['decay_old'] > 0:
            cutoff = self.args.hier_agent['decay_old']
            ratio = (cutoff - t_env + meta_batch['t_added'][:-1].float()) / cutoff
            ratio = ratio.max(th.zeros_like(ratio))
            td_mask *= ratio
        masked_td_error = td_error * td_mask
        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / td_mask.sum()
        stats['losses/alloc_q_loss'] = td_loss.cpu().item()

        # backprop Q loss
        q_loss = td_loss
        self.alloc_q_optimiser.zero_grad()
        q_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.alloc_q_params, self.args.grad_norm_clip)
        stats['train_metrics/alloc_q_grad_norm'] = grad_norm
        self.alloc_q_optimiser.step()

        # Log allocation metrics
        stats['alloc_metrics/best_prob'] = pi_stats['best_prob'].mean().cpu().item()
        # Compute what % of agents changed their task allocation (if previous alloc exists)
        active_ag = 1 - meta_batch['entity_mask'][:, :self.args.n_agents].float()
        ag_changed = (meta_batch['last_alloc'].argmax(dim=2) != new_alloc.detach().argmax(dim=2)).float()
        prev_al_exists = (meta_batch['last_alloc'].sum(dim=(1, 2)) >= 1).float()
        perc_changed_per_step = ((ag_changed * active_ag).sum(dim=1) / active_ag.sum(dim=1))
        perc_changed = (perc_changed_per_step * prev_al_exists).sum() / prev_al_exists.sum()
        stats['alloc_metrics/perc_ag_changed'] = perc_changed.cpu().item()
        # Measure abs value of difference between # of agents and # of entities
        # in each subtask (may not be useful for all tasks)
        nonagent2task = 1 - meta_batch['entity2task_mask'][:, self.args.n_agents:].float()
        ag_per_task = new_alloc.detach().sum(dim=1)
        nag_per_task = nonagent2task.sum(dim=1)
        absdiff_per_task = (ag_per_task - nag_per_task).abs()
        abs_diff_mean = absdiff_per_task.sum(dim=1) / (1 - meta_batch['task_mask'].float()).sum(dim=1)
        stats['alloc_metrics/ag_task_concentration'] = abs_diff_mean.mean().cpu().item()

        # Maximize probability of best allocation
        all_prop_log_pi = pi_stats['log_pi']  # log_pi of all sampled proposal actions
        bs = all_prop_log_pi.shape[0]
        best_prop_log_pi = all_prop_log_pi[th.arange(bs), pi_stats['best_prop_inds']]
        amort_step_loss = -best_prop_log_pi
        masked_amort_step_loss = amort_step_loss * mask
        amort_loss = masked_amort_step_loss.sum() / mask.sum()
        stats['losses/alloc_amort_loss'] = amort_loss.cpu().item()


        active_task = 1 - meta_batch['task_mask'].float().unsqueeze(1)
        ag2task = pi_stats['all_allocs'].detach()  # (bs, n_prop, na, nt)
        task_has_agents = (ag2task.sum(dim=2) > 0).float()
        any_task_no_agents = (task_has_agents.sum(dim=2, keepdim=True)
                              != active_task.sum(dim=2, keepdim=True)).float()
        stats['alloc_metrics/any_task_no_agents_pi'] = any_task_no_agents.mean().cpu().item()

        # entropy term
        entropy = pi_stats['entropy']
        entropy_loss = -entropy.mean()
        stats['losses/alloc_entropy'] = -entropy_loss.cpu().item()

        pi_loss = (amort_loss
                   + self.args.hier_agent['entropy_loss'] * entropy_loss)

        # backprop policy loss
        self.alloc_pi_optimiser.zero_grad()
        pi_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.alloc_pi_params, self.args.grad_norm_clip)
        stats['train_metrics/alloc_pi_grad_norm'] = grad_norm
        self.alloc_pi_optimiser.step()

        if (episode_num - self.last_alloc_target_update_episode) / self.args.alloc_target_update_interval >= 1.0:
            self._update_alloc_targets()
            self.last_alloc_target_update_episode = episode_num

        if t_env - self.log_alloc_stats_t >= self.args.learner_log_interval:
            for name, value in stats.items():
                self.logger.log_stat(name, value, t_env)
            self.log_alloc_stats_t = t_env

        return stats, new_alloc

    def _broadcast_decisions_to_batch(self, decisions, decision_pts):
        decision_pts = decision_pts.squeeze(-1)
        bs, ts = decision_pts.shape
        bcast_decisions = {k: th.zeros_like(v[[0]]).unsqueeze(0).repeat(bs * rep, ts, *(1 for _ in range(len(v.shape) - 1))) for k, (v, rep) in decisions.items()}
        for decname in bcast_decisions:
            value, rep = decisions[decname]
            bcast_decisions[decname][decision_pts.repeat(rep, 1)] = value
        for t in range(1, ts):
            for decname in bcast_decisions:
                rep = decisions[decname][1]
                prev_value = bcast_decisions[decname][:, t - 1]
                bcast_decisions[decname][:, t] = ((decision_pts[:, t].repeat(rep).reshape(bs * rep, 1, 1).to(prev_value.dtype) * bcast_decisions[decname][:, t])
                                                  + ((1 - decision_pts[:, t].repeat(rep).reshape(bs * rep, 1, 1)).to(prev_value.dtype) * prev_value))
        return bcast_decisions

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        # episode over (not including timeout) - determines when to bootstrap
        terminated = batch["terminated"][:, :-1].float()
        # env reset (either terminated or timed out) - determines what timesteps
        # to learn from - we can't learn from final ts bc there is no
        # transition
        reset = batch["reset"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - reset[:, :-1])
        org_mask = mask.clone()
        avail_actions = batch["avail_actions"]
        if self.args.agent['subtask_cond'] is not None:
            # Learning separate controllers for each task
            rewards = batch['task_rewards'][:, :-1]
            terminated = batch['tasks_terminated'][:, :-1].float()
            mask = mask.repeat(1, 1, self.args.n_tasks)
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            task_has_agents = (1 - batch['entity2task_mask'][:, :-1, :self.args.n_agents]).sum(2) > 0
            mask *= task_has_agents.float()

        # # Calculate estimated Q-Values
        # mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # enable things like dropout on mac and mixer, but not target_mac and target_mixer
        self.mac.train()
        self.target_mac.eval()
        if self.mixer is not None:
            self.mixer.train()
            self.target_mixer.eval()

        coach_h = None
        targ_coach_h = None
        coach_z = None
        targ_coach_z = None

        imagine_inps = None
        if self.args.agent['imagine']:
            imagine_inps, imagine_groups = self.mac.agent.make_imagined_inputs(batch)
        if self.use_copa:
            coach_h = self.mac.coach.encode(batch, imagine_inps=imagine_inps)
            targ_coach_h = self.target_mac.coach.encode(batch)
            decision_points = batch['hier_decision'].squeeze(-1)
            bs_rep = 1
            if self.args.agent['imagine']:
                bs_rep = 3
            coach_h_t0 = coach_h[decision_points.repeat(bs_rep, 1)]
            targ_coach_h_t0 = targ_coach_h[decision_points]
            coach_z_t0, coach_mu_t0, coach_logvar_t0 = self.mac.coach.strategy(coach_h_t0)
            coach_mu_t0 = coach_mu_t0.chunk(bs_rep, dim=0)[0]
            coach_logvar_t0 = coach_logvar_t0.chunk(bs_rep, dim=0)[0]
            targ_coach_z_t0, _, _ = self.target_mac.coach.strategy(targ_coach_h_t0)

            bcast_ins = {
                'coach_z_t0': (coach_z_t0, bs_rep),
                'coach_mu_t0': (coach_mu_t0, 1),
                'coach_logvar_t0': (coach_logvar_t0, 1),
                'targ_coach_z_t0': (targ_coach_z_t0, 1),
            }
            bcast_decisions = self._broadcast_decisions_to_batch(bcast_ins, batch['hier_decision'])
            coach_z = bcast_decisions['coach_z_t0']
            coach_mu = bcast_decisions['coach_mu_t0']
            coach_logvar = bcast_decisions['coach_logvar_t0']
            targ_coach_z = bcast_decisions['targ_coach_z_t0']


        batch_mult = 1
        if self.args.agent['imagine']:
            batch_mult += 2

        all_mac_out, mac_info = self.mac.forward(
            batch, t=None,
            coach_z=coach_z,
            imagine_inps=imagine_inps)
        rep_actions = actions.repeat(batch_mult, 1, 1, 1)
        all_chosen_action_qvals = th.gather(all_mac_out[:, :-1], dim=3, index=rep_actions).squeeze(3)  # Remove the last dim

        mac_out_tup = all_mac_out.chunk(batch_mult, dim=0)
        caq_tup = all_chosen_action_qvals.chunk(batch_mult, dim=0)

        mac_out = mac_out_tup[0]
        chosen_action_qvals = caq_tup[0]
        if self.args.agent['imagine']:
            caq_imagine = th.cat(caq_tup[1:], dim=2)

        self.target_mac.init_hidden(batch.batch_size)

        target_mac_out, _ = self.target_mac.forward(batch, coach_z=targ_coach_z, t=None, target=True)
        if self.args.agent['subtask_cond'] is not None:
            allocs = (1 - batch['entity2task_mask'][:, :, :self.args.n_agents])
            avail_actions_targ = parse_avail_actions(avail_actions[:, 1:], allocs[:, :-1], self.args)
        else:
            avail_actions_targ = avail_actions[:, 1:]
        target_mac_out = target_mac_out[:, 1:]

        # Mask out unavailable actions
        target_mac_out[avail_actions_targ == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()[:, 1:]
            mac_out_detach[avail_actions_targ == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            mix_ins, targ_mix_ins = self._get_mixer_ins(batch)

            chosen_action_qvals = self.mixer(chosen_action_qvals, mix_ins)
            gamma = self.args.gamma

            target_max_qvals = self.target_mixer(target_max_qvals, targ_mix_ins)
            target_max_qvals = self.target_mixer.denormalize(target_max_qvals)
            # Calculate 1-step Q-Learning targets
            targets = (rewards + gamma * (1 - terminated) * target_max_qvals).detach()
            if self.args.popart:
                targets = self.mixer.popart_update(
                    targets, mask)

            if self.args.agent['imagine']:
                # don't need last timestep
                imagine_groups = [gr[:, :-1] for gr in imagine_groups]
                caq_imagine = self.mixer(caq_imagine, mix_ins,
                                         imagine_groups=imagine_groups)
        else:
            targets = (rewards + self.args.gamma * (1 - terminated) * target_max_qvals).detach()

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if self.args.agent['imagine']:
            im_prop = self.args.lmbda
            im_td_error = (caq_imagine - targets.detach())
            im_masked_td_error = im_td_error * mask
            im_loss = (im_masked_td_error ** 2).sum() / mask.sum()
            loss = (1 - im_prop) * loss + im_prop * im_loss

        if self.use_copa and self.args.hier_agent['copa_vi_loss']:
            # VI loss
            q_mu, q_logvar = self.mac.copa_recog(batch)
            q_t = D.normal.Normal(q_mu, (0.5 * q_logvar).exp())
            coach_z = coach_z.chunk(bs_rep, dim=0)[0]  # if combining with REFIL, only train full info Z
            log_prob = q_t.log_prob(coach_z).clamp_(-1000, 0).sum(-1)
            # entropy loss
            p_ = D.normal.Normal(coach_mu, (0.5 * coach_logvar).exp())
            entropy = p_.entropy().clamp_(0, 10).sum(-1)

            # mask inactive agents
            agent_mask = 1 - batch['entity_mask'][:, :, :self.args.n_agents].float()
            log_prob = (log_prob * agent_mask).sum(-1) / (agent_mask.sum(-1) + 1e-8)
            entropy = (entropy * agent_mask).sum(-1) / (agent_mask.sum(-1) + 1e-8)

            vi_loss = (-log_prob[:, :-1] * org_mask.squeeze(-1)).sum() / org_mask.sum()
            entropy_loss = (-entropy[:, :-1] * org_mask.squeeze(-1)).sum() / org_mask.sum()
            
            loss += vi_loss * self.args.vi_lambda + entropy_loss * self.args.vi_lambda / 10

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("losses/q_loss", loss.item(), t_env)
            if self.args.agent['imagine']:
                self.logger.log_stat("losses/im_loss", im_loss.item(), t_env)
            if self.use_copa and self.args.hier_agent['copa_vi_loss']:
                self.logger.log_stat("losses/copa_vi_loss", vi_loss.item(), t_env)
                self.logger.log_stat("losses/copa_entropy_loss", entropy_loss.item(), t_env)
            self.logger.log_stat("train_metrics/q_grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("train_metrics/td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("train_metrics/q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("train_metrics/target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _update_alloc_targets(self):
        self.target_mac.load_alloc_state(self.mac)
        self.logger.console_logger.info("Updated allocation target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}opt.th".format(path))

    def load_models(self, path, pi_only=False, evaluate=False):
        self.mac.load_models(path, pi_only=pi_only)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path, pi_only=pi_only)
        if not evaluate and not pi_only:
            if self.mixer is not None:
                self.mixer.load_state_dict(th.load("{}mixer.th".format(path), map_location=lambda storage, loc: storage))
                self.target_mixer.load_state_dict(
                    th.load("{}mixer.th".format(path), map_location=lambda storage, loc: storage)
                )
            self.optimiser.load_state_dict(th.load("{}opt.th".format(path), map_location=lambda storage, loc: storage))
