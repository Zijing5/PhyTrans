import torch
from torch.distributions import Normal
import swanlab
class AWRSearcher:
    def __init__(self, z_dim=16, alpha=1, k_max=500, step_size=1e-3, eps=1e-5, num_env=1, device=None):
        self.z_dim = z_dim
        self.alpha = alpha  # bigger smoother
        self.k_max = k_max
        self.step_size = step_size
        self.D = []  # 存 (z_i, R_i) 元组
        self.omega_mean = torch.zeros(z_dim, device=device, requires_grad=True)
        self.omega_std = torch.ones(z_dim, device=device, requires_grad=True)
        self.omega_dist = Normal(self.omega_mean, self.omega_std)
        self.k_count = 0
        self.device=device
    def init_z_k_sample(self):
        z_k = self.omega_dist.sample().to(self.device)
        return z_k
    def adapt(self, z_k , R_k):  # get_return_fn(z): 用 policy(s,z) 在 env 跑一 episode，返回 R
        # for k in range(self.k_max):
        self.k_count+=1
        gap=1
        if self.k_count%gap == 0: # 每次调用更新
            # 跑 episode，得 R_k
            self.D.append((z_k, R_k))

            normalized = 1
            if normalized==1:
                Rs = torch.tensor([r for _, r in self.D], device=self.device, dtype=torch.float32)
                # 通过SGD更新 ω_{k+1} (假设简单梯度下降，拟合高斯加权似然)
                swanlab.log({"r": R_k})
                if len(Rs) > 1: # 至少有两个样本才能计算标准差
                    # 计算回报的均值和标准差
                    Rs_mean = Rs.mean()
                    Rs_std = Rs.std()
                    
                    # 防止除以零，给标准差一个极小值下限
                    Rs_std = torch.max(Rs_std, torch.tensor(1e-6, device=self.device, dtype=torch.float32))
                    
                    # 归一化回报差异
                    normalized_Rs_diff = (Rs - Rs_mean) / Rs_std
                else: # 如果只有一个样本，无法计算标准差，直接使用0作为差异
                    normalized_Rs_diff = torch.zeros_like(Rs)
                weights = torch.exp(normalized_Rs_diff / self.alpha).to(self.device)
            else:
                baseline = torch.mean(torch.tensor([r for _, r in self.D]))
                weights = torch.exp( (torch.tensor([r for _, r in self.D]) - baseline) / self.alpha ).to(self.device)  # (N,)
            swanlab.log({"mean": self.omega_mean.mean()})
            swanlab.log({"std": self.omega_std.mean()})


            old_omega_mean = self.omega_mean.clone()
            old_omega_std = self.omega_std.clone()
            opt = torch.optim.SGD([self.omega_mean, self.omega_std], lr=self.step_size)
            for step in range(20):  # SGD步数，视收敛调
                opt.zero_grad()
                zs = torch.stack([z for z, _ in self.D]).to(self.device)  # (N, z_dim)
                log_probs = self.omega_dist.log_prob(zs).sum(-1)  # (N,)
                loss = -torch.mean(weights * log_probs)  # 加权最大似然
                loss.backward()
                opt.step()
                self.omega_std.data.clamp_(1e-4, 1.0)  # 数值稳定
                self.omega_dist = Normal(self.omega_mean, self.omega_std)
                # print(f"{step}:", self.omega_mean)

            # 检查收敛（如停止条件）
            # if self.k_count > 5 and abs(old_omega_mean - self.omega_mean).mean() < 1e-3:
            #     self.k_count = self.k_max
            # if abs(old_omega_mean - self.omega_mean).mean() < 1e-3:
            #     print('ok')
            z_star = self.omega_dist.mean
        else: 
            z_star = z_k
        return z_star  # 最终z*

# # 使用示例
# searcher = AWRSearcher(z_dim=your_model.z_dim)
# def run_episode(z):
#     # 用 policy(s, z) 在真实 env 滚一回合，返回 sum(γ^t r_t)
#     return total_return

# z_star = searcher.adapt(your_policy, your_real_env, run_episode)
# # 后续部署：用 z_star 固定推理策略
