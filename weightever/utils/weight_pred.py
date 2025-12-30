import torch

class WeightPred:
    def __init__(self):
        self.box = 0

    def random_output(self):
        return 0
    def prepare_topK(self, failure_envs):
        return 
    def access_topK(self,boxdown_envids, mode = "weightmodel"):
        device = boxdown_envids.device
        num_envs = len(boxdown_envids)
        topKidx = torch.ones((num_envs,4),device=device,dtype=torch.float)
        if mode=="random":
            # 组1：从 [0,1] 中随机抽1个，形状 [num_envs, 1]
            group1_idx = torch.randint(low=0, high=2, size=(len(boxdown_envids),), device=device).to(torch.float)
            # 组2：从 [2,3] 中随机抽1个，形状 [num_envs, 1]
            group2_idx = torch.randint(low=2, high=4, size=(len(boxdown_envids),), device=device).to(torch.float)
            
            # 生成随机交换掩码（0表示不交换，1表示交换），形状与failure_envs一致
            swap_mask = torch.randint(0, 2, size=(len(boxdown_envids),), device=device).bool()

            # 1. 处理不交换的情况（mask为False）
            no_swap_indices = ~swap_mask  # 不交换的掩码
            topKidx[no_swap_indices, 1] = group1_idx[no_swap_indices]  # group1给列1
            topKidx[no_swap_indices, 3] = group2_idx[no_swap_indices]  # group2给列3

            # 2. 处理交换的情况（mask为True）
            swap_indices = swap_mask  # 交换的掩码
            topKidx[swap_indices, 1] = group2_idx[swap_indices]  # group2给列1
            topKidx[swap_indices, 3] = group1_idx[swap_indices]  # group1给列3
            
            
            topKidx[...,2] = torch.rand(size=(len(boxdown_envids),), device=device)  # 每个环境独立随机
            
        elif mode=="weightmodel":   
            pass
        elif mode=="base train":
            group_idx = torch.randint(low=0, high=4, size=(num_envs,), device=device).to(torch.float)
            topKidx[...,1] = group_idx/2
            topKidx[...,3] = 0
            topKidx[...,2] = 0 
        return topKidx

    def get_new_pos(self, boxdown_envids):
        # hardcode: topK #TODO:self.stand_points_offset should be (envs,k,3) and sorted and updated later
        topKidx = torch.ones((self.num_envs,4),device=self.device,dtype=torch.float)
        # 组1：从 [0,1] 中随机抽1个，形状 [num_envs, 1]
        group1_idx = torch.randint(low=0, high=2, size=(len(boxdown_envids),), device=self.device).to(torch.float)
        # 组2：从 [2,3] 中随机抽1个，形状 [num_envs, 1]
        group2_idx = torch.randint(low=2, high=4, size=(len(boxdown_envids),), device=self.device).to(torch.float)
        
        # 生成随机交换掩码（0表示不交换，1表示交换），形状与failure_envs一致
        swap_mask = torch.randint(0, 2, size=(len(boxdown_envids),), device=self.device).bool()

        # 1. 处理不交换的情况（mask为False）
        no_swap_indices = ~swap_mask  # 不交换的掩码
        topKidx[boxdown_envids[no_swap_indices], 1] = group1_idx[no_swap_indices]  # group1给列1
        topKidx[boxdown_envids[no_swap_indices], 3] = group2_idx[no_swap_indices]  # group2给列3

        # 2. 处理交换的情况（mask为True）
        swap_indices = swap_mask  # 交换的掩码
        topKidx[boxdown_envids[swap_indices], 1] = group2_idx[swap_indices]  # group2给列1
        topKidx[boxdown_envids[swap_indices], 3] = group1_idx[swap_indices]  # group1给列3
        
        
        topKidx[boxdown_envids,2] = torch.rand(size=(len(boxdown_envids),), device=self.device)  # 每个环境独立随机

        return topKidx[boxdown_envids]
