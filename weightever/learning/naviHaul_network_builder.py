# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0
class EnvEncoderE(nn.Module):
    def __init__(self, mu_dim, z_dim=16, hid=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(mu_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid),
        )
        self.mean = nn.Linear(hid, z_dim)
        self.log_std = nn.Linear(hid, z_dim)  # 输出log std
        self.z_dim = z_dim

    def forward(self, mu, deterministic=False):
        h = self.backbone(mu)
        m = self.mean(h)
        log_s = self.log_std(h).clamp(-5.0, 2.0)  # 避免数值炸
        s = torch.exp(log_s)
        if deterministic:
            z = m
        else:
            eps = torch.randn_like(s)
            z = m + s * eps
        # KL(q||N(0,I)) 按维求和，batch保留
        kl = 0.5 * torch.sum(m.pow(2) + s.pow(2) - 2*log_s - 1.0, dim=-1)
        return z, kl
    
class NaviHaulBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return
    class HaulNetwork(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)
                    
            amp_input_shape = kwargs.get('amp_input_shape')
            self._build_disc(amp_input_shape)

            obs_hid = params['env_encoder']['obs_hid']
            env_p_size = params['env_encoder']['ctx_dim']
            z_dim = params['env_encoder']['z_dim']
            env_hid = params['env_encoder']['env_hid']
            self.obs_encoder = nn.Sequential(
                nn.Linear(kwargs.pop('input_shape')[0], 4*obs_hid),
                nn.ReLU(),
                nn.Linear(4*obs_hid, obs_hid),
                nn.ReLU(),

            )
            self.pi_mlp = nn.Sequential(
                nn.Linear(obs_hid + z_dim, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
            )
            self.env_encoder = EnvEncoderE(env_p_size, z_dim, env_hid)
            self.env_state = None
            return
        def load(self, params):
            super().load(params)
            # share same network shape(except input shape)
            self._disc_units = params['disc']['units']
            self._disc_activation = params['disc']['activation']
            self._disc_initializer = params['disc']['initializer']
            return

        def forward(self, obs_dict):
            ctx_info = {}
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            ctx_info['env_info'] = obs_dict.get('env_info',None)
            ctx_info['curr_z'] = obs_dict.get('curr_z',None)


            actor_outputs = self.eval_actor(obs, ctx_info, obs_dict['is_train'])
            value = self.eval_critic(obs)

            output = actor_outputs + (value, states)

            return output

        def eval_actor(self, obs, ctx_info, is_train):
            if ctx_info['env_info'] is not None and ctx_info['curr_z'] is None: # in train procedure
                a_out = self.actor_cnn(obs)
                a_out = a_out.contiguous().view(a_out.size(0), -1)
                h = self.obs_encoder(a_out)
                # if is_train:
                deterministic_z = False # always enabled in training
                z, kl = self.env_encoder(ctx_info['env_info'], deterministic=deterministic_z)
                # else:
                #     if self.use_awr:
                #         z = self.omega_dist.sample()   # z ~ N(m_k, S_k)，整回合固定
                #     elif self.cached_z_star is not None:
                #         z = self.cached_z_star         # 之前AWR得到的最优z*
                #     else:
                #         z = torch.zeros(self.z_dim)    # 无适应的稳健默认

                hz = torch.cat([h, z], dim=-1)
                # a_out = self.actor_mlp(hz) # self.actor_mlp discarded!!
                a_out = self.pi_mlp(hz)
            elif ctx_info['curr_z'] is not None: # awr procedure # only awz provides curr_z
                a_out = self.actor_cnn(obs)
                a_out = a_out.contiguous().view(a_out.size(0), -1)
                h = self.obs_encoder(a_out)
                if len(ctx_info['curr_z'].shape)==1:
                    ctx_info['curr_z'] = ctx_info['curr_z'].unsqueeze(0)
                if len(h)>1:
                    ctx_info['curr_z'] = ctx_info['curr_z'].repeat((len(h),1))
                hz = torch.cat([h, ctx_info['curr_z']], dim=-1)
                # a_out = self.actor_mlp(hz) # self.actor_mlp discarded!!
                a_out = self.pi_mlp(hz)


            else: # navi
                a_out = self.actor_cnn(obs)
                a_out = a_out.contiguous().view(a_out.size(0), -1)
                a_out = self.actor_mlp(a_out)


            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return


        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value

        def eval_disc(self, amp_obs):
            disc_mlp_out = self._disc_mlp(amp_obs)
            disc_logits = self._disc_logits(disc_mlp_out)
            return disc_logits

        def get_disc_logit_weights(self):
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            weights = []
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._disc_logits.weight))
            return weights

        def _build_disc(self, input_shape):
            self._disc_mlp = nn.Sequential()

            mlp_args = {
                'input_size' : input_shape[0], 
                'units' : self._disc_units, 
                'activation' : self._disc_activation, 
                'dense_func' : torch.nn.Linear
            }
            self._disc_mlp = self._build_mlp(**mlp_args)
            
            mlp_out_size = self._disc_units[-1]
            self._disc_logits = torch.nn.Linear(mlp_out_size, 1)

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 

            torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias) 

            return

    class NaviNetwork(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)
                    
            amp_input_shape = kwargs.get('amp_input_shape')
            self._build_disc(amp_input_shape)

            return

        def load(self, params):
            super().load(params)

            self._disc_units = params['disc']['units']
            self._disc_activation = params['disc']['activation']
            self._disc_initializer = params['disc']['initializer']
            return

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)

            actor_outputs = self.eval_actor(obs)
            value = self.eval_critic(obs)

            output = actor_outputs + (value, states)

            return output

        def eval_actor(self, obs):
            a_out = self.actor_cnn(obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            a_out = self.actor_mlp(a_out)
                     
            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value

        def eval_disc(self, amp_obs):
            disc_mlp_out = self._disc_mlp(amp_obs)
            disc_logits = self._disc_logits(disc_mlp_out)
            return disc_logits

        def get_disc_logit_weights(self):
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            weights = []
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._disc_logits.weight))
            return weights

        def _build_disc(self, input_shape):
            self._disc_mlp = nn.Sequential()

            mlp_args = {
                'input_size' : input_shape[0], 
                'units' : self._disc_units, 
                'activation' : self._disc_activation, 
                'dense_func' : torch.nn.Linear
            }
            self._disc_mlp = self._build_mlp(**mlp_args)
            
            mlp_out_size = self._disc_units[-1]
            self._disc_logits = torch.nn.Linear(mlp_out_size, 1)

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 

            torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias) 

            return
    def build(self, name, **kwargs):
        kwargs["input_shape"] = kwargs.get("haul_input_shape", kwargs["input_shape"])
        haul = NaviHaulBuilder.HaulNetwork(self.params, **kwargs)
        kwargs["input_shape"] = kwargs.get("navi_input_shape", kwargs["input_shape"])
        navi = NaviHaulBuilder.NaviNetwork(self.params, **kwargs)
        return haul, navi