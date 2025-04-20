import torch
import torch.nn as nn
import torch.nn.functional as F


class RF_Diffusion(nn.Module):
    """
    用于Rectified Flow的神经网络模型
    预测从噪声到数据的速度场
    """

    def __init__(self, n_steps, dim, num_units=64, condition=True, cond_dim=0):
        super(RF_Diffusion, self).__init__()
        self.channels = 1
        self.self_condition = False  # RF不需要自条件
        self.condition = condition

        # 时间编码网络
        self.time_mlp = nn.Sequential(nn.Linear(1, num_units), nn.GELU(), nn.Linear(num_units, num_units))

        # 处理空间维度的网络
        self.linears_spatial = nn.ModuleList([
            nn.Linear(dim - 1, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
        ])

        # 处理时间维度的网络
        self.linears_temporal = nn.ModuleList([
            nn.Linear(1, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
        ])

        # 输出层
        self.output_spatial = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units, dim - 1))
        self.output_temporal = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units, 1))

        # 注意力权重网络 - 学习如何组合时间和空间信息
        self.linear_t = nn.Sequential(nn.Linear(num_units * 2, num_units), nn.ReLU(), nn.Linear(num_units, num_units),
                                      nn.ReLU(), nn.Linear(num_units, 2))
        self.linear_s = nn.Sequential(nn.Linear(num_units * 2, num_units), nn.ReLU(), nn.Linear(num_units, num_units),
                                      nn.ReLU(), nn.Linear(num_units, 2))

        # 条件编码网络
        if condition:
            self.cond_all = nn.Sequential(nn.Linear(cond_dim * 3, num_units), nn.ReLU(),
                                          nn.Linear(num_units, num_units))
            self.cond_temporal = nn.ModuleList(
                [nn.Linear(cond_dim, num_units),
                 nn.Linear(cond_dim, num_units),
                 nn.Linear(cond_dim, num_units)])
            self.cond_spatial = nn.ModuleList(
                [nn.Linear(cond_dim, num_units),
                 nn.Linear(cond_dim, num_units),
                 nn.Linear(cond_dim, num_units)])
            self.cond_joint = nn.ModuleList(
                [nn.Linear(cond_dim, num_units),
                 nn.Linear(cond_dim, num_units),
                 nn.Linear(cond_dim, num_units)])

    def get_attn(self, x, t, x_self_cond=None, cond=None):
        """获取注意力权重"""
        x_spatial, x_temporal = x[:, :, 1:].clone(), x[:, :, :1].clone()

        if self.condition:
            hidden_dim = int(cond.shape[-1] / 3)
            cond_temporal, cond_spatial, cond_joint = (cond[:, :, :hidden_dim], cond[:, :, hidden_dim:2 * hidden_dim],
                                                       cond[:, :, 2 * hidden_dim:])
            cond = self.cond_all(cond)
        else:
            cond = torch.zeros_like(x_spatial)

        # 时间嵌入
        t_embedding = self.time_mlp(t.unsqueeze(-1)).unsqueeze(dim=1)

        # 结合条件和时间信息
        cond_all = torch.cat((cond, t_embedding), dim=-1)

        # 计算空间和时间的注意力权重
        alpha_s = F.softmax(self.linear_s(cond_all), dim=-1).squeeze(dim=1)
        alpha_t = F.softmax(self.linear_t(cond_all), dim=-1).squeeze(dim=1)

        return alpha_s, alpha_t

    def forward(self, x, t, x_self_cond=None, cond=None):
        """
        模型前向传播
        x: [B, 1, dim+1] - 包含时间和空间坐标的输入
        t: [B] - 时间步
        cond: [B, 1, cond_dim*3] - 条件信息
        """
        # 分离时间和空间维度
        x_spatial, x_temporal = x[:, :, 1:].clone(), x[:, :, :1].clone()

        # 条件处理
        if self.condition and cond is not None:
            hidden_dim = int(cond.shape[-1] / 3)
            cond_temporal, cond_spatial, cond_joint = (cond[:, :, :hidden_dim], cond[:, :, hidden_dim:2 * hidden_dim],
                                                       cond[:, :, 2 * hidden_dim:])
            cond = self.cond_all(cond)
        else:
            cond = torch.zeros((x.shape[0], 1, self.linears_spatial[0].out_features), device=x.device)
            if self.condition:
                cond_temporal = cond_spatial = cond_joint = cond

        # 时间编码
        t_embedding = self.time_mlp(t.unsqueeze(-1)).unsqueeze(dim=1)

        # 结合条件和时间
        cond_all = torch.cat((cond, t_embedding), dim=-1)

        # 计算注意力权重
        alpha_s = F.softmax(self.linear_s(cond_all), dim=-1).squeeze(dim=1).unsqueeze(dim=2)
        alpha_t = F.softmax(self.linear_t(cond_all), dim=-1).squeeze(dim=1).unsqueeze(dim=2)

        # 多层处理
        for idx in range(3):
            # 时空特征提取
            x_spatial = self.linears_spatial[2 * idx](x_spatial)
            x_temporal = self.linears_temporal[2 * idx](x_temporal)

            # 添加时间编码
            x_spatial = x_spatial + t_embedding
            x_temporal = x_temporal + t_embedding

            # 添加条件
            if self.condition and cond is not None:
                cond_joint_emb = self.cond_joint[idx](cond_joint)
                cond_temporal_emb = self.cond_temporal[idx](cond_temporal)
                cond_spatial_emb = self.cond_spatial[idx](cond_spatial)

                x_spatial = x_spatial + cond_joint_emb + cond_spatial_emb
                x_temporal = x_temporal + cond_joint_emb + cond_temporal_emb

            # 非线性激活
            x_spatial = self.linears_spatial[2 * idx + 1](x_spatial)
            x_temporal = self.linears_temporal[2 * idx + 1](x_temporal)

        # 最后一层处理
        x_spatial = self.linears_spatial[-1](x_spatial)
        x_temporal = self.linears_temporal[-1](x_temporal)

        # 堆叠特征
        x_output = torch.cat((x_temporal, x_spatial), dim=1)  # [B, 2, num_units]

        # 应用注意力权重
        x_output_t = (x_output * alpha_t).sum(dim=1, keepdim=True)
        x_output_s = (x_output * alpha_s).sum(dim=1, keepdim=True)

        # 输出时空预测值
        pred_temporal = self.output_temporal(x_output_t)
        pred_spatial = self.output_spatial(x_output_s)

        # 合并时空维度
        pred = torch.cat((pred_temporal, pred_spatial), dim=-1)
        return pred  # [B, 1, 1+dim]
