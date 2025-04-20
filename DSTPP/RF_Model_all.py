import torch.nn as nn


class RF_Model_all(nn.Module):
    """
    封装Spatio-temporal Encoder和Rectified Flow模型
    """

    def __init__(self, transformer, rf_model):
        super(RF_Model_all, self).__init__()
        self.transformer = transformer  # DSTPP(section 3.1) Spatio-temporal Encoder
        self.diffusion = rf_model  # 保持接口一致性

    # def forward(self, x, cond=None):
    #     return self.diffusion(x, cond)
