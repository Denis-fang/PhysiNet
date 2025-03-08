class ModelArgs:
    def __init__(self):
        self.n_modalities = 4  # 默认使用4个模态
        self.use_MixtureOfMamba = True
        self.use_BaselineDenseMamba = False
        self.do_not_split_in_proj = False
        self.do_not_split_x_proj = False
        self.do_not_split_dt_proj = False
        self.do_not_split_out_proj = False 