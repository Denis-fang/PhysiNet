class GCDDLayer(nn.Module):
    """Gaussian Curvature-Driven Diffusion Layer"""
    def __init__(self, time_steps=100, dt=0.1, alpha=1.0, beta=0.1):
        super().__init__()
        self.time_steps = time_steps
        self.dt = dt
        self.alpha = alpha
        self.beta = beta 