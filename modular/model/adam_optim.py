import torch

class Adam:
  def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
      self.params = list(params)
      self.lr = lr
      self.betas = betas
      self.eps = eps
      self.weight_decay = weight_decay
      self.t = 0
      self.m = []
      self.v = []

      for param in self.params:
          self.m.append(torch.zeros_like(param))
          self.v.append(torch.zeros_like(param))

  def step(self):
      for i, param in enumerate(self.params):
          self.m[i] = self.betas[0]*self.m[i] + (1-self.betas[0])*param.grad
          self.v[i] = self.betas[1]*self.v[i] + (1-self.betas[1])*(param.grad**2)

          bias_correction1 = 1 - self.betas[0]**(self.t+1)
          bias_correction2 = 1 - self.betas[1]**(self.t+1)

          m_hat = self.m[i] / bias_correction1
          v_hat = self.v[i] / bias_correction2

          param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps) + self.weight_decay * param.data

      self.t += 1
