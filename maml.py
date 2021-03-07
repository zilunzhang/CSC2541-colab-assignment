import jax
import jax.numpy as np


class MAML:

    def __init__(self, ckpt_path, inference=False):
        self.ckpt_path = ckpt_path
        self.is_inference = inference

    def backbone(self):
        pass

    def forward(self, support_inputs, query_inputs, support_targets, query_targets):
        acc = 0.6
        loss = 1.1
        return acc, loss