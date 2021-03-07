import jax
import jax.numpy as np
from jax.experimental import stax
from jax.experimental.stax import (BatchNorm, Conv, Dense, Flatten, Relu, LogSoftmax, MaxPool)
from jax import random


class MAML:

    def __init__(self, batch_size, num_way, ckpt_path, img_width=28, inference=False):
        self.ckpt_path = ckpt_path
        self.is_inference = inference
        self.num_class = num_way
        self.batch_size = batch_size
        self.img_width = img_width
        self.init_param, self.backbone = self.init_backbone(out_channel=self.num_class)
        # print(print(jax.tree_map(self.backbone)))

    def init_backbone(self, in_channel=1, out_channel=128, hidden_unit=64):
        key = random.PRNGKey(1)

        init_fun, conv_net = stax.serial(
                                        Conv(
                                            hidden_unit, (3, 3),
                                            strides=(1, 1),
                                            padding=[(1, 1), (1, 1)]
                                        ),
                                        BatchNorm(),
                                        Relu,
                                        MaxPool(
                                            (2, 2),
                                            strides=(2, 2),
                                            padding="SAME",

                                        ),


                                        Conv(
                                            hidden_unit, (3, 3),
                                            strides=(1, 1),
                                            padding=[(1, 1), (1, 1)]

                                        ),
                                        BatchNorm(),
                                        Relu,
                                        MaxPool(
                                            (2, 2),
                                            strides=(2, 2),
                                            padding="SAME",

                                        ),


                                        Conv(
                                            hidden_unit, (3, 3),
                                            strides=(1, 1),
                                            padding=[(1, 1), (1, 1)]

                                        ),
                                        BatchNorm(),
                                        Relu,
                                        MaxPool(
                                            (2, 2),
                                            strides=(2, 2),
                                            padding="SAME",

                                        ),


                                        Conv(
                                            hidden_unit, (3, 3),
                                            strides=(1, 1),
                                            padding=[(1, 1), (1, 1)]

                                        ),
                                        BatchNorm(),
                                        Relu,
                                        MaxPool(
                                            (2, 2),
                                            strides=(2, 2),
                                            padding="SAME",

                                        ),

                                        Flatten,
                                        Dense(out_channel),
        )
        input_tensor_shape = (self.batch_size, self.img_width, self.img_width, in_channel)
        _, params = init_fun(key, input_tensor_shape)

        return params, conv_net

    def forward(self, support_inputs, query_inputs, support_targets, query_targets):
        acc = 0.6
        loss = 1.1
        all_tensor = np.concatenate([support_inputs, query_inputs], 0)
        all_label = np.concatenate([support_targets, query_targets])

        features = self.backbone(self.init_param, all_tensor)
        print(features.shape)
        return acc, loss