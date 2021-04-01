from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
import jax.numpy as jnp
from jax import grad
from jax import jit # for compiling functions for speedup
from jax import random # stax initialization uses jax.random
from jax.experimental import stax # neural network library
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, BatchNorm, LogSoftmax # neural network layers
import matplotlib.pyplot as plt # visualization
from jax.experimental import optimizers
from jax.tree_util import tree_multimap  # Element-wise manipulation of collections of numpy arrays
import jax
import yaml
import os
import torch
import cv2
import numpy as onp
from tqdm import tqdm
import pickle
from functools import partial


hidden_unit = 64
out_channel = 5
step_num = 0
# Use stax to set up network initialization and evaluation functions
net_init, net_apply = stax.serial(
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
                                        LogSoftmax,
        )
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
alpha = .01


def get_sample_image(way_num, support_inputs, query_inputs, support_targets, query_targets, save_dir, step):
    # only support episode number = 1
    assert support_inputs.shape[0] == query_inputs.shape[0] == support_targets.shape[0] == query_targets.shape[0]
    if support_inputs.shape[0] > 1:
        support_inputs = support_inputs[0].unsqueeze(0)
        query_inputs = query_inputs[0].unsqueeze(0)

    os.makedirs(save_dir, exist_ok=True)
    # (5, 84, 84, 3)
    support_data_permute = support_inputs.permute(0, 1, 3, 4, 2).squeeze(0)
    # (75, 84, 84, 3)
    query_data_permute = query_inputs.permute(0, 1, 3, 4, 2).squeeze(0)

    support_data_reshape = torch.reshape(support_data_permute, (way_num, -1, *support_data_permute.shape[1:]))
    query_data_reshape = torch.reshape(query_data_permute, (way_num, -1, *query_data_permute.shape[1:]))
    device = support_inputs.get_device()
    # (5, 1+15, 84, 84, 3)
    black = torch.zeros(support_data_reshape.shape[0], 1, *support_data_reshape.shape[-3:]) + 1
    black = black.cuda() if device != -1 else black
    complete_tensor = torch.cat([support_data_reshape, black, query_data_reshape], dim=1)
    present_list = []
    for row in complete_tensor:
        tensor_list = [tensor for tensor in row]
        tensor_row = torch.cat(tensor_list, dim=1)
        present_list.append(tensor_row)
    present_tensor = torch.cat(present_list, dim=0)
    img = present_tensor.cpu().numpy() * 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, 'sampled_image_{}.png'.format(step)), img)


def torch2jnp(support_inputs, query_inputs, support_targets, query_targets):
    assert support_inputs.shape[0] == query_inputs.shape[0] == support_targets.shape[0] == query_targets.shape[0]
    support_inputs = jnp.array(support_inputs.permute(0, 1, 3, 4, 2).cpu().detach().numpy())
    query_inputs = jnp.array(query_inputs.permute(0, 1, 3, 4, 2).cpu().detach().numpy())
    support_targets = jnp.array(support_targets.cpu().detach().numpy())
    query_targets = jnp.array(query_targets.cpu().detach().numpy())
    return support_inputs, query_inputs, support_targets, query_targets


def reptile_update(p, x, y):
    grads = grad(ce_loss)(p, x, y)
    inner_sgd_fn = lambda g, state: (state - alpha*g)
    # Element-wise manipulation of collections of numpy arrays
    return tree_multimap(inner_sgd_fn, grads, p)


def ce_loss(params, inputs, targets, not_grad=False):
    # Computes average loss for the batch
    predictions = net_apply(params, inputs)
    targets_one_hot = jax.nn.one_hot(targets, predictions.shape[1])
    ce_loss = -jnp.mean(jnp.sum(predictions * targets_one_hot, axis=1))
    acc = onp.sum(jnp.argmax(predictions, 1) == targets) / len(targets)

    if not_grad:
        return ce_loss, acc
    else:
        return ce_loss


def reptile_loss(p, x1, y1, x2, y2, epsilon, i=3, is_train=True):

    if is_train:
        # stack support and query data alone instance number axis (1 + 15)
        combined_x = jnp.concatenate([x1, x2], 0)
        combined_y = jnp.concatenate([y1, y2], 0)

        p2 = reptile_update(p, combined_x, combined_y)
        if i > 1:
            for _ in range(i - 1):
                p2 = reptile_update(p2, combined_x, combined_y)

        total_loss, acc = ce_loss(p2, combined_x, combined_y, not_grad=True)

        return total_loss, acc, param_opreation(p2, p, epsilon, "substract")

    else:
        p2 = reptile_update(p, x1, y1)
        if i > 1:
            for _ in range(i - 1):
                p2 = reptile_update(p2, x1, y1)

        total_loss, acc = ce_loss(p2, x2, y2, not_grad=True)

        return total_loss, acc


def param_opreation(p1, p2, epsilon, opreation):

    assert len(p1) == len(p2)
    result_list = []
    i = 0
    while i < len(p1):
        p1_i = p1[i]
        p2_i = p2[i]
        if len(p1_i) != 0 and len(p2_i) != 0:
            p_i0 = jnp.subtract(p1_i[0], p2_i[0]) * epsilon * -1
            p_i1 = jnp.subtract(p1_i[1], p2_i[1]) * epsilon * -1
            result_list.append((p_i0, p_i1))
        else:
            result_list.append(())
        i += 1

    return result_list


@jax.jit
def reptile_step(i, opt_state, x1, y1, x2, y2):

    episode_num = x1.shape[0]
    results = None
    accs = []
    ls = []
    # if i is not None:
    #     epsilon = 1e-3 * (1 - i / 10000)
    # else:
    #     epsilon = 1e-3 * (1 - 9999 / 10000)
    epsilon = 1e-3
    j = 0
    while j < episode_num:
        p = get_params(opt_state)
        if i is not None:
            l, acc,  param_diff = reptile_loss(p, x1[j], y1[j], x2[j], y2[j], epsilon)
            results = param_diff
        else:
            l, acc = reptile_loss(p, x1[j], y1[j], x2[j], y2[j], epsilon, is_train=False)
        accs.append(acc)
        ls.append(l)
        j += 1

    ls = jnp.mean(jnp.array(ls))
    accs = jnp.mean(jnp.array(accs))

    if i is not None:
        return opt_update(i, results, opt_state), ls, accs
    else:
        return ls, accs


def train(config, train_dataloader, val_dataloader):

    rng = random.PRNGKey(config["seed"])
    in_shape = (-1, config["shape"][0], config["shape"][1], config["shape"][2])
    best_val_acc = 0
    best_val_acc_std = 0
    out_shape, net_params = net_init(rng, in_shape)
    opt_state = opt_init(net_params)
    train_losses = []
    train_accs = []
    val_x = []
    val_accs = []
    val_losses = []
    sampled_interval = config["total_train_iter"] // 10

    for train_iter_count in tqdm(range(config["total_train_iter"])):
        train_batch = next(iter(train_dataloader))
        train_support_inputs, train_support_targets = train_batch["train"]
        train_query_inputs, train_query_targets = train_batch["test"]

        if train_iter_count % sampled_interval == 0:
            get_sample_image(config["N"], train_support_inputs, train_query_inputs, train_support_targets,
                             train_query_targets, config["ckpt_path"], train_iter_count)

        train_support_inputs, train_query_inputs, train_support_targets, train_query_targets = \
            torch2jnp(train_support_inputs, train_query_inputs, train_support_targets, train_query_targets)

        opt_state, train_loss, train_acc = reptile_step(train_iter_count, opt_state, train_support_inputs, train_support_targets, train_query_inputs , train_query_targets)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        if (train_iter_count + 1) % config["val_interval"] == 0:
            val_accs_local = []
            val_losses_local = []
            for _ in range(config["val_size"]):
                val_batch = next(iter(val_dataloader))
                val_support_inputs, val_support_targets = val_batch["train"]
                val_query_inputs, val_query_targets = val_batch["test"]
                val_support_inputs, val_query_inputs, val_support_targets, val_query_targets = \
                    torch2jnp(val_support_inputs, val_query_inputs, val_support_targets, val_query_targets)
                val_loss, val_acc = reptile_step(None, opt_state, val_support_inputs, val_support_targets, val_query_inputs , val_query_targets)
                val_accs_local.append(val_acc)
                val_losses_local.append(val_loss)
            val_acc_mean = onp.mean(onp.array(val_accs_local))
            val_acc_std = onp.std(onp.array(val_accs_local))
            val_loss_mean = onp.mean(onp.array(val_losses_local))

            print("average validation accuracy at: {}, mean: {}, std: {}".format(train_iter_count + 1, val_acc_mean, val_acc_std))
            if val_acc_mean > best_val_acc:
                best_val_acc = val_acc_mean
                best_val_acc_std = val_acc_std
                trained_params = optimizers.unpack_optimizer_state(opt_state)
                with open(os.path.join(config["ckpt_path"], "best_ckpt.pkl"), "wb") as ckpt:
                    pickle.dump(trained_params, ckpt)
                ckpt.close()
                del trained_params
                print("best validation ckpt saved")
            val_x.append(train_iter_count)
            val_accs.append(val_acc_mean)
            val_losses.append(val_loss_mean)

    plt.plot(onp.arange(config["total_train_iter"]), train_losses, label='train loss', alpha=0.5)
    plt.plot(val_x, val_losses, label='val loss')
    plt.legend(loc="best")
    plt.savefig(os.path.join(config["ckpt_path"], "./maml_jax_loss.png"))
    plt.close()
    plt.plot(onp.arange(config["total_train_iter"]), train_accs, label='train accuracy', alpha=0.5)
    plt.plot(val_x, val_accs, label='val accuracy')
    plt.legend(loc="best")
    plt.savefig(os.path.join(config["ckpt_path"], "./maml_jax_acc.png"))
    print("best validation accuracy: {}, std: {}".format(best_val_acc, best_val_acc_std))


def test(config, test_dataloader):
    best_params = pickle.load(open(os.path.join(config["ckpt_path"], "best_ckpt.pkl"), "rb"))
    best_opt_state = optimizers.pack_optimizer_state(best_params)
    print()
    print("best ckpt loaded")
    test_accs_local = []
    test_losses_local = []
    for _ in range(config["test_size"]):
        test_batch = next(iter(test_dataloader))
        test_support_inputs, test_support_targets = test_batch["train"]
        test_query_inputs, test_query_targets = test_batch["test"]
        test_support_inputs, test_query_inputs, test_support_targets, test_query_targets = \
            torch2jnp(test_support_inputs, test_query_inputs, test_support_targets, test_query_targets)
        test_loss, test_acc = reptile_step(None, best_opt_state, test_support_inputs, test_support_targets, test_query_inputs, test_query_targets)
        test_accs_local.append(test_acc)
        test_losses_local.append(test_loss)
    test_acc_mean = onp.mean(onp.array(test_accs_local))
    test_acc_std = onp.std(onp.array(test_accs_local))
    test_loss_mean = onp.mean(onp.array(test_losses_local))
    print("test accuracy: {}, std: {}".format(test_acc_mean, test_acc_std))
    print("test loss: {}".format(test_loss_mean))
    print()


def main():
    config = yaml.load(open("./config.yaml"), Loader=yaml.Loader)
    # config = yaml.load(open("./config_small.yaml"), Loader=yaml.Loader)

    train_dataset = omniglot("data", ways=config["N"], shots=config["K"], test_shots=config["Q"], meta_train=True, download=True)
    val_dataset = omniglot("data", ways=config["N"], shots=config["K"], test_shots=config["Q"], meta_val=True, download=True)
    test_dataset = omniglot("data", ways=config["N"], shots=config["K"], test_shots=config["Q"], meta_test=True, download=True)

    # bug if set > 0 (OSError: [Errno 39] Directory not empty: '/tmp/pymp-f0p5cvzq')
    train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=config["batch_size"], num_workers=0)
    val_dataloader = BatchMetaDataLoader(val_dataset, batch_size=config["batch_size"], num_workers=0)
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=config["batch_size"], num_workers=0)

    train(config, train_dataloader, val_dataloader)
    test(config, test_dataloader)


if __name__ == "__main__":
    from datetime import datetime
    start = datetime.now()
    main()
    end = datetime.now()
    print("program takes: {}".format(end-start))