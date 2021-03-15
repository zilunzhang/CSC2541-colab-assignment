from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
import jax
import jax.numpy as np
import numpy as onp
import yaml
import random
import torch
from maml import MAML
from utils import *



class MAMLTrainer:
    def __init__(self, train_dataloader, val_dataloader, test_dataloader, model, config):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.config = config
        self.best_validation_accuracy = 0


    def train(self):
        total_train_iter = self.config["total_train_iter"]
        val_interval = self.config["val_interval"]
        val_size = self.config["val_size"]

        train_iter_count = 0
        # train the model, main loop
        while train_iter_count < total_train_iter:

            train_batch = next(iter(self.train_dataloader))
            train_support_inputs, train_support_targets = train_batch["train"]
            train_query_inputs, train_query_targets = train_batch["test"]

            if train_iter_count == 0:
                get_sample_image(self.config["N"], train_support_inputs, train_query_inputs, train_support_targets, train_query_targets, self.config["ckpt_path"])

            train_support_inputs, train_query_inputs, train_support_targets, train_query_targets = \
                torch2jnp(train_support_inputs, train_query_inputs, train_support_targets, train_query_targets)



            # print("Train iteration count: {}".format(train_iter_count))
            # print('Train support set inputs shape: {}'.format(train_support_inputs.shape))    # (1, 25, 1, 28, 28)
            # print('Train query set inputs shape: {}'.format(train_query_inputs.shape))    # (1, 75, 1, 28, 28)
            # print('Train support set targets shape: {}'.format(train_support_targets.shape))  # (1, 25)
            # print('Train query set targets shape: {}'.format(train_query_targets.shape))  # (1, 75)
            # print()
            # TODO implement MAML model
            train_acc, train_loss = self.model.forward(train_support_inputs, train_query_inputs, train_support_targets, train_query_targets)
            print("train iteration: {:d}, train acc: {:.4f}, train loss: {:.4f}".format(train_iter_count, train_acc, train_loss))

            # perform validation
            if train_iter_count % val_interval == 0 and train_iter_count > 0:
                val_accs = []
                val_losses = []
                val_iter_count = 0
                while val_iter_count < val_size:
                    val_batch = next(iter(self.val_dataloader))
                    val_support_inputs, val_support_targets = val_batch["train"]
                    val_query_inputs, val_query_targets = val_batch["test"]
                    val_support_inputs, val_query_inputs, val_support_targets, val_query_targets = \
                        torch2jnp(val_support_inputs, val_query_inputs, val_support_targets, val_query_targets)
                    # print("Val iteration count: {}".format(val_iter_count))
                    # print('Val support set inputs shape: {}'.format(val_support_inputs.shape))  # (1, 25, 1, 28, 28)
                    # print('Val query set inputs shape: {}'.format(val_query_inputs.shape))  # (1, 75, 1, 28, 28)
                    # print('Val support set targets shape: {}'.format(val_support_targets.shape))  # (1, 25)
                    # print('Val query set targets shape: {}'.format(val_query_targets.shape))  # (1, 75)
                    # print()
                    # TODO implement MAML model
                    val_acc, val_loss = self.model.forward(val_support_inputs, val_query_inputs, val_support_targets, val_query_targets)
                    val_accs.append(val_acc)
                    val_losses.append(val_loss)
                    val_iter_count += 1
                val_acc_mean = np.mean(np.array(val_accs))
                val_acc_std = np.std(np.array(val_accs))
                val_loss_mean = np.mean(np.array(val_losses))
                print("------------------------------------------------------------------------------------------")
                print("validation summary at iteration {:d}: \n    val acc: {:.2f} \u00B1 {:.2f}, val loss: {:.2f}".format(train_iter_count, val_acc_mean, val_acc_std, val_loss_mean))
                print("------------------------------------------------------------------------------------------")
                if val_acc_mean > self.best_validation_accuracy:
                    # TODO: use jax saves model weight
                    print("find better model weight at iteration: {}".format(train_iter_count))
                    print("--------------------------------------------")
                    self.best_validation_accuracy = val_acc_mean
                    pass
            train_iter_count += 1


    def test(self):
        ckpt_path = self.config["ckpt_path"]
        test_size = self.config["test_size"]
        test_iter_count = 0
        test_accs = []
        test_losses = []
        model = MAML(self.config["batch_size"], self.config["N"], ckpt_path=ckpt_path, inference=True)
        while test_iter_count < test_size:
            test_batch = next(iter(self.test_dataloader))
            test_support_inputs, test_support_targets = test_batch["train"]
            test_query_inputs, test_query_targets = test_batch["test"]
            test_support_inputs, test_query_inputs, test_support_targets, test_query_targets = \
                torch2jnp(test_support_inputs, test_query_inputs, test_support_targets, test_query_targets)
            # print("Test iteration count: {}".format(test_iter_count))
            # print('Test support set inputs shape: {}'.format(test_support_inputs.shape))  # (1, 25, 1, 28, 28)
            # print('Test query set inputs shape: {}'.format(test_query_inputs.shape))  # (1, 75, 1, 28, 28)
            # print('Test support set targets shape: {}'.format(test_support_targets.shape))  # (1, 25)
            # print('Test query set targets shape: {}'.format(test_query_targets.shape))  # (1, 75)
            # print()
            # TODO implement MAML model
            test_acc, test_loss = model.forward(test_support_inputs, test_query_inputs, test_support_targets, test_query_targets)
            test_accs.append(test_acc)
            test_losses.append(test_loss)
            test_iter_count += 1

        test_acc_mean = np.mean(np.array(test_accs))
        test_acc_std = np.std(np.array(test_accs))
        test_loss_mean = np.mean(np.array(test_losses))

        print("------------------------------------------------------------------------------------------")
        print("test summary: \n    test acc: {:.2f} \u00B1 {:.2f}, test loss: {:.2f}".format(test_acc_mean, test_acc_std, test_loss_mean))
        print("------------------------------------------------------------------------------------------")


def main():

    config = yaml.load(open("./config.yaml"), Loader=yaml.FullLoader)

    # # set random seed, not working
    # seed = config["seed"]
    # onp.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    train_dataset = omniglot("data", ways=config["N"], shots=config["K"], test_shots=config["Q"], meta_train=True, download=True)
    val_dataset = omniglot("data", ways=config["N"], shots=config["K"], test_shots=config["Q"], meta_val=True, download=True)
    test_dataset = omniglot("data", ways=config["N"], shots=config["K"], test_shots=config["Q"], meta_test=True, download=True)

    train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=config["batch_size"], num_workers=0)
    val_dataloader = BatchMetaDataLoader(val_dataset, batch_size=config["batch_size"], num_workers=0)
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=config["batch_size"], num_workers=0)

    model = MAML(config["batch_size"] * config["N"] * (config["K"] + config["Q"]), config["N"], config["ckpt_path"])
    trainer = MAMLTrainer(train_dataloader, val_dataloader, test_dataloader, model, config)
    trainer.train()
    trainer.test()

if __name__ == "__main__":
    main()