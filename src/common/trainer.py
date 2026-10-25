import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator


class AbstractTrainer(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        raise NotImplementedError("Method [next] should be implemented.")

    def evaluate(self, eval_data):
        raise NotImplementedError("Method [next] should be implemented.")


class Trainer(AbstractTrainer):
    def __init__(self, config, model, mg=False):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config["learner"]
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.eval_step = min(config["eval_step"], self.epochs)
        self.stopping_step = config["stopping_step"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.valid_metric = config["valid_metric"].lower()
        self.valid_metric_bigger = config["valid_metric_bigger"]
        self.test_batch_size = config["eval_batch_size"]
        self.device = config["device"]
        self.model_name = config["model"]
        self.weight_decay = 0.0
        if config["weight_decay"] is not None:
            wd = config["weight_decay"]
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config["req_training"]

        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config["metrics"], config["topk"])):
            tmp_dd[f"{j.lower()}@{k}"] = 0.0
        self.best_valid_score = -float("inf") if self.valid_metric_bigger else float("inf")
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.best_model_params = None
        self.optimizer = self._build_optimizer()

        lr_scheduler = config["learning_rate_scheduler"]
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config["eval_type"]
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None
        self.mg = mg
        self.alpha1 = config["alpha1"]
        self.alpha2 = config["alpha2"]
        self.beta = config["beta"]

    def _build_optimizer(self):
        if self.learner.lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.learner.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        if not self.req_training:
            return 0.0, []
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            second_inter = interaction.clone() if self.mg else None
            losses = (
                loss_func(interaction)

            )

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
            if self._check_nan(loss):
                self.logger.info(
                    "Loss is nan at epoch: {}, batch index: {}. Exiting.".format(
                        epoch_idx, batch_idx
                    )
                )
                return loss, torch.tensor(0.0)

            if self.mg and batch_idx % self.beta == 0:
                first_loss = self.alpha1 * loss
                first_loss.backward()

                self.optimizer.step()
                self.model.post_optimizer_step()
                self.optimizer.zero_grad()

                losses = loss_func(second_inter)
                if isinstance(losses, tuple):
                    loss = sum(losses)
                else:
                    loss = losses

                if self._check_nan(loss):
                    self.logger.info(
                        "Loss is nan at epoch: {}, batch index: {}. Exiting.".format(
                            epoch_idx, batch_idx
                        )
                    )
                    return loss, torch.tensor(0.0)
                second_loss = -1 * self.alpha2 * loss
                second_loss.backward()
            else:
                loss.backward()

            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            self.model.post_optimizer_step()
            loss_batches.append(loss.detach())

        return total_loss, loss_batches

    def _valid_epoch(self, valid_data):
        valid_result = self.evaluate(valid_data)
        valid_score = (
            valid_result[self.valid_metric]
            if self.valid_metric
            else valid_result["NDCG@20"]
        )
        return valid_score, valid_result

    def _check_nan(self, loss):
        if not torch.isfinite(loss):
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = "epoch %d training [time: %.2fs, " % (
            epoch_idx,
            e_time - s_time,
        )
        if isinstance(losses, tuple):
            train_loss_output += ", ".join(
                "train_loss%d: %.4f" % (idx + 1, loss)
                for idx, loss in enumerate(losses)
            )
        else:
            train_loss_output += "train loss: %.4f" % losses
        return train_loss_output + "]"

    def fit(
        self, train_data, valid_data=None, test_data=None, saved=False, verbose=True
    ):
        self.best_test_upon_valid = {}
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                break
            self.lr_scheduler.step()
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            self.model.post_epoch_processing()
            if verbose:
                self.logger.info(self._generate_train_loss_output(
                    epoch_idx, training_start_time, time(), train_loss))
            should_validate = valid_data is not None and self.eval_step > 0 and (
                (epoch_idx + 1) % self.eval_step == 0 or epoch_idx + 1 == self.epochs)
            if not should_validate:
                continue
            valid_score, valid_result = self._valid_epoch(valid_data)
            self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                valid_score, self.best_valid_score, self.cur_step,
                max_step=self.stopping_step, bigger=self.valid_metric_bigger)
            if verbose:
                self.logger.info("epoch %d valid result: %s", epoch_idx, dict2str(valid_result))
            if update_flag:
                self.best_valid_result = valid_result
                self.best_model_params = {
                    key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
            if stop_flag:
                if verbose:
                    self.logger.info("Early stopping at epoch %d", epoch_idx)
                break
        if self.best_model_params is None:
            self.best_model_params = {
                key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        self.model.load_state_dict(self.best_model_params)
        if test_data is not None:
            self.best_test_upon_valid = self.evaluate(test_data, is_test=True)
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        self.model.eval()
        self.model.is_final_test = is_test
        matrices = {}
        prepare = getattr(self.model, "prepare_for_eval", None)
        clear = getattr(self.model, "clear_eval_cache", None)
        try:
            if prepare is not None:
                prepare()
            for batched_data in eval_data:
                scores = self.model.full_sort_predict(batched_data)
                if not isinstance(scores, dict):
                    view = getattr(self.model, "current_eval_view", "student")
                    scores = {"student" if is_test or view == "all" else view: scores}
                for view, values in scores.items():
                    masked = batched_data[1]
                    values[masked[0], masked[1]] = -float("inf")
                    topk = torch.topk(values, max(self.config["topk"]), dim=-1).indices
                    matrices.setdefault(view, []).append(topk)
            results = {}
            for view, matrix_list in matrices.items():
                results[view] = self.evaluator.evaluate(matrix_list, eval_data, is_test=is_test, idx=idx)
                if not is_test and hasattr(self.model, "record_eval_metrics"):
                    self.model.record_eval_metrics(view, results[view])
            if not results:
                raise ValueError("Cannot evaluate an empty dataloader")
            return results["student"] if "student" in results else next(iter(results.values()))
        finally:
            if clear is not None:
                clear()
            self.model.is_final_test = False

    def plot_train_loss(self, show=True, save_path=None):
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
