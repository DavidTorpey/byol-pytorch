import torch
import torch.nn.functional as F
import numpy as np


class BYOLTrainer:
    def __init__(
            self, online_network, target_network, predictor,
            optimizer, scheduler, batch_size, epochs, device,
            dataset, m, run_folder
    ):
        self.online_network = online_network
        self.target_network = target_network
        self.predictor = predictor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.dataset = dataset
        self.m = m
        self.m_base = m
        self.run_folder = run_folder

        self.model_name = 'model_{}.pth'.format(self.dataset)

    @torch.no_grad()
    def _update_target_network_parameters(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return -2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def train(self, train_loader, val_loader, K):
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        self.initializes_target_network()

        for epoch_counter in range(self.epochs):
            print(epoch_counter + 1, self.epochs)
            for xis, xjs in train_loader:
                self.optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(xis, xjs)

                loss.backward()

                self.optimizer.step()
                n_iter += 1

                self._update_target_network_parameters()

                self.m = 1 - (1 - self.m_base) * (np.cos(np.pi * n_iter / K) + 1) / 2

            valid_loss = self._validate(val_loader)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_model(
                    './results/{}/{}'.format(self.run_folder, self.model_name)
                )

            if epoch_counter >= 10:
                self.scheduler.step()

            valid_n_iter += 1

            if epoch_counter % 10 == 0:
                self.save_model(
                    './results/{}/checkpoints/'.format(self.run_folder) + str(epoch_counter) + '_' + self.model_name
                )

    def _validate(self, val_loader):
        with torch.no_grad():
            self.online_network.eval()
            self.target_network.eval()
            self.predictor.eval()

            valid_loss = 0.0
            counter = 0
            for xis, xjs in val_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(xis, xjs)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter

        self.online_network.train()
        self.target_network.train()
        self.predictor.train()

        return valid_loss

    def _step(self, batch_view_1, batch_view_2):
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)

        return loss.mean()

    def save_model(self, path):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
