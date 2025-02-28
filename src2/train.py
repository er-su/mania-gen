import torch
from torch import nn
from pathlib import Path
from termcolor import colored
from dataset import OsuDataset
from pytorch_model import OsuGen
from pytorch_model import hyperparams
from torch.utils.data import DataLoader, random_split

class Trainer:
    def __init__(self, model: OsuGen, dataset: OsuDataset, hyperparams, checkpoint_path: Path=Path("checkpoints")):
        self.model = model
        self.dataset = dataset
        self.num_epochs = hyperparams["num_epochs"]
        self.batch_size = hyperparams["batch_size"]
        self.learning_rate = hyperparams["learning_rate"]
        self.checkpoint_path = checkpoint_path
        self.train_set, self.val_set = random_split(self.dataset, [0.8, 0.2])
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.gamma = 2
        self.alpha = 0.25
        self.cuda = torch.cuda.is_available()
    
    def train_one_epoch(self, epoch_index):

        print(colored(f"######### EPOCH {epoch_index} TRAINING #########", "blue"))

        self.model.train()
        for i, data in enumerate(self.train_loader):
            mels, beat_fracs, beat_nums, onsets, actions = data
            if self.cuda:
                mels = mels.to("cuda")
                beat_fracs = beat_fracs.to("cuda")
                beat_nums = beat_nums.to("cuda")
                onsets = onsets.to("cuda")
                actions = actions.to("cuda")
            
            self.optimizer.zero_grad()

            onset_preds, action_preds = self.model(mels, beat_fracs, beat_nums)

            # Flatten across all batches into single mega tensor to calculate loss
            onset_preds = torch.reshape(onset_preds, [-1])
            onsets = torch.reshape(onsets, [-1])
            # softmax applied to convert probabilites for all possible actions
            action_preds = torch.reshape(action_preds, [-1, action_preds.shape[-1]]).softmax(-1) # batch, timestep, num_combo -> batch * timestep, num_combo
            actions = torch.reshape(actions, [-1])

            # Create tensor where each value is the probablity of the correct action for loss func
            action_preds = action_preds[torch.arange(len(action_preds)), actions]
            # Create alpha mask, devaluing when no actions exist
            action_alphas = torch.where(actions == 0, self.alpha, 1)

            onset_loss = FocalLosses.focal_loss(onsets, onset_preds, self.gamma, self.alpha)
            action_loss = FocalLosses.action_focal_loss(action_preds, self.gamma, action_alphas)

            batch_loss = onset_loss * 4 + action_loss
            batch_loss.backward()
            self.optimizer.step()

            onset_acc = (onset_preds == onsets).float().mean()
            action_acc = (action_preds.argmax(dim=-1) == actions).float().mean()

            print(f"Batch {i} onset loss: {onset_loss}")
            print(f"Batch {i} action loss: {action_loss}")
            print(f"Batch {i} onset accuracy: {onset_acc}")
            print(f"Batch {i} action accuracy: {action_acc}")

    def val_one_epoch(self, epoch_index):
        print(colored(f"######### EPOCH {epoch_index} VALIDATI #########", "blue"))

        self.model.eval()
        running_vloss = 0.0

        with torch.inference_mode():
            for i, vdata in enumerate(self.val_loader):
                mels, beat_fracs, beat_nums, onsets, actions = vdata
                if self.cuda:
                    mels = mels.to("cuda")
                    beat_fracs = beat_fracs.to("cuda")
                    beat_nums = beat_nums.to("cuda")
                    onsets = onsets.to("cuda")
                    actions = actions.to("cuda")

                onset_preds, action_preds = self.model(mels, beat_fracs, beat_nums)

                # Flatten across all batches into single mega tensor to calculate loss
                onset_preds = torch.reshape(onset_preds, [-1])
                onsets = torch.reshape(onsets, [-1])
                # softmax applied to convert probabilites for all possible actions
                action_preds = torch.reshape(action_preds, [-1, action_preds.shape[-1]]).softmax(-1) # batch, timestep, num_combo -> batch * timestep, num_combo
                actions = torch.reshape(actions, [-1])

                # Create tensor where each value is the probablity of the correct action for loss func
                action_preds = action_preds[torch.arange(len(action_preds)), actions]
                # Create alpha mask, devaluing when no actions exist
                action_alphas = torch.where(actions == 0, self.alpha, 1)

                onset_vloss = FocalLosses.focal_loss(onsets, onset_preds, self.gamma, self.alpha)
                action_vloss = FocalLosses.action_focal_loss(action_preds, self.gamma, action_alphas)

                running_vloss += (onset_vloss + action_vloss)

                onset_vacc = (onset_preds == onsets).float().mean()
                action_vacc = (action_preds.argmax(dim=-1) == actions).float().mean()

                print(f"Batch {i} onset vloss: {onset_vloss}")
                print(f"Batch {i} action vloss: {action_vloss}")
                print(f"Batch {i} onset vaccuracy: {onset_vacc}")
                print(f"Batch {i} actionv accuracy: {action_vacc}")

        return (running_vloss / (i + 1))
    
    def train(self):
        if self.cuda:
            self.model.to("cuda")

        
        for epoch in self.num_epochs:
            self.train_one_epoch(epoch)
            avg_vloss = round(self.val_one_epoch(epoch), 4)

            save_fn = f"model_{epoch}_{avg_vloss}.pt"
            torch.save(self.model.state_dict(), self.checkpoint_path / save_fn)


class FocalLosses:
    def focal_loss(y_actual, y_pred, gamma, alpha):
        return -(torch.mean(y_actual * (1 - y_pred) ** gamma * torch.log(y_pred) + 
                            (1 - y_actual) * alpha * (y_pred) ** gamma * torch.log(1 - y_pred)))
    
    def action_focal_loss(y_pred, gamma, alpha):
        return -(torch.mean(alpha * (1 - y_pred) ** gamma * torch.log(y_pred)))