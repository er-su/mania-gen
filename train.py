import torch
from pathlib import Path
from termcolor import colored
from dataset import OsuDataset
from pytorch_model import OsuGen
from pytorch_model import hyperparams
from torch.utils.data import DataLoader, random_split

# Given a dataset, model, and hyperparameters, trains the model
class Trainer:
    def __init__(self, model: OsuGen, dataset: OsuDataset, hyperparams, checkpoint_path: Path=Path("checkpoints")):
        self.model = model
        self.dataset = dataset
        self.num_epochs = hyperparams["num_epochs"]
        self.batch_size = hyperparams["batch_size"]
        self.learning_rate = hyperparams["learning_rate"]
        self.checkpoint_path = checkpoint_path
        self.train_set, self.val_set = random_split(self.dataset, [0.95, 0.05])
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.onset_gamma = 2
        self.action_gamma = 3
        self.onset_alpha = 0.28
        self.action_alpha = 0.26
        self.pos_onset_alpha = 1.27
        self.pos_action_alpha = 1.28
        self.cuda = torch.cuda.is_available()
    
    # Training for a single epoch
    def train_one_epoch(self, epoch_index):

        print(colored(f"######### EPOCH {epoch_index} TRAINING #########", "blue"))

        self.model.train()
        for i, data in enumerate(self.train_loader):
            mels, beat_fracs, beat_nums, difficulties, onsets, actions = data
            actions_shift = actions[:,:-1]
            actions = actions[:,1:]
            if self.cuda:
                mels = mels.to("cuda")
                beat_fracs = beat_fracs.to("cuda")
                beat_nums = beat_nums.to("cuda")
                onsets = onsets.to("cuda")
                actions = actions.to("cuda")
                actions_shift = actions_shift.to("cuda")
            
            onset_preds, action_preds = self.model(mels, beat_fracs, beat_nums, actions_shift)

            # Flatten across all batches into single mega tensor to calculate loss
            onset_preds = torch.reshape(onset_preds, [-1])
            onsets = torch.reshape(onsets, [-1])
            # Softmax applied to convert to probabilites for all possible actions
            action_preds = torch.reshape(action_preds, [-1, action_preds.shape[-1]]).softmax(dim=-1) # batch, timestep, num_combo -> batch * timestep, num_combo
            actions = torch.reshape(actions, [-1])

            # Create tensor where each value is the probablity of the correct action for loss func
            action_preds = action_preds[torch.arange(len(action_preds)), actions]
            # Create alpha mask, devaluing when no actions exist
            action_alphas = torch.where(actions == 0, self.action_alpha, self.pos_action_alpha)

            onset_loss = FocalLosses.focal_loss(onsets, onset_preds, self.onset_gamma, self.onset_alpha, self.pos_onset_alpha)
            action_loss = FocalLosses.action_focal_loss(action_preds, self.action_gamma, action_alphas)

            self.optimizer.zero_grad()
            batch_loss = onset_loss * 5 + action_loss
            batch_loss.backward()
            self.optimizer.step()

            onset_acc = (onset_preds == onsets).float().mean()
            action_acc = (action_preds.argmax(dim=-1) == actions).float().mean()

            print(f"Batch {i} onset loss: {onset_loss}")
            print(f"Batch {i} action loss: {action_loss}")
            print(f"Batch {i} onset accuracy: {onset_acc}")
            print(f"Batch {i} action accuracy: {action_acc}")

    # Perform validation for a single epoch
    def val_one_epoch(self, epoch_index):
        print(colored(f"######### EPOCH {epoch_index} VALIDATI #########", "blue"))

        self.model.eval()
        running_vloss = 0.0

        with torch.inference_mode():
            for i, vdata in enumerate(self.val_loader):
                mels, beat_fracs, beat_nums, difficulties, onsets, actions = vdata
                actions_shift = actions[:,:-1]
                actions = actions[:,1:]
                if self.cuda:
                    mels = mels.to("cuda")
                    beat_fracs = beat_fracs.to("cuda")
                    beat_nums = beat_nums.to("cuda")
                    onsets = onsets.to("cuda")
                    actions = actions.to("cuda")
                    actions_shift = actions_shift.to("cuda")

                onset_preds, action_preds = self.model(mels, beat_fracs, beat_nums, actions_shift)

                # Flatten across all batches into single mega tensor to calculate loss
                onset_preds = torch.reshape(onset_preds, [-1])
                onsets = torch.reshape(onsets, [-1])
                # softmax applied to convert probabilites for all possible actions
                action_preds = torch.reshape(action_preds, [-1, action_preds.shape[-1]]).softmax(-1) # batch, timestep, num_combo -> batch * timestep, num_combo
                actions = torch.reshape(actions, [-1])

                # Create tensor where each value is the probablity of the correct action for loss func
                action_preds = action_preds[torch.arange(len(action_preds)), actions]
                # Create alpha mask, devaluing when no actions exist
                action_alphas = torch.where(actions == 0, self.action_alpha, self.pos_action_alpha)

                onset_vloss = FocalLosses.focal_loss(onsets, onset_preds, self.onset_gamma, self.onset_alpha, self.pos_onset_alpha)
                action_vloss = FocalLosses.action_focal_loss(action_preds, self.action_gamma, action_alphas)

                running_vloss += (onset_vloss.item() + action_vloss.item())

                onset_vacc = (onset_preds >= onsets).float().mean()
                action_vacc = (action_preds.argmax(dim=-1) == actions).float().mean()

                print(f"Batch {i} onset vloss: {onset_vloss}")
                print(f"Batch {i} action vloss: {action_vloss}")
                print(f"Batch {i} onset vaccuracy: {onset_vacc}")
                print(f"Batch {i} actionv accuracy: {action_vacc}")

        return (running_vloss / (i + 1))
    
    # Train a brand new model
    def train(self):
        if self.cuda:
            self.model.to("cuda")

        
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            avg_vloss = round(self.val_one_epoch(epoch), 4)

            if epoch % 20 == 0:
                save_fn = f"new_model_{epoch}_{avg_vloss}.pt"
                torch.save(self.model.state_dict(), self.checkpoint_path / save_fn)

    # Pass in a pre-existing model from a checkpoint to train from
    def train_from_checkpoint(self, checkpoint_file: Path):
        self.model.load_state_dict(torch.load(checkpoint_file))
        if self.cuda:
            self.model.to("cuda")

        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            avg_vloss = round(self.val_one_epoch(epoch), 4)

            if epoch % 20 == 0:
                save_fn = f"model_{epoch}_{avg_vloss}.pt"
                torch.save(self.model.state_dict(), self.checkpoint_path / save_fn)

# Define focal losses for both onsets and actions
# Based on paper "Focal loss for dense object detection"
# by Lin, T.Y., et al
class FocalLosses:
    def focal_loss(y_actual, y_pred, gamma, alpha, pos_alpha):
        return -((y_actual * pos_alpha * (1 - y_pred) ** gamma * torch.log(y_pred) + 
                            (1 - y_actual) * alpha * (y_pred) ** gamma * torch.log(1 - y_pred))).mean()
    
    def action_focal_loss(y_pred, gamma, alpha):
        return -((alpha * (1 - y_pred) ** gamma * torch.log(y_pred))).mean()
    
if __name__ == "__main__":
    print(f"Running with cuda: {torch.cuda.is_available()}")
    model = OsuGen(hyperparams=hyperparams, difficulty=(3,4,5))
    train_from = Path("checkpoints/V1.pt")
    # Train using also preexisting dataset and dataloader 
    # Can be found at https://drive.google.com/drive/folders/1wNUPNz9u28aUMQuqA6e9-OwSxL_SJ6qw?usp=sharing
    dataset = OsuDataset2(Path("beatmap/4keys"), Path("audio"))
    checkpoint_path = Path("checkpoints")

    trainer = Trainer(model, dataset, hyperparams, checkpoint_path)
    trainer.train_from_checkpoint(train_from)