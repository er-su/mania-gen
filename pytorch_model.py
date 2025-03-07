import torch
import numpy as np
from torch import nn

hyperparams = {
      "batch_size": 32,
      "num_epochs": 181,
      "action_embd_dim": 32,
      "beat_frac_embd_dim": 32,
      "beat_num_embd_dim": 16,
      "onset_gru_output_size": 256,
      "action_gru_output_size": 256,
      "onset_as_input_size": 128,
      "onset_dense_size": 128,
      "action_dense_size": 128,
      "num_gru_layers": 2,
      "learning_rate": 0.0008
}

class OsuGen(nn.Module):
    def __init__(self, hyperparams, num_keys=4, difficulty=(4,5)):
        super().__init__()
        self.bf_embd_dim = hyperparams["beat_frac_embd_dim"]
        self.bn_embd_dim = hyperparams["beat_num_embd_dim"]
        self.action_embd_dim = hyperparams["action_embd_dim"]
        self.onset_gru_size = hyperparams["onset_gru_output_size"]
        self.action_gru_size = hyperparams["action_gru_output_size"]
        self.onset_as_input = hyperparams["onset_as_input_size"]
        self.onset_dense_size = hyperparams["onset_dense_size"]
        self.action_dense_size = hyperparams["action_dense_size"]
        self.batch_size = hyperparams["batch_size"]
        self.num_gru_layers = hyperparams["num_gru_layers"]
        self.num_combos = 4 ** num_keys
        self.CUDA = torch.cuda.is_available()
        
        self.convo = nn.Sequential(
            nn.Conv2d(3, 8, (5,3), stride=(1,2), padding=(2,1)),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 16, (5,3), stride=(1,2), padding=(2,1)),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, (5,3), stride=(1,2), padding=(2,1)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, (5,3), stride=(1,2), padding=(2,1)),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        self.onset_GRU = nn.GRU(input_size=(320 + self.bf_embd_dim + self.bn_embd_dim),
                                hidden_size=self.onset_gru_size,
                                num_layers=self.num_gru_layers,
                                batch_first=True,
                                bidirectional=True)
        
        self.onset_dense = nn.Sequential(
            nn.Linear(in_features=self.onset_gru_size * 2, out_features=self.onset_dense_size),
            nn.GELU(),
            nn.Linear(in_features=self.onset_dense_size, out_features=1),
            nn.Sigmoid()
        )

        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.onset_as_input = nn.Sequential(nn.Linear(self.action_gru_size * 2, self.action_dense_size), nn.GELU())
        
        self.action_GRU = nn.GRU(input_size=(320 + self.action_dense_size + self.bf_embd_dim + self.bn_embd_dim + self.action_embd_dim),
                                 hidden_size=self.action_gru_size,
                                 num_layers=self.num_gru_layers,
                                 batch_first=True)
        
        self.action_dense = nn.Sequential(
            nn.Linear(in_features=self.action_gru_size,
                      out_features=self.action_dense_size),
            nn.GELU(),
            nn.Linear(in_features=self.action_dense_size,
                      out_features=self.num_combos),
        )

        self.beat_frac_embd = nn.Embedding(49, self.bf_embd_dim)
        self.beat_num_embd = nn.Embedding(4, self.bn_embd_dim)
        self.action_embd = nn.Embedding(self.num_combos, self.action_embd_dim)

    def forward(self, mel, beat_frac, beat_num, actions):
        # Onsets
        conv_output = self.convo(mel)
        conv_flat = self.flatten(conv_output.permute(0, 2, 1, 3))

        bf_embd = self.beat_frac_embd(beat_frac)
        bn_embd = self.beat_num_embd(beat_num)

        onset_in = torch.cat([conv_flat, bf_embd, bn_embd], dim=-1)

        onset_gru_out, _ = self.onset_GRU(onset_in)
        onsets = self.onset_dense(onset_gru_out)
        onsets = onsets.squeeze()

        # Find actions
        action_embd = self.action_embd(actions)
        onsets_as_input = self.onset_as_input(onset_gru_out)
        action_in = torch.cat([conv_flat, bf_embd, bn_embd, onsets_as_input, action_embd], dim=-1)

        action_gru_out, _ = self.action_GRU(action_in)
        actions = self.action_dense(action_gru_out)
        actions = actions.squeeze()

        return onsets, actions

    # Infer by selecting the most likely output
    def old_infer(self, mels, beat_frac, beat_num):
        # Onsets
        conv_output = self.convo(mels)
        conv_flat = self.flatten(conv_output.permute(0, 2, 1, 3))

        bf_embd = self.beat_frac_embd(beat_frac)
        bn_embd = self.beat_num_embd(beat_num)

        onset_in = torch.cat([conv_flat, bf_embd, bn_embd], dim=-1)

        onset_gru_out, _ = self.onset_GRU(onset_in)
        onsets_as_input = self.onset_as_input(onset_gru_out)

        final_out = torch.zeros([mels.shape[0], mels.shape[2]])
        initial_action_embd = self.action_embd(torch.zeros([mels.shape[0], 1], dtype=torch.long))
        prev = torch.zeros([2, mels.shape[0], self.action_gru_size])
        concat_in = torch.concat([conv_flat, bf_embd, bn_embd, onsets_as_input], dim=-1)

        for i in range(mels.shape[2]):
            full_in = torch.cat([concat_in[:, i:i+1], initial_action_embd], dim=-1)
            action_gru_out, prev = self.action_GRU(full_in, prev)
            actions = self.action_dense(action_gru_out)
            pred = actions.argmax(dim=-1)
            final_out[:,i] = pred
            initial_action_embd = self.action_embd(pred)

        return final_out.squeeze()
    
    # Infer by probabalistically selecting the top 4 predictions
    def infer(self, mels, beat_frac, beat_num):
        # Onsets
        conv_output = self.convo(mels)
        conv_flat = self.flatten(conv_output.permute(0, 2, 1, 3))

        bf_embd = self.beat_frac_embd(beat_frac)
        bn_embd = self.beat_num_embd(beat_num)

        onset_in = torch.cat([conv_flat, bf_embd, bn_embd], dim=-1)

        onset_gru_out, _ = self.onset_GRU(onset_in)
        onsets_as_input = self.onset_as_input(onset_gru_out)

        final_out = torch.zeros([mels.shape[0], mels.shape[2]])
        initial_action_embd = self.action_embd(torch.zeros([mels.shape[0], 1], dtype=torch.long))
        prev = torch.zeros([2, mels.shape[0], self.action_gru_size])
        concat_in = torch.concat([conv_flat, bf_embd, bn_embd, onsets_as_input], dim=-1)

        for i in range(mels.shape[2]):
            full_in = torch.cat([concat_in[:, i:i+1], initial_action_embd], dim=-1)
            action_gru_out, prev = self.action_GRU(full_in, prev)
            actions = self.action_dense(action_gru_out)
            probs = nn.functional.softmax(actions, dim=-1)  # Convert logits to probabilities

            # Extract top 4 only if certainty is less than 35%
            topk = torch.topk(probs, 4)
            prepred = actions.argmax(dim=-1)
            if prepred.item() > 0 and probs[:,:,prepred] < 0.35:
                pred = torch.multinomial(topk[0].squeeze(1), num_samples=1)
                pred = topk[1][:,:,pred].squeeze(dim=0).squeeze(dim=0)
            else:
                pred = prepred

            final_out[:,i] = pred
            initial_action_embd = self.action_embd(pred)

        return final_out.squeeze()
    


if __name__ == "__main__":
    features = np.load("postprocess/1654907 Hoshimachi Suisei - Kakero/audio_features.npy", allow_pickle=True).item()
    labels = np.load("postprocess/1654907 Hoshimachi Suisei - Kakero/4k/beatmap_5_a.npy", allow_pickle=True).item()
    
    actions = torch.as_tensor([labels["actions"]])
    mel = np.array([features["mel"]])
    beat_frac = np.array([features["beat_frac"]])
    beat_num = np.array([features["beat_num"]])
    
    mel = torch.from_numpy(mel)
    beat_frac = torch.from_numpy(beat_frac)
    beat_num = torch.from_numpy(beat_num)
    model = OsuGen(hyperparams)
    
    onset_out, action_out = model.forward(mel, beat_frac, beat_num)