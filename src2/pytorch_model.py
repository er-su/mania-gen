import torch
from torch import nn
import numpy as np

hyperparams = {
      "batch_size": 16,
      "num_epochs": 1,
      "action_emb_dim": 32,
      "beat_frac_embd_dim": 32,
      "beat_num_embd_dim": 8,
      "onset_gru_output_size": 128,
      "action_gru_output_size": 128,
      "onset_as_input_size": 128,
      "onset_dense_size": 128,
      "action_dense_size": 128,
      "num_gru_layers": 2
}

class OsuGen(nn.Module):
    def __init__(self, hyperparams, num_keys=4, difficulty=(4,5)):
        super().__init__()
        self.bf_embd_dim = hyperparams["beat_frac_embd_dim"]
        self.bn_embd_dim = hyperparams["beat_num_embd_dim"]
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
        
        self.action_GRU = nn.GRU(input_size=(320 + self.onset_gru_size + self.bf_embd_dim + self.bn_embd_dim),
                                 hidden_size=self.action_gru_size,
                                 num_layers=self.num_gru_layers,
                                 batch_first=True)
        self.action_dense = nn.Sequential(
            nn.Linear(in_features=self.action_gru_size,
                      out_features=self.action_dense_size),
            nn.GELU(),
            nn.Linear(in_features=self.action_dense_size,
                      out_features=self.num_combos),
            #nn.LogSoftmax(dim=-1)
        )

        self.beat_frac_embd = nn.Embedding(49, self.bf_embd_dim)
        self.beat_num_embd = nn.Embedding(4, self.bn_embd_dim)

    def forward(self, mel, beat_frac, beat_num):
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
        onsets_as_input = self.onset_as_input(onset_gru_out)
        action_in = torch.cat([conv_flat, bf_embd, bn_embd, onsets_as_input], dim=-1)
        action_gru_out, _ = self.action_GRU(action_in)
        actions = self.action_dense(action_gru_out)
        actions = actions.squeeze()

        return onsets, actions
    
    


if __name__ == "__main__":
    features = np.load("postprocess/1654907 Hoshimachi Suisei - Kakero/audio_features.npy", allow_pickle=True).item()
    labels = np.load("postprocess/1654907 Hoshimachi Suisei - Kakero/4k/beatmap_5_a.npy", allow_pickle=True).item()
    actions = torch.as_tensor([labels["actions"]])
    print(f"Actions shape is: {actions.shape}")
    actionsgt = actions[:,1:]
    print(f"Actionsgt shape is: {actionsgt.shape}")

    mel = np.array([features["mel"]])
    beat_frac = np.array([features["beat_frac"]])
    beat_num = np.array([features["beat_num"]])
    """ print(mel.shape)
    print(beat_frac.shape)
    print(beat_num.shape) """
    mel = torch.from_numpy(mel)
    beat_frac = torch.from_numpy(beat_frac)
    beat_num = torch.from_numpy(beat_num)
    model = OsuGen(hyperparams)
    
    onset_out, action_out = model.forward(mel, beat_frac, beat_num)

    ns_pred = torch.reshape(action_out, [-1, action_out.shape[-1]]).softmax(dim=-1)
    print(f"Action reshape: {ns_pred.shape}")
    ns_label = torch.reshape(actions, [-1])
    print(f"Action labels shape: {ns_label.shape}")

    p_y = ns_pred[torch.arange(len(ns_pred)), ns_label]
    print(f"shape of p_y is {p_y.shape}")
    weight_mask = torch.where(ns_label == 0, 0.5, 1)

    print(f"Weight mask dim: {weight_mask.shape}")
