import torch
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    def __init__(self, data, transform=None, pair_mode="none", n_phone_classes=None):
        self.data = data
        self.transform = transform
        self.pair_mode = pair_mode
        self.n_phone_classes = n_phone_classes
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        for day in range(self.n_days):
            for trial in range(len(data[day]["sentenceDat"])):
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])

                if self.pair_mode == "none":
                    self.phone_seqs.append(data[day]["phonemes"][trial])
                    self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                else:
                    seq = data[day]["phonemes"][trial]
                    L = int(data[day]["phoneLens"][trial])
                    if self.pair_mode == "overlap":
                        if L >= 2:
                            pairs = []
                            for i in range(L - 1):
                                p1 = int(seq[i])
                                p2 = int(seq[i + 1])
                                bid = (p1 - 1) * self.n_phone_classes + (p2 - 1) + 1
                                pairs.append(bid)
                            self.phone_seqs.append(pairs)
                            self.phone_seq_lens.append(L - 1)
                        else:
                            self.phone_seqs.append([])
                            self.phone_seq_lens.append(0)
                    elif self.pair_mode == "nonoverlap":
                        pairs = []
                        for i in range(0, L - 1, 2):
                            p1 = int(seq[i])
                            p2 = int(seq[i + 1])
                            bid = (p1 - 1) * self.n_phone_classes + (p2 - 1) + 1
                            pairs.append(bid)
                        self.phone_seqs.append(pairs)
                        self.phone_seq_lens.append(len(pairs))
                self.days.append(day)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        return (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )
