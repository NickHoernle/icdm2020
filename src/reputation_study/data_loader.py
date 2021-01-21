import json
import torch
import os

from torch.utils import data


class ReputationDataset(data.Dataset):
    def __init__(self,
               data_path: str = '../data/reputation_data',
               dset_type: str = "train",
               threshold_achievement: int = 25
               ):

        super(ReputationDataset, self).__init__()

        with open("{}/data_indexes.json".format(data_path), 'r') as f:
            list_IDs = json.load(f)
            if dset_type == "all":
                self.list_IDs = list_IDs["train"]+list_IDs["test"]+list_IDs["validate"]
            else:
                self.list_IDs = list_IDs[dset_type]

        self.threshold_achievement = threshold_achievement
        self.data_path = data_path

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        id = self.list_IDs[index]
        x = torch.load(os.path.join(self.data_path, f'user_{id}.pt'))
        # return x[0, :]+x[1, :]+x[2, :], x[-1, :]
        # return x[0, :], x[-1, :]
        # return x[1, :], x[-1, :]
        return x[2, :], x[-1, :]


class ReputationDatasetAllActions(ReputationDataset):
    def __getitem__(self, index):
        id = self.list_IDs[index]
        x = torch.load(os.path.join(self.data_path, f'user_{id}.pt'))
        return x[0, :], x[1, :], x[2, :], x[-1, :]