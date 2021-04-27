import json
import torch
import os

from torch.utils import data


class ReputationDataset(data.Dataset):
    def __init__(self,
               data_path: str = '../data/reputation_data',
               dset_type: str = "train",
               threshold_achievement: int = 25,
               subsample: bool=True
               ):

        super(ReputationDataset, self).__init__()

        with open("{}/data_indexes.json".format(data_path), 'r') as f:
            list_IDs = json.load(f)
            if dset_type == "all":
                self.list_IDs = list_IDs["train"]+list_IDs["test"]+list_IDs["validate"]
            else:
                self.list_IDs = list_IDs[dset_type]

        if subsample:
            self.list_IDs = self.list_IDs[:10000]

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
        reputation = x[-1, :]
        return x[0], x[1], x[2]#, (reputation > 1).astype(float)

    @property
    def activity_names(self):
        return ['Question', 'Answer', 'Edit']


class ElectorateDatasetAllActions(ReputationDataset):

    def __init__(self,
               data_path: str = '../data/pt_electorate_data',
               dset_type: str = "train",
               threshold_achievement: int = 25,
               **kwargs
               ):

        super(ElectorateDatasetAllActions, self).__init__(data_path=data_path,
                                                          dset_type=dset_type,
                                                          threshold_achievement=threshold_achievement,
                                                          **kwargs)

    @property
    def activity_names(self):
        return ['Answers', 'Questions', 'Comments', 'Edits', 'AnswerVotes', 'QuestionVotes', 'ReviewTasks']

    def __getitem__(self, index):
        id = self.list_IDs[index]
        x = torch.load(os.path.join(self.data_path, f'user_{id}.pt'))
        return [v.squeeze(dim=0) for v in x.split(1, dim=0)]


class StrunkWhiteDatasetAllActions(ElectorateDatasetAllActions):
    def __init__(self,
               data_path: str = '../data/pt_s',
               dset_type: str = "train",
               threshold_achievement: int = 25
               ):

        super(StrunkWhiteDatasetAllActions, self).__init__(data_path=data_path,
                                                          dset_type=dset_type,
                                                          threshold_achievement=threshold_achievement)
