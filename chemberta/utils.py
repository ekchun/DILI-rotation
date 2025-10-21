import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils
import torch.utils.data

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc(x)
        return x
    
class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, df, feat_dict, split_df, split):
        # super().__init__()
        # split_df = split_df[split_df['subset'] == split]
        # print(split_df['cmpd_id'].head())
        # self.df = df[df['index'] in split_df['index'].to_list()]
        
        split_data = df.merge(split_df, left_on='spid', right_on='cmpd_id')
        self.train_split = split_data[split_data['subset'] == split]

        # indices = split_df['index']
        self.feat = feat_dict

    def __len__(self):
        return len(self.train_split)
    
    def __getitem__(self, index):
        spid = self.train_split.iloc[index]['spid']
        x = self.feat[spid]
        
        y = self.train_split.iloc[index]['category_target']
        return x, y


