import pandas as pd
from torch.utils.data import Dataset

class ApplianceEnergyUsageDataset(Dataset):
    def __init__(self, energy_data_file):
        super().__init__()
        self.energy_data = pd.read_csv(energy_data_file)

    def __getitem__(self, index):
        x = self.energy_data.iloc[index, 1:3]
        return x

    def __len__(self):
        return len(self.energy_data)
    

dataset = ApplianceEnergyUsageDataset('../../datasets/appliances_energy_prediction/energydata_complete.csv')
print(dataset)
print(len(dataset))
