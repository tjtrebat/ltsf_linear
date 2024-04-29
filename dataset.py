import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ApplianceEnergyUsageDataset(Dataset):
    def __init__(self, energy_data_file, sequence_length, prediction_length):
        super().__init__()
        self.energy_data = pd.read_csv(energy_data_file)
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

    def __getitem__(self, index):
        sequence_begin = index
        sequence_end = sequence_begin + self.sequence_length
        x = self.energy_data.iloc[sequence_begin:sequence_end, 1:3].values
        prediction_begin = sequence_end
        prediction_end = prediction_begin + self.prediction_length
        y = self.energy_data.iloc[prediction_begin:prediction_end, 1:3].values
        return x, y

    def __len__(self):
        return len(self.energy_data) - self.sequence_length - self.prediction_length + 1
    
energy_data_file = '../../datasets/appliances_energy_prediction/energydata_complete.csv'
sequence_length = 1
prediction_length = 2
dataset = ApplianceEnergyUsageDataset(energy_data_file, 
                                      sequence_length, 
                                      prediction_length)
print(len(dataset))
x, y = dataset[len(dataset) - 1]
print('Sequence:')
print(x)
print('\nPrediction:')
print(y)