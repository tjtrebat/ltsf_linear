import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class ApplianceEnergyUsageDataset(Dataset):
    def __init__(self, energy_data, sequence_length, prediction_length):
        super().__init__()
        self.energy_data = energy_data
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length        

    def __getitem__(self, index):
        sequence_begin = index
        sequence_end = sequence_begin + self.sequence_length
        x = self.energy_data[sequence_begin:sequence_end]
        prediction_begin = sequence_end
        prediction_end = prediction_begin + self.prediction_length
        y = self.energy_data[prediction_begin:prediction_end]
        return x, y

    def __len__(self):
        return len(self.energy_data) - self.sequence_length - self.prediction_length + 1    


energy_data_file = '../../datasets/appliances_energy_prediction/energydata_complete.csv'
energy_data = pd.read_csv(energy_data_file, usecols=[1, 2])
training_samples = int(len(energy_data) * 0.8)
scaler = StandardScaler()
train_data = scaler.fit_transform(energy_data.iloc[:training_samples].values)

def get_train_dataset(sequence_length, prediction_length):
    dataset = ApplianceEnergyUsageDataset(train_data, 
                                          sequence_length, 
                                          prediction_length)
    return dataset

def get_test_dataset(sequence_length, prediction_length):
    test_data = scaler.transform(energy_data.iloc[
        training_samples - sequence_length:].values)
    dataset = ApplianceEnergyUsageDataset(test_data,
                                          sequence_length,
                                          prediction_length)
    return dataset
