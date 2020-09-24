import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem


validation_split = .15
shuffle_dataset = True
random_seed = 42
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset to train")
args = parser.parse_args()
dataset = args.dataset


delaney = pd.read_csv(dataset + ".csv", skiprows=1,
                      names=['SMILES', 'id', 'QM', 'Reserve1', 'NMR', 'Reserve2'])
dataset_size = len(delaney)
invalid_id = []
for i in range(dataset_size):
    smi = delaney.loc[i]['SMILES']
    try:
        mol = Chem.MolFromSmiles(smi)
        AllChem.Compute2DCoords(mol)

    except:
        print(smi + "was not valid SMILES\n")
        invalid_id.append(i)
delaney.drop(labels=invalid_id, axis=0)

# Creating data indices for training and validation splits:
dataset_size = len(delaney)
print('dataset_size= %s' % (dataset_size))
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
# print(train_indices)

# Creating PT data samplers and loaders:
train_sampler = delaney.loc[train_indices]
train_QM_mean = train_sampler['QM'].mean()
train_QM_std = train_sampler['QM'].std()
train_sampler['QM'] = (
    train_sampler['QM'] - train_QM_mean) / train_QM_std

valid_sampler = delaney.loc[val_indices]
valid_sampler['QM'] = (valid_sampler['QM']-train_QM_mean)/train_QM_std

train_NMR_mean = train_sampler['NMR'].mean()
train_NMR_std = train_sampler['NMR'].std()
train_sampler['NMR'] = (
    train_sampler['NMR'] - train_NMR_mean) / train_NMR_std

valid_sampler['NMR'] = (valid_sampler['NMR']-train_NMR_mean)/train_NMR_std

mean_file = open(dataset + '_mean_std.txt', 'w')
mean_file.writelines('train_QM_mean= %s\n' % (train_QM_mean))
mean_file.writelines('train_QM_std= %s\n' % (train_QM_std))
mean_file.writelines('train_NMR_mean= %s\n' % (train_NMR_mean))
mean_file.writelines('train_NMR_std= %s' % (train_NMR_std))


train_sampler.to_csv(dataset+"_train.csv", index=False)
valid_sampler.to_csv(dataset+"_valid.csv", index=False)
