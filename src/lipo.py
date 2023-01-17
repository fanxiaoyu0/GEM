import numpy as np
import pandas as pd
import json
import sys
import pickle as pkl
import random
from tqdm import tqdm
# import time
# import matplotlib.pyplot as plt
# import pdb
import os
from threading import Thread, Lock
from rdkit.Chem import AllChem

import paddle as pdl
import paddle.nn as nn
from paddle import optimizer 
import pgl
from pgl.utils.data import Dataset
from pgl.utils.data.dataloader import Dataloader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, '../lib/PaddleHelix/')
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_MMFF3d
from pahelix.model_zoo.gem_model import GeoGNNModel

pdl.seed(1024)
np.random.seed(1024)
random.seed(1024)
mutex = Lock()

def get_smiles_list_pkl():
    dataset_df = pd.read_csv("../data/lipo/raw/dataset.csv")
    smiles_list = dataset_df["smiles"].tolist()
    smiles_list = list(set(smiles_list))
    print(len(smiles_list))
    pkl.dump(smiles_list, open('../data/lipo/intermediate/smiles_list.pkl', 'wb'))

def calculate_3D_structure_(smiles_list):
    n = len(smiles_list)
    global p
    index = 0
    while True:
        mutex.acquire()
        if p >= n:
            mutex.release()
            break
        index = p
        p += 1
        mutex.release()

        smiles = smiles_list[index]
        print(index, ':', round(index / n * 100, 2), '%', smiles)
        try:
            molecule = AllChem.MolFromSmiles(smiles)
            molecule_graph = mol_to_geognn_graph_data_MMFF3d(molecule)
        except:
            print("Invalid smiles!", smiles)
            mutex.acquire()
            with open('../data/lipo/result/invalid_smiles.txt', 'a') as f:
                f.write(str(smiles) + '\n')
            mutex.release()
            continue

        global smiles_to_graph_dict
        mutex.acquire()
        smiles_to_graph_dict[smiles] = molecule_graph
        mutex.release()   

def calculate_3D_structure():
    smiles_list_unique = pkl.load(open('../data/lipo/intermediate/smiles_list.pkl', 'rb'))
    global smiles_to_graph_dict
    smiles_to_graph_dict = {}
    global p
    p = 0
    thread_count = 16
    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=calculate_3D_structure_, args=(smiles_list_unique, )))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pkl.dump(smiles_to_graph_dict, open('../data/lipo/intermediate/smiles_to_graph_dict.pkl', 'wb'))
    print('Valid smiles count:', len(smiles_to_graph_dict))

def construct_data_list():
    dataset_df = pd.read_csv("../data/lipo/raw/dataset.csv")
    smiles_to_graph_dict:dict = pkl.load(open('../data/lipo/intermediate/smiles_to_graph_dict.pkl','rb'))
    
    data_list_train = []
    data_list_validate = []
    data_list_test = []

    for index, row in dataset_df.iterrows():
        smiles = row["smiles"]
        label = row["label"]
        dataset_type = row["dataset_type"]
        if smiles not in smiles_to_graph_dict:
            continue
        data_item = {
            "smiles": smiles,
            "graph": smiles_to_graph_dict[smiles],
            "label": label
        }
        if dataset_type == "train":
            data_list_train.append(data_item)
        elif dataset_type == "valid":
            data_list_validate.append(data_item)
        elif dataset_type == "test":
            data_list_test.append(data_item)

    pkl.dump(data_list_train, open('../data/lipo/intermediate/data_list_train.pkl', 'wb'))
    pkl.dump(data_list_validate, open('../data/lipo/intermediate/data_list_validate.pkl', 'wb'))
    pkl.dump(data_list_test, open('../data/lipo/intermediate/data_list_test.pkl', 'wb'))

class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

def collate_fn(data_batch):
    atom_names = ["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic", "hybridization"]
    bond_names = ["bond_dir", "bond_type", "is_in_ring"]
    bond_float_names = ["bond_length"]
    bond_angle_float_names = ["bond_angle"]
    
    atom_bond_graph_list = []
    bond_angle_graph_list = []
    smiles_list = []
    label_list = []

    for data_item in data_batch:
        graph = data_item['graph']
        ab_g = pgl.Graph(
                num_nodes=len(graph[atom_names[0]]),
                edges=graph['edges'],
                node_feat={name: graph[name].reshape([-1, 1]) for name in atom_names},
                edge_feat={name: graph[name].reshape([-1, 1]) for name in bond_names + bond_float_names})
        ba_g = pgl.Graph(
                num_nodes=len(graph['edges']),
                edges=graph['BondAngleGraph_edges'],
                node_feat={},
                edge_feat={name: graph[name].reshape([-1, 1]) for name in bond_angle_float_names})
        atom_bond_graph_list.append(ab_g)
        bond_angle_graph_list.append(ba_g)
        smiles_list.append(data_item['smiles'])
        label_list.append(data_item['label'])

    atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
    bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
    # TODO: reshape due to pgl limitations on the shape
    def _flat_shapes(d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])
    _flat_shapes(atom_bond_graph.node_feat)
    _flat_shapes(atom_bond_graph.edge_feat)
    _flat_shapes(bond_angle_graph.node_feat)
    _flat_shapes(bond_angle_graph.edge_feat)

    return atom_bond_graph, bond_angle_graph, np.array(label_list, dtype=np.float32), smiles_list

def get_data_loader(batch_size):
    data_list_train = pkl.load(open('../data/lipo/intermediate/data_list_train.pkl', 'rb'))
    data_list_validate = pkl.load(open('../data/lipo/intermediate/data_list_validate.pkl', 'rb'))
    data_list_test = pkl.load(open('../data/lipo/intermediate/data_list_test.pkl', 'rb'))
    data_loader_train = Dataloader(MyDataset(data_list_train), batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
    data_loader_validate = Dataloader(MyDataset(data_list_validate), batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
    data_loader_test = Dataloader(MyDataset(data_list_test), batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
    return data_loader_train, data_loader_validate, data_loader_test

class ADMET(nn.Layer):
    def __init__(self):
        super(ADMET, self).__init__()
        compound_encoder_config = json.load(open('../lib/PaddleHelix/apps/pretrained_compound/ChemRL/GEM/model_configs/geognn_l8.json', 'r'))
        self.encoder = GeoGNNModel(compound_encoder_config)
        self.encoder.set_state_dict(pdl.load("../lib/PaddleHelix/apps/pretrained_compound/ChemRL/GEM/weight/regr.pdparams"))
        self.mlp = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, atom_bond_graph, bond_angle_graph):
        node_repr, edge_repr, graph_repr = self.encoder(atom_bond_graph.tensor(), bond_angle_graph.tensor())
        x = graph_repr
        x = self.mlp(x)
        return x

def evaluate(model, data_loader):
    model.eval()
    label_true = pdl.to_tensor([], dtype=pdl.float32, place=pdl.CUDAPlace(0))
    label_predict = pdl.to_tensor([], dtype=pdl.float32, place=pdl.CUDAPlace(0))
    for (atom_bond_graph, bond_angle_graph, label_true_batch, smiles_batch) in data_loader:
        label_predict_batch = model(atom_bond_graph, bond_angle_graph)
        label_true_batch = pdl.to_tensor(label_true_batch, dtype=pdl.float32, place=pdl.CUDAPlace(0)).unsqueeze(1)

        label_true = pdl.concat((label_true, label_true_batch.detach()), axis=0)
        label_predict = pdl.concat((label_predict, label_predict_batch.detach()), axis=0)
    
    label_predict = label_predict.cpu().numpy()
    label_true = label_true.cpu().numpy()
    rmse = round(np.sqrt(mean_squared_error(label_true, label_predict)), 4)
    mae = round(mean_absolute_error(label_true, label_predict), 4)
    r2 = round(r2_score(label_true, label_predict), 4)
    metric = {'rmse': rmse, 'mae': mae, 'r2': r2}
    return metric

def trial(model_version):
    data_loader_train, data_loader_validate, data_loader_test = get_data_loader(batch_size=256)

    model = ADMET()

    criterion = nn.MSELoss()
    scheduler = optimizer.lr.CosineAnnealingDecay(learning_rate=1e-3, T_max=15, eta_min=1e-6)
    opt = optimizer.Adam(scheduler, parameters=model.parameters(), weight_decay=1e-5)

    current_best_metric = 1e10
    max_bearable_epoch = 50
    current_best_epoch = 0
    for epoch in range(800):
        model.train()
        for (atom_bond_graph, bond_angle_graph, label_true_batch, smiles_batch) in data_loader_train:
            label_predict_batch = model(atom_bond_graph, bond_angle_graph)
            label_true_batch = pdl.to_tensor(label_true_batch, dtype=pdl.float32, place=pdl.CUDAPlace(0)).unsqueeze(1)
            loss = criterion(label_predict_batch, label_true_batch)
            # print(label_predict_batch)
            # print(label_true_batch)
            # pdb.set_trace()

            loss.backward()
            opt.step()
            opt.clear_grad()
        scheduler.step()

        metric_train = evaluate(model, data_loader_train)
        metric_validate = evaluate(model, data_loader_validate)
        metric_test = evaluate(model, data_loader_test)

        if metric_validate['rmse'] < current_best_metric:
            current_best_metric = metric_validate['rmse']
            current_best_epoch = epoch
            pdl.save(model.state_dict(), "../weight/lipo/" + model_version + ".pkl")
        print("=========================================================")
        print("Epoch", epoch)
        print("Train", metric_train)
        print("Validate", metric_validate)
        print("Test", metric_test)
        print('current_best_epoch', current_best_epoch, 'current_best_metric', current_best_metric)
        if epoch > current_best_epoch + max_bearable_epoch:
            break

if __name__ == '__main__':
    # get_smiles_list_pkl()
    # calculate_3D_structure()
    # construct_data_list()
    trial(model_version='1')
    print("All is well!")

