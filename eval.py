# -*- coding:utf-8 -*-
"""Eval code
"""
import numpy as np
import pandas as pd
import argparse
import torch as th
import torch.nn as nn
from Model.sch_qm import SchNetModel
from torch.utils.data import DataLoader
from Model.Alchemy_dataset_qm import TencentAlchemyDataset, batcher
from torch.nn import init



# def dataset_split(file):
#     delaney = pd.read_csv("delaney.csv")
#     test_set = delaney.sample(frac=0.1, random_state=0)
#     train_set = delaney.drop(test_set.index)
#     test_set.to_csv("delaney_test.csv", index=False)
#     train_set.to_csv("delaney_train.csv", index=False)


def eval(model="sch_qm", saved_model='', device=th.device("cpu"), eval_dataset=''):
    print("start")
    test_dataset = TencentAlchemyDataset()
    test_file = eval_dataset
    test_dataset.mode = "Train"
    test_dataset.transform = None
    test_dataset.file_path = test_file
    test_dataset._load()

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=0,
    )

    if model == "sch_qm":
        model = SchNetModel(norm=False, output_dim=1)
    print(model)
    # if model.name in ["SchNet"]:
    #     model.set_mean_std(mean, std, device)
    model.load_state_dict(th.load(saved_model))
    model.to(device)

    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    # optimizer = th.optim.Adam(model.parameters(), lr=0.0001)

    val_loss, val_mae = 0, 0
    res_file = open(eval_dataset+"_res.txt", 'w')
    for jdx, batch in enumerate(test_loader):

        batch.graph.to(device)
        batch.label = batch.label.to(device)
        batch.mask = batch.mask.to(device)
        batch.qm = batch.qm.to(device)

        res = model(batch.graph, batch.mask, batch.qm)

        res_np = res.cpu().detach().numpy()
        label_np = batch.label.cpu().detach().numpy()
        for i in range(len(res_np)):            
            res_file.write(str(label_np[i][0])+'\n')
            res_file.write(str(res_np[i][0]) + '\t')

        loss = loss_fn(res, batch.label)
        mae = MAE_fn(res, batch.label)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        val_mae += mae.detach().item()
        val_loss += loss.detach().item()
    val_mae /= jdx + 1
    val_loss /= jdx + 1
    print(
        " val_loss: {:.7f}, val_mae: {:.7f}".format(
            val_loss, val_mae
        ))
    print("test_dataset.mean= %s" % (test_dataset.mean))
    print("test_dataset.std= %s" % (test_dataset.std))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-M",
                        "--model",
                        help="model name (sch_qm)",
                        default="sch_qm")
    parser.add_argument("--saved_model", '-S',help="number of epochs", default=10000)
    parser.add_argument("--eval_dataset",'-E', help="dataset to train", default="")
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    assert args.model in ["sch_qm"]
    # dataset_split("delaney.csv")
    eval(args.model, args.saved_model, device, args.eval_dataset)
