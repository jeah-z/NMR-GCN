# -*- coding:utf-8 -*-
"""Sample training code
"""
import numpy as np
import pandas as pd
import argparse
import torch as th
import torch.nn as nn
from sch_qm import SchNetModel
from torch.utils.data import DataLoader
from Alchemy_dataset_qm import TencentAlchemyDataset, batcher
from torch.nn import init

def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform(m.weight)
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            #init.constant(m.weight, 0.1)
            # if m.bias:
            #     init.constant(m.bias, 0)

# def dataset_split(file):
#     delaney = pd.read_csv("delaney.csv")
#     test_set = delaney.sample(frac=0.1, random_state=0)
#     train_set = delaney.drop(test_set.index)
#     test_set.to_csv("delaney_test.csv", index=False)
#     train_set.to_csv("delaney_train.csv", index=False)


def train(model="sch", epochs=80, device=th.device("cpu"), train_file='', test_file='',save=''):
    print("start")
    # train_dir = "./"
    # train_file = dataset+"_train.csv"
    alchemy_dataset = TencentAlchemyDataset()
    alchemy_dataset.mode = "Train"
    alchemy_dataset.transform = None
    alchemy_dataset.file_path = train_file
    alchemy_dataset._load()

    test_dataset = TencentAlchemyDataset()
    # test_dir = train_dir
    # test_file = dataset+"_valid.csv"
    test_dataset.mode = "Train"
    test_dataset.transform = None
    test_dataset.file_path = test_file
    test_dataset._load()

    alchemy_loader = DataLoader(
        dataset=alchemy_dataset,
        batch_size=1,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=0,
    )
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
    # if model.name in ["MGCN", "SchNet"]:
    #     model.set_mean_std(alchemy_dataset.mean, alchemy_dataset.std, device)
    model.to(device)
    initNetParams(model)
    model.load_state_dict(th.load('./C_qm/model_100'))
    # print("test_dataset.mean= %s" % (alchemy_dataset.mean))
    # print("test_dataset.std= %s" % (alchemy_dataset.std))

    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    optimizer = th.optim.Adam(model.parameters(), lr=0.0001)
    #scheduler = th.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1, last_epoch=-1)


    for epoch in range(epochs):

        w_loss, w_mae = 0, 0
        model.train()
        #scheduler.step()
        for idx, batch in enumerate(alchemy_loader):
            batch.graph.to(device)
            batch.label = batch.label.to(device)
            batch.mask = batch.mask.to(device)
            batch.qm = batch.qm.to(device)

            res = model(batch.graph, batch.mask, batch.qm)
            loss = loss_fn(res, batch.label)
            mae = MAE_fn(res, batch.label)

            optimizer.zero_grad()
            mae.backward()
            optimizer.step()

            w_mae += mae.detach().item()
            w_loss += loss.detach().item()
        w_mae /= idx + 1
        w_loss /= idx + 1

        print("Epoch {:2d}, loss: {:.7f}, mae: {:.7f}".format(
            epoch, w_loss, w_mae))

        val_loss, val_mae = 0, 0
        for jdx, batch in enumerate(test_loader):
            batch.graph.to(device)
            batch.label = batch.label.to(device)
            batch.mask = batch.mask.to(device)
            batch.qm = batch.qm.to(device)

            res = model(batch.graph, batch.mask, batch.qm)
            loss = loss_fn(res, batch.label)
            mae = MAE_fn(res, batch.label)

            # optimizer.zero_grad()
            # mae.backward()
            # optimizer.step()

            val_mae += mae.detach().item()
            val_loss += loss.detach().item()
        val_mae /= jdx + 1
        val_loss /= jdx + 1
        print(
            "Epoch {:2d}, val_loss: {:.7f}, val_mae: {:.7f}".format(
                epoch, val_loss, val_mae
            ))

        if epoch % 20 == 0:
            #op_file = open('save/'+str(epoch)+'re')
            res_op = []
            th.save(model.state_dict(), save+'/model_'+str(epoch))
            for jdx, batch in enumerate(test_loader):
                batch.graph.to(device)
                batch.label = batch.label.to(device)
                batch.mask = batch.mask.to(device)
                batch.qm = batch.qm.to(device)
                res = model(batch.graph, batch.mask, batch.qm)
                res_op.append([batch.label.cpu().detach().numpy(), res.cpu().detach().numpy()])
            res_pd = pd.DataFrame(res_op)
            res_pd.to_csv(save+'/res_'+str(epoch)+'.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-M",
                        "--model",
                        help="model name (sch_qm)",
                        default="sch")
    parser.add_argument("--epochs", help="number of epochs", default=10000)
    parser.add_argument("--train_file", help="dataset to train", default="")
    parser.add_argument("--test_file", help="dataset to test", default="")
    parser.add_argument("--save", help="save option", default="")
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    assert args.model in ["sch_qm"]
    # dataset_split("delaney.csv")
    train(args.model, int(args.epochs), device, args.train_file, args.test_file, args.save)
