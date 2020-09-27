# NMR-GCN
A code to predict atomic NMR chemical shift based GCN and QM calculation


This code was based on https://github.com/tencent-alchemy/Alchemy. If this script is of any help to you, please cite them.

- K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.
SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017) [link](http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions)  
```
- @article{chen2019alchemy,
  title={Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models},
  author={Chen, Guangyong and Chen, Pengfei and Hsieh, Chang-Yu and Lee, Chee-Kong and Liao, Benben and Liao, Renjie and Liu, Weiwen and Qiu, Jiezhong and Sun, Qiming and Tang, Jie and Zemel, Richard and Zhang, Shengyu},
  journal={arXiv preprint arXiv:1906.09427},
  year={2019}
}
```
## Dependencies  

+ PyTorch 
+ dgl 
+ RDKit
+ Numpy
+ Pandas
+ argparse

## Usage:

### How to preprocess the dataset:
`python dataset_split.py --dataset ./DATA/C_NMR/C-NMR`


### How to train the model: 
`python train_qm.py --model sch_qm --epochs 10000 --train_file ./DATA/C_NMR/C-NMR_train.csv --test_file ./DATA/C_NMR/C-NMR_valid.csv --save saved_model/C-C`

### How to eval the model: 
`python eval.py -M sch_qm -S saved_model/N-N/model_200 -E DATA/N_NMR/N-NMR_valid.csv`





