import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, coalesce

from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits, class_rand_splits
from eval import *
from parse import parse_method, parser_add_main_args

from torch_geometric.utils import to_dense_adj
# from torchmetrics.classification import BinaryAUROC
from GraphAttacker import GraphAttacker
import networkx as nx

import os
import warnings
warnings.filterwarnings('ignore')

import uuid
unique_id = str(uuid.uuid4())


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_A_H_A(H, mode=1):
    if mode == 1:
        H = H - H.mean(dim=1, keepdim=True)
        H = F.normalize(H, p=2, dim=1)
        H = torch.mm(H, H.t())
        H.fill_diagonal_(0.)
        return H

    elif mode == 2:
        H_center = H.mean(dim=0, keepdim=True).detach()
        H = H - H.mean(dim=1, keepdim=True)
        H = F.normalize(H, p=2, dim=1)
        H_center = H_center - H_center.mean(dim=1, keepdim=True)
        H_center = F.normalize(H_center, p=2, dim=1)
        H = torch.mm(H, H_center.t())
        return H


def append_results_to_file(filename, gnn, dataset, defence_auc, defence_sigma, defence_k, defence_p, defence_limit, 
                           attack_1_auc_mean, attack_2_auc_mean, results_mean, 
                           attack_1_auc_std, attack_2_auc_std, results_std):

    if not os.path.exists(filename):
        with open(filename, "a") as f:
            f.write("gnn,dataset,defence_auc,defence_sigma,defence_k,defence_p,defence_limit,attack_1_auc_mean,attack_2_auc_mean,results_mean,attack_1_auc_std,attack_2_auc_std,results_std,attack_1_auc,attack_2_auc,results\n")

    attack_1_auc_mean_100 = attack_1_auc_mean * 100
    attack_2_auc_mean_100 = attack_2_auc_mean * 100
    attack_1_auc_std_100 = attack_1_auc_std * 100
    attack_2_auc_std_100 = attack_2_auc_std * 100

    attack_1_auc = f"{attack_1_auc_mean_100:.2f} ± {attack_1_auc_std_100:.2f}"
    attack_2_auc = f"{attack_2_auc_mean_100:.2f} ± {attack_2_auc_std_100:.2f}"
    results = f"{results_mean:.2f} ± {results_std:.2f}"

    line = f"{gnn},{dataset},{defence_auc:.2f},{defence_sigma:.2f},{defence_k:.2f},{defence_p:.2f},{defence_limit},{attack_1_auc_mean_100:.2f},{attack_2_auc_mean_100:.2f},{results_mean:.2f},{attack_1_auc_std_100:.2f},{attack_2_auc_std_100:.2f},{results_std:.2f},{attack_1_auc},{attack_2_auc},{results}\n"

    with open(filename, "a") as f:
        f.write(line)


def maximum_independent_set_from_edge_index(edge_index, num_nodes):
    print('Calculating maximum independent set...')
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_nodes)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.t().tolist())

    min_vertex_cover = nx.algorithms.approximation.min_weighted_vertex_cover(G, weight=1)
    max_indep_set = set(G.nodes) - min_vertex_cover
    max_indep_set = torch.tensor(list(max_indep_set), dtype=torch.long)
    print(f'Done, size: {len(max_indep_set)} / {num_nodes}')

    # G.remove_nodes_from(max_indep_set)
    # min_vertex_cover = nx.algorithms.approximation.min_weighted_vertex_cover(G, weight=1)
    # max_indep_set_2 = set(G.nodes) - min_vertex_cover
    # max_indep_set_2 = torch.tensor(list(max_indep_set_2), dtype=torch.long)
    # print(f'Done, size: {len(max_indep_set_2)} / {num_nodes}')

    # max_indep_set = nx.algorithms.approximation.maximum_independent_set(G)

    # return (max_indep_set, max_indep_set_2)
    return max_indep_set


### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

dataset.label = dataset.label.to(device)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

### Load method ###
model = parse_method(args, n, c, d, device)

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('questions'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

args.method = args.gnn
logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

# --------------------------------------------------

# dataset.label = dataset.label[torch.randperm(n).to(device)]

node_label = dataset.label.squeeze(1)
dataset.graph['node_label'] = node_label
dataset.graph['num_classes'] = c

# indices = torch.stack([torch.arange(n), torch.arange(n)]).to(device)
# values = torch.ones(n).to(device)
# A_I = torch.sparse_coo_tensor(indices, values, (n, n))
# A_I = A_I.to_sparse_csr()

# dataset.graph['node_feat'] = torch.randn(n, d).to(device) + dataset.graph['node_feat']

# --------------------------------------------------

device_auc = device
# device_auc = torch.device('cpu')

A_X_vec = dataset.graph['node_feat']
A_X_vec = F.normalize(A_X_vec, p=2, dim=1)
A_X_vec = torch.mm(A_X_vec, A_X_vec.t())
A_X_vec = torch.relu(A_X_vec)
A_X_vec.fill_diagonal_(0.)
A_X_vec = A_X_vec.view(-1).to('cpu')

A_vec = dataset.graph['edge_index']
A_vec = to_dense_adj(A_vec, max_num_nodes=n)[0]
A_vec.fill_diagonal_(0.)
A_vec = A_vec.view(-1).to('cpu')

A_Y_vec = (node_label.unsqueeze(0) == node_label.unsqueeze(1)).float()
A_Y_vec.fill_diagonal_(0.)
A_Y_vec = A_Y_vec.view(-1).to('cpu')

# max_indep_set = maximum_independent_set_from_edge_index(dataset.graph['edge_index'], n)
# max_indep_set = max_indep_set.to(device)
max_indep_set = None

attack_1_auc_all = []
attack_2_auc_all = []
attack_3_auc_all = []
attack_4_auc_all = []
attack_5_auc_all = []
attack_6_auc_all = []

# --------------------------------------------------

### Training loop ###
for run in range(args.runs):
    if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo', 'cora', 'citeseer', 'pubmed'):
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    best_test = float('-inf')

    best_attack_auc = float('-inf')
    best_attack_proj_auc = float('-inf')

    # if args.save_model:
    #     save_model(args, model, optimizer, run)

    # --------------------------------------------------

    # dataset.label = torch.randint(0, c, dataset.label.size()).to(device)
    # dataset.label = torch.zeros_like(dataset.label)

    # --------------------------------------------------

    display_step = 1000
    initial_finished = True

    for epoch in range(args.epochs):
        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
        logger.add_result(run, result[:-1])

        if args.defence_auc < 1.0:
            attacker = GraphAttacker(args=args, model=model, graph=dataset.graph, split_idx=split_idx, A_X_vec=A_X_vec, A_Y_vec=A_Y_vec, A_vec=A_vec, 
                                    lr=0.01, weight_decay=0.00000, epochs=5000, 
                                    device=device, device_auc=device_auc, max_indep_set=max_indep_set)
            attack_auc, _, _, _, _, _ = attacker.attack(target=1)
        else:
            attack_auc = args.defence_auc

        if (attack_auc > args.defence_auc) or ((not initial_finished) and (attack_auc > args.defence_auc_initial)):
            model.train()

            for _ in range(5):

                H_A, _ = model(dataset.graph['node_feat'], dataset.graph['edge_index'], get_embedding=True)
                A_H_A = get_A_H_A(H_A, mode=1)

                # --------------------------------------------------
                if args.defence_relu:
                    A_H_A = torch.relu(A_H_A)
                if args.defence_l1:
                    A_H_A = A_H_A.abs()
                if args.defence_l2:
                    A_H_A = A_H_A**2
                # --------------------------------------------------

                A_H_A_positive = A_H_A[dataset.graph['edge_index'][0], dataset.graph['edge_index'][1]]
                mean_positive = A_H_A_positive.mean()
                mean_all = A_H_A.mean()

                # --------------------------------------------------
                if args.defence_positive and not args.defence_all:
                    loss = 1 + mean_positive
                if not args.defence_positive and args.defence_all:
                    loss = 1 - mean_all
                if args.defence_positive and args.defence_all:
                    loss = 2 + mean_positive - mean_all
                if not args.defence_positive and not args.defence_all:
                    raise ValueError('Invalid defence mode')
                # --------------------------------------------------

                # A_H_A = get_A_H_A(H_A, mode=2)
                # # A_H_A = A_H_A**2
                # loss = A_H_A.mean() * 0.1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        else:
            if not initial_finished:
                initial_finished = True

            if result[1] > best_val:
                best_val = result[1]
                best_test = result[2]

                # if (not (args.defence_auc < 1.0)):
                #     attacker = GraphAttacker(args=args, model=model, graph=dataset.graph, split_idx=split_idx, A_X_vec=A_X_vec, A_Y_vec=A_Y_vec, A_vec=A_vec, 
                #                             lr=0.01, weight_decay=0.00000, epochs=5000, 
                #                             device=device, device_auc=device_auc, max_indep_set=max_indep_set)
                #     attack_auc, _, _, _, _, _ = attacker.attack(target=1)

                # best_attack_auc = attack_auc
                # best_attack_proj_auc, _, _, _, _, _ = attacker.attack(target=2)
                # best_attack_proj_auc = attack_auc

                if args.defence_visual:
                    save_model(args, model, optimizer, run, id=unique_id)

                if args.save_model:
                    save_model(args, model, optimizer, run, id=unique_id)
                    # model_state_dict = model.state_dict()

            model.train()
            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

            if args.dataset in ('questions'):
                if dataset.label.shape[1] == 1:
                    true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
                else:
                    true_label = dataset.label
                loss = criterion(out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))

            else:
                out = F.log_softmax(out, dim=1)
                loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --------------------------------------------------

            # A_H_A = get_A_H_A(H_A, mode=1)
            # A_H_A = A_H_A[dataset.graph['edge_index'][0], dataset.graph['edge_index'][1]]
            # A_H_A = torch.relu(A_H_A)

            # loss = A_H_A.mean() * 0.001

            # if epoch < 500:
            #     A_H_A = get_A_H_A(H_A, mode=2)
            #     # A_H_A = A_H_A.abs() + A_H_A**2
            #     # A_H_A = A_H_A.abs()
            #     # A_H_A = A_H_A**2
            #     loss = (1 - A_H_A.mean()) * 0.001
            #     # loss = A_H_A.mean() * 0.001

            # if epoch >= 500:
            #     A_H_A = get_A_H_A(H_A, mode=2)
            #     # A_H_A = A_H_A.abs() + A_H_A**2
            #     # A_H_A = A_H_A.abs()
            #     # A_H_A = A_H_A**2
            #     loss += (1 - A_H_A.mean()) * 10.0

            # if epoch >= 500:
            #     A_H_A = get_A_H_A(H_A, mode=2)
            #     A_H_A = A_H_A.abs()
            #     loss += A_H_A.mean() * 30.0

            # H_I, _ = model(dataset.graph['node_feat'], A_I, get_embedding=True)
            # H_I_center = H_I.mean(dim=0, keepdim=True).detach()

            # H_I = H_I - H_I.mean(dim=1, keepdim=True)
            # H_I_center = H_I_center - H_I_center.mean(dim=1, keepdim=True)

            # cosine_similarity = F.cosine_similarity(H_I, H_I_center, dim=1)

            # loss += -cosine_similarity.abs().mean() * 10.0

            # --------------------------------------------------

        # if epoch % args.display_step == 0:
        if epoch % display_step == 0:
            print(f'Epoch: {epoch:04d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * result[0]:6.2f}%, '
                f'Val: {100 * result[1]:5.2f}%, '
                f'Test: {100 * result[2]:5.2f}%, '
                f'B-Val: {100 * best_val:5.2f}%, '
                f'B-Test: {100 * best_test:5.2f}%')

            # print(f'Epoch: {epoch:04d}, '
            #     f'Loss: {loss:.4f}, '
            #     f'Train: {100 * result[0]:6.2f}%, '
            #     f'Val: {100 * result[1]:5.2f}%, '
            #     f'Test: {100 * result[2]:5.2f}%, '
            #     f'B-Val: {100 * best_val:5.2f}%, '
            #     f'B-Test: {100 * best_test:5.2f}%, ', end='')

            # print(f'AUC: {attack_auc:.8f}, '
            #     f'B-AUC: {best_attack_auc:.8f}, '
            #     f'B-PAUC: {best_attack_proj_auc:.8f}')

        # try:
        #     del attacker
        # except NameError:
        #     pass

        torch.cuda.empty_cache()

    print('--------------------------------------------------')
    logger.print_statistics(run)
    print('--------------------------------------------------')

    # --------------------------------------------------

    if args.save_model:
        model, _ = load_model(args, model, optimizer, run, id=unique_id)
        # model.load_state_dict(model_state_dict)

        attacker = GraphAttacker(args=args, model=model, graph=dataset.graph, split_idx=split_idx, A_X_vec=A_X_vec, A_Y_vec=A_Y_vec, A_vec=A_vec, 
                                lr=0.01, weight_decay=0.00000, epochs=5000, 
                                device=device, device_auc=device_auc, max_indep_set=max_indep_set)
        attack_auc, _, _, _, _, _ = attacker.attack(target=1)
        best_attack_auc = attack_auc
        best_attack_proj_auc = attack_auc

        load_model(args, model, optimizer, run, id=unique_id, delete=True)
        del attacker
        torch.cuda.empty_cache()

    # --------------------------------------------------

    if args.defence_visual:
        model, _ = load_model(args, model, optimizer, run, id=unique_id)
        attacker = GraphAttacker(args=args, model=model, graph=dataset.graph, split_idx=split_idx, A_X_vec=A_X_vec, A_Y_vec=A_Y_vec, A_vec=A_vec, 
                                lr=0.01, weight_decay=0.00000, epochs=5000, 
                                device=device, device_auc=device_auc, max_indep_set=max_indep_set)

        if args.defence_visual_mode == 1:
            _, _, _, _, _, _ = attacker.attack(target=3)
        elif args.defence_visual_mode == 2:
            _, _, _, _, _, _ = attacker.attack(target=4)
        else:
            raise ValueError('Invalid defence visual mode')

        load_model(args, model, optimizer, run, id=unique_id, delete=True)
        del attacker
        torch.cuda.empty_cache()

    # --------------------------------------------------

    # save_model(args, model, optimizer, run)

    # lr=0.01, weight_decay=0.00001

    # model, _ = load_model(args, model, optimizer, run)
    # attacker = GraphAttacker(args=args, model=model, graph=dataset.graph, A_X_vec=A_X_vec, A_Y_vec=A_Y_vec, A_vec=A_vec, 
    #                          lr=0.01, weight_decay=0.00000, epochs=5000, 
    #                          device=device, device_auc=device_auc)
    # attack_1_auc, attack_2_auc, attack_3_auc, attack_4_auc, attack_5_auc, attack_6_auc = attacker.attack()
    # del attacker
    # torch.cuda.empty_cache()

    attack_1_auc = attack_3_auc = attack_4_auc = attack_5_auc = attack_6_auc = best_attack_auc
    attack_2_auc = best_attack_proj_auc

    print('--------------------------------------------------')
    print(f'Run {run+1:02d}:')
    print(f'Attack 1 AUC: {attack_1_auc:.8f}')
    print(f'Attack 2 AUC: {attack_2_auc:.8f}')
    print(f'Attack 3 AUC: {attack_3_auc:.8f}')
    print(f'Attack 4 AUC: {attack_4_auc:.8f}')
    print(f'Attack 5 AUC: {attack_5_auc:.8f}')
    print(f'Attack 6 AUC: {attack_6_auc:.8f}')
    print('--------------------------------------------------')

    attack_1_auc_all.append(attack_1_auc)
    attack_2_auc_all.append(attack_2_auc)
    attack_3_auc_all.append(attack_3_auc)
    attack_4_auc_all.append(attack_4_auc)
    attack_5_auc_all.append(attack_5_auc)
    attack_6_auc_all.append(attack_6_auc)

    # --------------------------------------------------

print('--------------------------------------------------')
results = logger.print_statistics()
print('--------------------------------------------------')

### Save results ###
save_result(args, results)
results_mean = results.mean()
results_std = results.std()

# --------------------------------------------------

attack_1_auc_all = torch.tensor(attack_1_auc_all)
attack_2_auc_all = torch.tensor(attack_2_auc_all)
attack_3_auc_all = torch.tensor(attack_3_auc_all)
attack_4_auc_all = torch.tensor(attack_4_auc_all)
attack_5_auc_all = torch.tensor(attack_5_auc_all)
attack_6_auc_all = torch.tensor(attack_6_auc_all)

attack_1_auc_mean = attack_1_auc_all.mean().item()
attack_2_auc_mean = attack_2_auc_all.mean().item()
attack_3_auc_mean = attack_3_auc_all.mean().item()
attack_4_auc_mean = attack_4_auc_all.mean().item()
attack_5_auc_mean = attack_5_auc_all.mean().item()
attack_6_auc_mean = attack_6_auc_all.mean().item()

attack_1_auc_std = attack_1_auc_all.std().item()
attack_2_auc_std = attack_2_auc_all.std().item()
attack_3_auc_std = attack_3_auc_all.std().item()
attack_4_auc_std = attack_4_auc_all.std().item()
attack_5_auc_std = attack_5_auc_all.std().item()
attack_6_auc_std = attack_6_auc_all.std().item()

print('--------------------------------------------------')
print(f'Attack 1 AUC: {attack_1_auc_mean:.8f} ± {attack_1_auc_std:.8f}')
print(f'Attack 2 AUC: {attack_2_auc_mean:.8f} ± {attack_2_auc_std:.8f}')
print(f'Attack 3 AUC: {attack_3_auc_mean:.8f} ± {attack_3_auc_std:.8f}')
print(f'Attack 4 AUC: {attack_4_auc_mean:.8f} ± {attack_4_auc_std:.8f}')
print(f'Attack 5 AUC: {attack_5_auc_mean:.8f} ± {attack_5_auc_std:.8f}')
print(f'Attack 6 AUC: {attack_6_auc_mean:.8f} ± {attack_6_auc_std:.8f}')
print('--------------------------------------------------')

print('--------------------------------------------------')
print(f'{attack_1_auc_mean:.8f}')
print(f'{attack_2_auc_mean:.8f}')
print(f'{attack_3_auc_mean:.8f}')
print(f'{attack_4_auc_mean:.8f}')
print(f'{attack_5_auc_mean:.8f}')
print(f'{attack_6_auc_mean:.8f}')
print('--------------------------------------------------')

if not args.defence_visual:
    filename = 'results_csv/' + args.dataset + '.csv'
    append_results_to_file(filename, args.gnn, args.dataset, args.defence_auc, args.defence_sigma, args.defence_k, args.defence_p, args.defence_limit, 
                           attack_1_auc_mean, attack_2_auc_mean, results_mean, attack_1_auc_std, attack_2_auc_std, results_std)

# --------------------------------------------------
