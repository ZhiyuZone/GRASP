import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, remove_self_loops
from torchmetrics.classification import BinaryAUROC
import os
import matplotlib.pyplot as plt
from parse import parse_method
import numpy as np
import seaborn as sns
import networkx as nx
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


class GraphAttacker:
    def __init__(self, args, model, graph, split_idx, A_X_vec, A_Y_vec, A_vec, 
                 lr=0.01, weight_decay=0.00000, epochs=5000, 
                 device='cpu', device_auc='cpu', max_indep_set=None):

        self.args = args
        self.device = device
        self.device_auc = device_auc

        # model.reset_parameters()
        model.eval()
        self.model = model
        self.N = graph['num_nodes']
        # self.D = graph['node_feat'].shape[1]
        # self.C = graph['num_classes']
        self.X = graph['node_feat']
        # self.X = torch.zeros(self.N, self.D).to(device)
        # self.X = torch.randn(self.N, self.D).to(device)
        self.A = graph['edge_index']
        self.Y = graph['node_label']
        self.A_vec = A_vec
        # self.A_vec = self._get_A_vec().to('cpu')
        self.split_idx = split_idx

        self.H_A, self.Z_A, self.Y_A = self._get_H_A_and_Z_A_and_Y_A()
        # self.H_I, self.Z_I, self.Y_I = self._get_H_I_and_Z_I_and_Y_I()

        # self.A_X_vec = self._get_A_X_vec().to('cpu')
        # self.A_X_vec = A_X_vec
        # self.A_Y_vec = A_Y_vec
        # self.A_H_A_vec = self._get_A_H_A_vec().to('cpu')
        # self.A_Z_A_vec = self._get_A_Z_A_vec().to('cpu')
        # self.A_Y_A_vec = self._get_A_Y_A_vec().to('cpu')
        # self.A_H_I_vec = self._get_A_H_I_vec().to('cpu')
        # self.A_Z_I_vec = self._get_A_Z_I_vec().to('cpu')
        # self.A_Y_I_vec = self._get_A_Y_I_vec().to('cpu')

        # self.A_H_A_modified_vec = self._get_A_H_A_modified_vec().to('cpu')

        # self.A_attack_params = nn.Parameter(torch.zeros(self.N * (self.N - 1) // 2, requires_grad=True).to(device))
        # self.optimizer = optim.Adam([self.A_attack_params], lr=lr, weight_decay=weight_decay)
        # # self.optimizer = optim.SGD([self.A_attack_params], lr=lr, weight_decay=weight_decay)
        # self.epochs = epochs
        self.metric = BinaryAUROC().to(device_auc)

        # # self._plot_distributions(self.A_H_A_vec, self.A_vec)

        self.max_indep_set = max_indep_set


    def _get_H_A_and_Z_A_and_Y_A(self):
        with torch.no_grad():
            H_A, Z_A = self.model(self.X, self.A, get_embedding=True)
            Y_A = torch.softmax(Z_A, dim=1)
            return H_A.detach(), Z_A.detach(), Y_A.detach()


    def _get_H_I_and_Z_I_and_Y_I(self):
        with torch.no_grad():
            indices = torch.stack([torch.arange(self.N), torch.arange(self.N)]).to(self.device)
            values = torch.ones(self.N).to(self.device)
            A_I = torch.sparse_coo_tensor(indices, values, (self.N, self.N))
            A_I = A_I.to_sparse_csr()

            H_I, Z_I = self.model(self.X, A_I, get_embedding=True)
            Y_I = torch.softmax(Z_I, dim=1)
            return H_I.detach(), Z_I.detach(), Y_I.detach()


    def _get_A_vec(self):
        A_vec = to_dense_adj(self.A, max_num_nodes=self.N)[0]
        A_vec.fill_diagonal_(0.)
        A_vec = A_vec.view(-1)
        return A_vec


    def _get_A_X_vec(self):
        A_X_vec = self.X
        A_X_vec = F.normalize(A_X_vec, p=2, dim=1)
        A_X_vec = torch.mm(A_X_vec, A_X_vec.t())
        A_X_vec = F.relu(A_X_vec)
        A_X_vec.fill_diagonal_(0.)
        A_X_vec = A_X_vec.view(-1)
        return A_X_vec


    def _get_A_H_A_modified_vec(self):
        A_H_A_vec = self._get_H_modified(self.H_A)
        A_H_A_vec = A_H_A_vec - A_H_A_vec.mean(dim=1, keepdim=True)
        A_H_A_vec = F.normalize(A_H_A_vec, p=2, dim=1)
        A_H_A_vec = torch.mm(A_H_A_vec, A_H_A_vec.t())
        A_H_A_vec = F.relu(A_H_A_vec)
        A_H_A_vec.fill_diagonal_(0.)
        A_H_A_vec = A_H_A_vec.view(-1)
        return A_H_A_vec


    def _get_H_modified(self, H):
        class_means = torch.zeros_like(H)

        for class_idx in self.Y.unique():
            class_mask = (self.Y == class_idx)
            class_samples = H[class_mask]
            if class_samples.size(0) > 0:
                class_means[class_mask] = class_samples.mean(dim=0)

        H_modified = H + 0.5 * (H - class_means)
        # H_modified = H + 0.5 * (class_means - H)
        return H_modified


    def _get_A_H_A_vec(self):
        A_H_A_vec = self.H_A
        A_H_A_vec = A_H_A_vec - A_H_A_vec.mean(dim=1, keepdim=True)
        A_H_A_vec = F.normalize(A_H_A_vec, p=2, dim=1)
        A_H_A_vec = torch.mm(A_H_A_vec, A_H_A_vec.t())
        A_H_A_vec = F.relu(A_H_A_vec)
        A_H_A_vec.fill_diagonal_(0.)
        A_H_A_vec = A_H_A_vec.view(-1)
        return A_H_A_vec


    def _get_A_Z_A_vec(self):
        A_Z_A_vec = self.Z_A
        A_Z_A_vec = A_Z_A_vec - A_Z_A_vec.mean(dim=1, keepdim=True)
        A_Z_A_vec = F.normalize(A_Z_A_vec, p=2, dim=1)
        A_Z_A_vec = torch.mm(A_Z_A_vec, A_Z_A_vec.t())
        A_Z_A_vec = F.relu(A_Z_A_vec)
        A_Z_A_vec.fill_diagonal_(0.)
        A_Z_A_vec = A_Z_A_vec.view(-1)
        return A_Z_A_vec


    def _get_A_Y_A_vec(self):
        A_Y_A_vec = self.Y_A
        # A_Y_A_vec = A_Y_A_vec - A_Y_A_vec.mean(dim=1, keepdim=True)
        # A_Y_A_vec = F.normalize(A_Y_A_vec, p=2, dim=1)
        A_Y_A_vec = torch.mm(A_Y_A_vec, A_Y_A_vec.t())
        A_Y_A_vec = F.relu(A_Y_A_vec)
        A_Y_A_vec.fill_diagonal_(0.)
        A_Y_A_vec = A_Y_A_vec.view(-1)
        return A_Y_A_vec


    def _get_A_H_I_vec(self):
        A_H_I_vec = self.H_I
        A_H_I_vec = A_H_I_vec - A_H_I_vec.mean(dim=1, keepdim=True)
        A_H_I_vec = F.normalize(A_H_I_vec, p=2, dim=1)
        A_H_I_vec = torch.mm(A_H_I_vec, A_H_I_vec.t())
        A_H_I_vec = F.relu(A_H_I_vec)
        A_H_I_vec.fill_diagonal_(0.)
        A_H_I_vec = A_H_I_vec.view(-1)
        return A_H_I_vec


    def _get_A_Z_I_vec(self):
        A_Z_I_vec = self.Z_I
        A_Z_I_vec = A_Z_I_vec - A_Z_I_vec.mean(dim=1, keepdim=True)
        A_Z_I_vec = F.normalize(A_Z_I_vec, p=2, dim=1)
        A_Z_I_vec = torch.mm(A_Z_I_vec, A_Z_I_vec.t())
        A_Z_I_vec = F.relu(A_Z_I_vec)
        A_Z_I_vec.fill_diagonal_(0.)
        A_Z_I_vec = A_Z_I_vec.view(-1)
        return A_Z_I_vec


    def _get_A_Y_I_vec(self):
        A_Y_I_vec = self.Y_I
        # A_Y_I_vec = A_Y_I_vec - A_Y_I_vec.mean(dim=1, keepdim=True)
        # A_Y_I_vec = F.normalize(A_Y_I_vec, p=2, dim=1)
        A_Y_I_vec = torch.mm(A_Y_I_vec, A_Y_I_vec.t())
        A_Y_I_vec = F.relu(A_Y_I_vec)
        A_Y_I_vec.fill_diagonal_(0.)
        A_Y_I_vec = A_Y_I_vec.view(-1)
        return A_Y_I_vec


    def _get_A_attack(self):
        # A_attack_params = F.sigmoid(self.A_attack_params)
        A_attack = torch.zeros(self.N, self.N).to(self.device)
        upper_tri_indices = torch.triu_indices(self.N, self.N, 1).to(self.device)
        A_attack[upper_tri_indices[0], upper_tri_indices[1]] = self.A_attack_params
        A_attack = A_attack + A_attack.t()
        return A_attack


    def _get_upper_tri_vec(self, vec):
        upper_tri_indices = torch.triu_indices(self.N, self.N, 1).to(self.device)
        vec = vec.to(self.device)[upper_tri_indices[0] * self.N + upper_tri_indices[1]].to(vec.device)
        return vec


    def _get_auc(self, pred, true, batch_size=-1):
        self.metric.reset()
        length = len(true)

        if batch_size <= 0 or batch_size >= length:
            # pred = self._get_upper_tri_vec(pred)
            # true = self._get_upper_tri_vec(true)
            auc = self.metric(pred.to(self.device_auc), true.to(self.device_auc))
        else:
            perm = torch.randperm(length)
            for start in range(0, length, batch_size):
                end = min(start + batch_size, length)
                batch_perm = perm[start:end]
                batch_pred = pred[batch_perm]
                batch_true = true[batch_perm]
                self.metric.update(batch_pred.to(self.device_auc), batch_true.to(self.device_auc))
            auc = self.metric.compute()

        self.metric.reset()
        torch.cuda.empty_cache()
        return auc


    def _similarity_function(self, H, function="correlation"):
        if function == "dot":
            H = H - H.mean(dim=1, keepdim=True)
            return torch.mm(H, H.t())

        elif function == "correlation":
            H = H - H.mean(dim=1, keepdim=True)
            H = F.normalize(H, p=2, dim=1)
            return torch.mm(H, H.t())

        elif function == "cosine":
            H = F.normalize(H, p=2, dim=1)
            return torch.mm(H, H.t())

        elif function == "euclidean":
            H = H.unsqueeze(1) - H.unsqueeze(0)
            return -torch.sqrt(torch.sum(H ** 2, dim=2))

        elif function == "braycurtis":
            D = torch.sum(H.unsqueeze(1) + H.unsqueeze(0), dim=2)
            H = torch.sum(torch.abs(H.unsqueeze(1) - H.unsqueeze(0)), dim=2)
            return -H / D

        elif function == "manhattan":
            H = H.unsqueeze(1) - H.unsqueeze(0)
            return -torch.sum(torch.abs(H), dim=2)

        elif function == "sqeuclidean":
            H = H.unsqueeze(1) - H.unsqueeze(0)
            return -torch.sum(H ** 2, dim=2)

        else:
            raise ValueError(f"Unsupported function: {function}")


    def _get_A_cosine_vec(self, H, sigma=0.0, function="correlation"):
        # function = "dot"
        # H = H - H.mean(dim=1, keepdim=True)
        # H = F.normalize(H, p=2, dim=1)
        # H = torch.mm(H, H.t())
        H = self._similarity_function(H, function=function)

        # H = F.relu(H)
        H.fill_diagonal_(0.)
        H = H.view(-1)
        return H


    # def _get_A_cosine_vec(self, H, sigma=0.0, function="correlation"):
    #     H = H - H.mean(dim=1, keepdim=True)

    #     H = F.normalize(H, p=2, dim=1)

    #     # norms = H.norm(p=2, dim=1, keepdim=True)
    #     # norms = torch.sqrt((norms**2).mean() - sigma**2 * H.size(1))
    #     # norms[norms > 1.5 * sigma**2 * H.size(1)] -= sigma**2 * H.size(1)
    #     # norms = torch.sqrt(norms)
    #     # H = H / norms

    #     H = torch.mm(H, H.t())
    #     # H = F.relu(H)
    #     H.fill_diagonal_(0.)
    #     H = H.view(-1)
    #     return H


    def _get_noise(self, x, k=0, sigma=0.0, p=0.5):
        n, d = x.size()

        if k == -2:
            noise = torch.normal(0.0, sigma, size=(1, d), device=x.device)
            noise = torch.cat((noise, -noise), dim=0)
            noise = noise[torch.randint(0, 2, (n,), device=x.device)]

        elif k == -1:
            noise = torch.normal(0.0, sigma, size=x.size(), device=x.device)
            noise[self.max_indep_set] = torch.normal(0.0, sigma, size=(1, d), device=x.device)

        elif k == 0:
            noise = torch.normal(0.0, sigma, size=x.size(), device=x.device)

        elif k == 0.5:
            noise_1 = torch.normal(0.0, sigma, size=x.size(), device=x.device)
            noise_2 = torch.normal(0.0, sigma, size=(1, d), device=x.device)
            noise_2 = noise_2.repeat(n, 1)
            noise = 0.5 * noise_1 + 0.5 * noise_2

        elif k == 1.5:
            noise = torch.normal(0.0, sigma, size=x.size(), device=x.device)
            # noise[torch.rand(n) < p] = torch.normal(0.0, sigma, size=(1, d), device=x.device)
            noise[torch.rand(n) < p**0.5] = torch.normal(0.0, sigma, size=(1, d), device=x.device)

        elif k >= 1:
            noise = torch.normal(0.0, sigma, size=(k, d), device=x.device)
            noise = noise[torch.randint(0, k, (n,), device=x.device)]

        return noise


    def _plot_vector(self, A_attack, A_indices, k, sigma, privacy, utility, mode=1, step=0.1, function=None):
        if mode == 1:
            if isinstance(A_attack, torch.Tensor):
                A_attack = A_attack.cpu().numpy()
            if isinstance(A_indices, torch.Tensor):
                A_indices = A_indices.cpu().numpy()
            A_indices_1 = np.where(A_indices==0.0)[0]
            A_indices_2 = np.where(A_indices==1.0)[0]

            results_fig = 'results_fig'
            dataset_dir = os.path.join(results_fig, self.args.dataset)
            os.makedirs(dataset_dir, exist_ok=True)

            plt.figure(figsize=(6, 3))
            plt.rcParams.update({'font.family': 'Times New Roman'})

            A_indices_1 = A_attack[A_indices_1]
            A_indices_2 = A_attack[A_indices_2]
            # A_indices_1 = A_indices_1[A_indices_1 > 0.1]
            # A_indices_2 = A_indices_2[A_indices_2 > 0.1]

            if function is None or function in ("correlation", "cosine"):
                sns.kdeplot(A_indices_1, color='b', label='A=0', shade=True, alpha=0.6, clip=(-1, 1), linewidth=2)
                sns.kdeplot(A_indices_2, color='r', label='A=1', shade=True, alpha=0.6, clip=(-1, 1), linewidth=2)
            elif function == "braycurtis":
                sns.kdeplot(A_indices_1, color='b', label='A=0', shade=True, alpha=0.6, clip=(-1, 0))
                sns.kdeplot(A_indices_2, color='r', label='A=1', shade=True, alpha=0.6, clip=(-1, 0))
            else:
                sns.kdeplot(A_indices_1, color='b', label='A=0', shade=True, alpha=0.6, clip=(-np.inf, 0))
                sns.kdeplot(A_indices_2, color='r', label='A=1', shade=True, alpha=0.6, clip=(-np.inf, 0))

            ax = plt.gca()
            ax.spines['top'].set_linewidth(3)
            ax.spines['right'].set_linewidth(3)
            ax.spines['left'].set_linewidth(3)
            ax.spines['bottom'].set_linewidth(3)

            plt.axvline(x=0, color='k', linestyle='--', linewidth=3)
            plt.axvline(x=1, color='k', linestyle='--', linewidth=3)

            # if np.min(A_attack) <= 0.05:
            #     plt.axvline(x=0, color='k', linestyle='--', linewidth=3)
            # if np.max(A_attack) >= 0.95:
            #     plt.axvline(x=1, color='k', linestyle='--', linewidth=3)

            fontsize = 32
            # plt.subplots_adjust(left=0.13, right=0.87, top=0.87, bottom=0.13)
            plt.subplots_adjust(left=0.03, right=0.97, top=0.85, bottom=0.15)

            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

            plt.xlabel('')
            plt.ylabel('')

            # plt.title(r"$\mathrm{\sigma}$" + f" = {sigma}     AUC = {privacy:.4f}", fontsize=fontsize)
            plt.title(r"$\mathregular{\sigma}$" + f" = {sigma:.2f}     AUC = {privacy:.4f}", fontsize=fontsize)
            # plt.title(r"$\sigma$" + f" = {sigma}    AUC = {privacy:.4f}", fontsize=fontsize)
            # plt.title(f"AUC = {privacy:.4f}", fontsize=fontsize)

            # plt.title("Title", color='none')
            # plt.xlabel("X Axis", color='none')
            # plt.ylabel("Y Axis", color='none')

            # plt.gca().spines['top'].set_color('none')
            # plt.gca().spines['right'].set_color('none')
            # plt.gca().spines['left'].set_color('none')
            # plt.gca().spines['bottom'].set_color('none')

            # plt.grid(True, color='none')

            # plt.xticks(color='none')
            # plt.yticks(color='none')

            # plt.tick_params(axis='x', colors='none')
            # plt.tick_params(axis='y', colors='none')

            # if function is None:
            #     plt.title(f"dataset={self.args.dataset}, k={k}, sigma={sigma:.1f}, privacy={privacy:.4f}, utility={utility:.4f}", fontsize=fontsize)
            # else:
            #     plt.title(f"dataset={self.args.dataset}, func={function}, privacy={privacy:.4f}", fontsize=fontsize)

            # if function is None:
            #     plt.xlabel('Similarity', fontsize=fontsize)
            # else:
            #     plt.xlabel(f'{function} similarity', fontsize=fontsize)
            # plt.xlabel(r'$\sigma$ = ' + f'{sigma:.1f}', fontsize=fontsize)
            # plt.ylabel('Density', fontsize=fontsize)
            # plt.legend(fontsize=fontsize)

            ax.set_yticks([])
            ax.set_xticks([0.0, 0.5, 1.0])

            plt.tick_params(axis='both', which='major', labelsize=fontsize, width=3)

            if function is None:
                file_name = os.path.join(dataset_dir, f"{self.args.dataset}_k={k}_sigma={sigma:.2f}_privacy={privacy:.4f}_utility={utility:.4f}.png")
            else:
                file_name = os.path.join(dataset_dir, f"{self.args.dataset}_privacy={privacy:.4f}_func={function}.png")

            plt.savefig(file_name, dpi=500)
            # plt.savefig(file_name, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved {file_name}")

        elif mode == 2:
            if isinstance(A_attack, torch.Tensor):
                A_attack = A_attack.cpu().numpy()
            if isinstance(A_indices, torch.Tensor):
                A_indices = A_indices.cpu().numpy()
            A_indices = np.where(A_indices)[0]

            results_fig = 'results_fig'
            dataset_dir = os.path.join(results_fig, self.args.dataset)
            os.makedirs(dataset_dir, exist_ok=True)

            plt.figure(figsize=(7, 7))
            plt.hist(A_attack, bins=np.arange(min(A_attack), max(A_attack) + step, step), 
                    color='b', alpha=0.6, label='A=0', density=True)
            plt.hist(A_attack[A_indices], bins=np.arange(min(A_attack), max(A_attack) + step, step), 
                    color='r', alpha=0.6, label='A=1', density=True)

            plt.title(f"dataset={self.args.dataset}, k={k}, sigma={sigma:.1f}, privacy={privacy:.4f}, utility={utility:.4f}", fontsize=12)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.xlabel('correlation similarity', fontsize=12)
            plt.ylabel('frequency', fontsize=12)
            # plt.legend(fontsize=12)

            file_name = os.path.join(dataset_dir, f"{self.args.dataset}_k={k}_sigma={sigma:.1f}_privacy={privacy:.4f}_utility={utility:.4f}.png")
            plt.savefig(file_name, dpi=300)
            plt.close()
            print(f"Saved {file_name}")

        elif mode == 3:
            if isinstance(A_attack, torch.Tensor):
                A_attack = A_attack.cpu().numpy()
            if isinstance(A_indices, torch.Tensor):
                A_indices = A_indices.cpu().numpy()
            A_indices = np.where(A_indices)[0]

            results_fig = 'results_fig'
            dataset_dir = os.path.join(results_fig, self.args.dataset)
            os.makedirs(dataset_dir, exist_ok=True)

            indices = range(len(A_attack))
            plt.figure(figsize=(7, 7))
            plt.scatter(indices, A_attack, color='b', label='A=0', s=1)
            plt.scatter(A_indices, A_attack[A_indices], color='r', label='A=1', s=1)

            plt.title(f"dataset={self.args.dataset}, k={k}, sigma={sigma:.1f}, privacy={privacy:.4f}, utility={utility:.4f}", fontsize=12)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.xlabel('index of node pairs', fontsize=12)
            plt.ylabel('correlation similarity', fontsize=12)
            # plt.legend(fontsize=12)

            file_name = os.path.join(dataset_dir, f"{self.args.dataset}_k={k}_sigma={sigma:.1f}_privacy={privacy:.4f}_utility={utility:.4f}.png")
            plt.savefig(file_name, dpi=300)
            plt.close()
            print(f"Saved {file_name}")


    def _eval_acc(self, y_true, y_pred):
        acc_list = []
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return sum(acc_list)/len(acc_list)


    def _attack_1(self, target=None):
        if target is not None:
            if target == 1:
                A_attack = self._get_A_H_A_vec()
                attack_1_auc = attack_2_auc = attack_3_auc = \
                    attack_4_auc = attack_5_auc = attack_6_auc = self._get_auc(A_attack, self.A_vec)

            elif target == 2:
                A_attack = self._get_A_Z_A_vec()
                attack_1_auc = attack_2_auc = attack_3_auc = \
                    attack_4_auc = attack_5_auc = attack_6_auc = self._get_auc(A_attack, self.A_vec)

            elif target == 3:
                # A_attack = self._get_A_cosine_vec(self.H_A, sigma=0.0)
                # privacy = self._get_auc(A_attack, self.A_vec)
                # utility = self._eval_acc(self.Y.unsqueeze(1)[self.split_idx['test']], 
                #                          self.model.pred_local(self.H_A)[self.split_idx['test']])

                # A_attack, A_attack_sorted_indices = torch.sort(A_attack)
                # A_indices = self.A_vec[A_attack_sorted_indices.to(self.A_vec.device)]
                A_indices = self.A_vec
                # self._plot_vector(A_attack, A_indices, k=0, sigma=0.0, privacy=privacy, utility=utility)

                for k in (0, ):
                    for sigma in (0.00, 0.25, 0.50, 0.75, 1.00):
                # for k in (0, 1, 1.5):
                    # for sigma in (0.00, 0.25, 0.50, 0.75, 1.00):
                        H_attack = self.H_A + self._get_noise(self.H_A, k=k, sigma=sigma)
                        A_attack = self._get_A_cosine_vec(H_attack, sigma=sigma)
                        privacy = self._get_auc(A_attack, self.A_vec)
                        utility = self._eval_acc(self.Y.unsqueeze(1)[self.split_idx['test']], 
                                                 self.model.pred_local(H_attack)[self.split_idx['test']])

                        # A_attack = A_attack[A_attack_sorted_indices]
                        self._plot_vector(A_attack, A_indices, k, sigma, privacy=privacy, utility=utility)

                attack_1_auc = attack_2_auc = attack_3_auc = \
                    attack_4_auc = attack_5_auc = attack_6_auc = 0.0

            elif target == 4:
                utility = self._eval_acc(self.Y.unsqueeze(1)[self.split_idx['test']], 
                                         self.model.pred_local(self.H_A)[self.split_idx['test']])

                for function in ("correlation", "cosine", "euclidean", "braycurtis", "manhattan", "sqeuclidean"):
                    A_attack = self._get_A_cosine_vec(self.H_A, sigma=0.0, function=function)
                    privacy = self._get_auc(A_attack, self.A_vec)
                    A_indices = self.A_vec
                    self._plot_vector(A_attack, A_indices, k=0, sigma=0.0, privacy=privacy, utility=utility, function=function)

                attack_1_auc = attack_2_auc = attack_3_auc = \
                    attack_4_auc = attack_5_auc = attack_6_auc = 0.0

            else:
                raise ValueError('Invalid target')

        else:
            A_attack_1 = self.A_H_A_vec
            A_attack_2 = self.A_Z_A_vec
            # # A_attack_2 = self.A_H_A_modified_vec
            # A_attack_3 = self.A_Y_A_vec
            # A_attack_4 = self.A_H_I_vec
            # A_attack_5 = self.A_Z_I_vec
            # A_attack_6 = self.A_Y_I_vec

            # attack_1_auc = self._get_auc(A_attack_1, self.A_vec)
            # attack_2_auc = self._get_auc(A_attack_2, self.A_vec)
            # attack_3_auc = self._get_auc(A_attack_3, self.A_vec)
            # attack_4_auc = self._get_auc(A_attack_4, self.A_vec)
            # attack_5_auc = self._get_auc(A_attack_5, self.A_vec)
            # attack_6_auc = self._get_auc(A_attack_6, self.A_vec)

            # attack_1_auc = self._get_auc(A_attack_1, self.A_Y_vec)
            # attack_2_auc = self._get_auc(A_attack_2, self.A_Y_vec)
            # attack_3_auc = self._get_auc(A_attack_3, self.A_Y_vec)
            # attack_4_auc = self._get_auc(A_attack_4, self.A_Y_vec)
            # attack_5_auc = self._get_auc(A_attack_5, self.A_Y_vec)
            # attack_6_auc = self._get_auc(A_attack_6, self.A_Y_vec)

            attack_1_auc = self._get_auc(A_attack_1, self.A_vec)
            attack_2_auc = self._get_auc(A_attack_2, self.A_vec)
            attack_3_auc = attack_1_auc
            attack_4_auc = attack_1_auc
            attack_5_auc = attack_1_auc
            attack_6_auc = attack_1_auc

        return attack_1_auc, attack_2_auc, attack_3_auc, attack_4_auc, attack_5_auc, attack_6_auc


    def _get_A_norm(self, A):
        # D = (A != 0.).float().sum(dim=1).pow(-0.5)
        D = A.sum(dim=1).pow(-0.5)
        D[torch.isinf(D)] = 0.
        A = D.unsqueeze(1) * A
        A = A * D.unsqueeze(0)
        return A


    def _get_A_cosine(self, H):
        H = H - H.mean(dim=1, keepdim=True)
        H = F.normalize(H, p=2, dim=1)
        H = torch.mm(H, H.t())
        H = F.relu(H)
        H.fill_diagonal_(0.)
        return H


    def _gradient_projection(self):
        with torch.no_grad():
            self.A_attack_params.clamp_(0.0, 1.0)


    def _cos_loss(self, pred, true):
        pred = pred - pred.mean(dim=1, keepdim=True)
        true = true - true.mean(dim=1, keepdim=True)
        loss = 1 - F.cosine_similarity(pred, true, dim=1)
        return loss.mean()


    def _mat_loss(self, H, random_indices):
        # pred = self._get_A_attack()[random_indices, :][:, random_indices].view(-1)
        # pred = self._get_A_attack().view(-1)
        pred = self._get_A_attack()
        pred.fill_diagonal_(1.)
        pred = self._get_A_norm(pred)
        pred.fill_diagonal_(0.)
        pred = pred.view(-1)
        true = self._get_A_cosine_vec(H.detach())
        # true = (true > torch.quantile(true, 0.9)).float()
        # true = self.A_vec.to(self.device)
        loss = F.mse_loss(pred, true)
        # loss = F.binary_cross_entropy(pred, true)
        # loss = -torch.log(pred[true > torch.quantile(true, 0.99)] + 1e-8)
        # loss = -torch.log(1 - pred[true <= 0.8] + 1e-8)
        # loss = pred[true <= 0.0]**2
        return loss.mean()


    def _weighted_loss(self, H):
        pred = self._get_A_attack()
        pred.fill_diagonal_(1.)
        pred = self._get_A_norm(pred)
        pred.fill_diagonal_(0.)
        pred = pred.view(-1)
        # true = self._get_A_cosine_vec(H.detach())
        # true = self.A_vec.to(self.device)
        true = self.A_X_vec.to(self.device)
        loss = -true * pred
        return loss.mean()


    def _entropy_regularization(self, vec):
        entropy = -(vec * torch.log(vec + 1e-8) + (1 - vec) * torch.log(1 - vec + 1e-8))
        # entropy = vec * (1. - vec)
        # entropy = vec ** 2 * (1. - vec) ** 2
        # entropy = -vec * torch.log(vec + 1e-8)
        # entropy = -(1 - vec) * torch.log(1 - vec + 1e-8)
        return entropy.mean()


    # def _update_A_attack_params(self, H):
    #     with torch.no_grad():
    #         # A = self._get_A_cosine(H).detach()
    #         A = self._get_A_cosine(self.H_A).detach()
    #         # A = to_dense_adj(self.A, max_num_nodes=self.N)[0]
    #         # A.fill_diagonal_(0.)
    #         upper_tri_indices = torch.triu_indices(self.N, self.N, 1).to(self.device)
    #         self.A_attack_params.copy_(A[upper_tri_indices[0], upper_tri_indices[1]])


    def _attack_2(self):
        k = int(self.N * 1.0)
        random_indices = torch.randperm(self.N)[:k].to(self.device)

        for epoch in range(self.epochs):
            A_attack = self._get_A_attack()
            A_attack.fill_diagonal_(1.) # ?
            A_attack = self._get_A_norm(A_attack)
            H_A_attack, _ = self.model(self.X, A_attack, get_embedding=True, dense_adj=True)
            # H_A_attack = A_attack @ self.H_I

            # A_true = to_dense_adj(self.A, max_num_nodes=self.N)[0]
            # A_true.fill_diagonal_(1.)
            # A_true = self._get_A_norm(A_true)
            # H_A_true, Z_A_true = self.model(self.X, A_true, get_embedding=True, dense_adj=True)

            # loss_1 = F.mse_loss(H_A_attack[random_indices], self.H_A[random_indices])
            loss_1 = self._cos_loss(H_A_attack[random_indices], self.H_A[random_indices])

            # loss_2 = self._mat_loss(H_A_attack[random_indices], random_indices)
            # loss_2 = self._mat_loss(H_A_attack, random_indices)
            # loss_2 = F.mse_loss(Z_A_attack[random_indices], self.Z_A[random_indices])
            # loss_2 = self._cos_loss(Z_A_attack[random_indices], self.Z_A[random_indices])

            # loss_l1 = torch.mean(torch.abs(self.A_attack_params))
            # loss_entropy = self._entropy_regularization(self.A_attack_params)
            # loss_entropy = self._entropy_regularization(A_attack.view(-1))
            # loss_entropy_vec = self._entropy_regularization(H_A_attack_vec)

            # weighted_loss = self._weighted_loss(H_A_attack)

            loss_train = loss_1

            self.optimizer.zero_grad()
            loss_train.backward()
            self.optimizer.step()
            self._gradient_projection()

            # if (epoch + 1) % 100 == 0:
            # if epoch == 500:
            #     self._update_A_attack_params(H_A_attack)

            if (epoch + 1) % 500 == 0:
                with torch.no_grad():
                    A_attack.fill_diagonal_(0.)
                    A_attack = A_attack.view(-1).to('cpu')

                    A_attack_1 = self._get_A_attack().view(-1).to('cpu')
                    A_attack_2 = A_attack
                    A_attack_3 = self._get_A_cosine_vec(H_A_attack).to('cpu')
                    A_attack_4 = A_attack_1 + A_attack_3
                    # A_attack_5 = A_attack_1 + A_attack_2 + A_attack_3
                    # A_attack_6 = self.A_H_A_vec
                    A_attack_5 = A_attack_2 + A_attack_3
                    A_attack_6 = A_attack_1 + A_attack_2 + A_attack_3

                    attack_1_auc = self._get_auc(A_attack_1, self.A_vec)
                    attack_2_auc = self._get_auc(A_attack_2, self.A_vec)
                    attack_3_auc = self._get_auc(A_attack_3, self.A_vec)
                    attack_4_auc = self._get_auc(A_attack_4, self.A_vec)
                    attack_5_auc = self._get_auc(A_attack_5, self.A_vec)
                    attack_6_auc = self._get_auc(A_attack_6, self.A_vec)

                    print(f'Epoch: {epoch+1:05d}, '
                        f'Attack 1: {attack_1_auc:.8f}, '
                        f'Attack 2: {attack_2_auc:.8f}, '
                        f'Attack 3: {attack_3_auc:.8f}, '
                        f'Attack 4: {attack_4_auc:.8f}, '
                        f'Attack 5: {attack_5_auc:.8f}, '
                        f'Attack 6: {attack_6_auc:.8f}')

        with torch.no_grad():
            if A_attack.size(0) != self.A_vec.size(0):
                A_attack.fill_diagonal_(0.)
                A_attack = A_attack.view(-1).to('cpu')

            A_attack_1 = self._get_A_attack().view(-1).to('cpu')
            A_attack_2 = A_attack
            A_attack_3 = self._get_A_cosine_vec(H_A_attack).to('cpu')
            A_attack_4 = A_attack_1 + A_attack_3
            # A_attack_5 = A_attack_1 + A_attack_2 + A_attack_3
            # A_attack_6 = self.A_H_A_vec
            A_attack_5 = A_attack_2 + A_attack_3
            A_attack_6 = A_attack_1 + A_attack_2 + A_attack_3

            # A_norm_vec = to_dense_adj(self.A, max_num_nodes=self.N)[0]
            # A_norm_vec.fill_diagonal_(1.)
            # A_norm_vec = self._get_A_norm(A_norm_vec)
            # A_norm_vec.fill_diagonal_(0.)
            # A_norm_vec = A_norm_vec.view(-1).to('cpu')
            # self._plot_distributions(A_attack_2, self.A_vec)

            attack_1_auc = self._get_auc(A_attack_1, self.A_vec)
            attack_2_auc = self._get_auc(A_attack_2, self.A_vec)
            attack_3_auc = self._get_auc(A_attack_3, self.A_vec)
            attack_4_auc = self._get_auc(A_attack_4, self.A_vec)
            attack_5_auc = self._get_auc(A_attack_5, self.A_vec)
            attack_6_auc = self._get_auc(A_attack_6, self.A_vec)

        return attack_1_auc, attack_2_auc, attack_3_auc, attack_4_auc, attack_5_auc, attack_6_auc


    def _attack_3(self):
        model = parse_method(self.args, self.N, self.C, self.D, self.device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=self.args.weight_decay, lr=self.args.lr)

        indices = torch.stack([torch.arange(self.N), torch.arange(self.N)]).to(self.device)
        values = torch.ones(self.N).to(self.device)
        A_I = torch.sparse_coo_tensor(indices, values, (self.N, self.N))
        A_I = A_I.to_sparse_csr()

        criterion = nn.CrossEntropyLoss()

        for epoch in range(3000):
            model.train()
            H, Z = model(self.X, A_I, get_embedding=True)
            # Y_pred = F.log_softmax(Z, dim=1)

            loss_1 = self._cos_loss(H, self.H_A)
            loss_2 = criterion(Z, self.Y)
            loss = loss_1 + loss_2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch: {epoch+1:05d}, Loss: {loss.item():.8f}')

        model.eval()
        self.model = model
        return self._attack_2()


    def _attack_4(self):
        model = parse_method(self.args, self.N, self.C, self.D, self.device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=self.args.weight_decay, lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()

        runs = 10
        for run in range(runs):
            print('----------------------------------------')
            print(f'Run: {run+1}/{runs}')
            print('----------------------------------------')
            self.A_attack_params.requires_grad = False
            A_attack = self._get_A_attack()
            A_attack.fill_diagonal_(1.)
            A_attack = self._get_A_norm(A_attack)

            for epoch in range(500):
                model.train()
                H, Z = model(self.X, A_attack, get_embedding=True, dense_adj=True)

                loss_1 = self._cos_loss(H, self.H_A)
                # loss_2 = criterion(Z, self.Y)
                loss = loss_1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 50 == 0:
                    print(f'Epoch: {epoch+1:05d}, Loss: {loss.item():.8f}')

            self.A_attack_params.requires_grad = True
            model.eval()
            self.model = model
            if run == runs - 1:
                return self._attack_2()
            else:
                self._attack_2()


    def _attack_5(self):
        k = int(self.N * 1.0)
        random_indices = torch.randperm(self.N)[:k].to(self.device)

        model = parse_method(self.args, self.N, self.C, self.D, self.device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=self.args.weight_decay, lr=self.args.lr)

        for epoch in range(self.epochs):
            model.train()

            A_attack = self._get_A_attack()
            A_attack.fill_diagonal_(1.)
            A_attack = self._get_A_norm(A_attack)
            H_A_attack, Z_A_attack = model(self.X, A_attack, get_embedding=True, dense_adj=True)

            loss_1 = self._cos_loss(H_A_attack[random_indices], self.H_A[random_indices])
            # loss_2 = self._cos_loss(Z_A_attack[random_indices], self.Z_A[random_indices])
            loss_train = loss_1

            optimizer.zero_grad()
            self.optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            self.optimizer.step()
            self._gradient_projection()

            if (epoch + 1) % 100 == 0:
                with torch.no_grad():
                    A_attack.fill_diagonal_(0.)
                    A_attack = A_attack.view(-1).to('cpu')

                    A_attack_1 = self._get_A_attack().view(-1).to('cpu')
                    A_attack_2 = A_attack
                    A_attack_3 = self._get_A_cosine_vec(H_A_attack).to('cpu')
                    A_attack_4 = A_attack_1 + A_attack_3
                    A_attack_5 = A_attack_2 + A_attack_3
                    A_attack_6 = A_attack_1 + A_attack_2 + A_attack_3

                    attack_1_auc = self._get_auc(A_attack_1, self.A_vec)
                    attack_2_auc = self._get_auc(A_attack_2, self.A_vec)
                    attack_3_auc = self._get_auc(A_attack_3, self.A_vec)
                    attack_4_auc = self._get_auc(A_attack_4, self.A_vec)
                    attack_5_auc = self._get_auc(A_attack_5, self.A_vec)
                    attack_6_auc = self._get_auc(A_attack_6, self.A_vec)

                    print(f'Epoch: {epoch+1:05d}, '
                        f'Attack 1: {attack_1_auc:.8f}, '
                        f'Attack 2: {attack_2_auc:.8f}, '
                        f'Attack 3: {attack_3_auc:.8f}, '
                        f'Attack 4: {attack_4_auc:.8f}, '
                        f'Attack 5: {attack_5_auc:.8f}, '
                        f'Attack 6: {attack_6_auc:.8f}')

        with torch.no_grad():
            if A_attack.size(0) != self.A_vec.size(0):
                A_attack.fill_diagonal_(0.)
                A_attack = A_attack.view(-1).to('cpu')

            A_attack_1 = self._get_A_attack().view(-1).to('cpu')
            A_attack_2 = A_attack
            A_attack_3 = self._get_A_cosine_vec(H_A_attack).to('cpu')
            A_attack_4 = A_attack_1 + A_attack_3
            A_attack_5 = A_attack_2 + A_attack_3
            A_attack_6 = A_attack_1 + A_attack_2 + A_attack_3

            attack_1_auc = self._get_auc(A_attack_1, self.A_vec)
            attack_2_auc = self._get_auc(A_attack_2, self.A_vec)
            attack_3_auc = self._get_auc(A_attack_3, self.A_vec)
            attack_4_auc = self._get_auc(A_attack_4, self.A_vec)
            attack_5_auc = self._get_auc(A_attack_5, self.A_vec)
            attack_6_auc = self._get_auc(A_attack_6, self.A_vec)

        return attack_1_auc, attack_2_auc, attack_3_auc, attack_4_auc, attack_5_auc, attack_6_auc


    def _plot_distributions(self, vec1, vec2, bin_width=0.02, save_path="figures/distribution_plot.png"):

        if len(vec1) != len(vec2):
            raise ValueError("")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        bins = torch.arange(0.0, 1.0 + bin_width, bin_width)

        hist1 = torch.zeros(len(bins) - 1)
        hist2 = torch.zeros(len(bins) - 1)

        for i in range(len(bins) - 1):
            mask = (vec1 >= bins[i]) & (vec1 < bins[i + 1])
            hist1[i] = mask.sum()

            hist2[i] = (mask & (vec2 == 1)).sum()
            # mask = (vec2 >= bins[i]) & (vec2 < bins[i + 1])
            # hist2[i] = mask.sum()

        total_vec1 = len(vec1)
        total_vec2_1 = (vec2 == 1).sum()
        # total_vec2 = len(vec2)

        freq1 = hist1 / total_vec1
        freq2 = hist2 / total_vec2_1 if total_vec2_1 > 0 else torch.zeros_like(hist2)
        # freq2 = hist2 / total_vec2

        plt.figure(figsize=(10, 6))

        plt.plot(bins[:-1].numpy(), freq1.numpy(), marker='o', linestyle='-', color='b', alpha=0.6, label="vec1")

        plt.plot(bins[:-1].numpy(), freq2.numpy(), marker='x', linestyle='-', color='r', alpha=0.6, label="vec2")

        for i in range(len(freq1)):

            plt.text(bins[i].item(), freq1[i].item(), f"{freq1[i].item():.3f}", fontsize=7, ha='center', va='bottom')

            plt.text(bins[i].item(), freq2[i].item(), f"{freq2[i].item():.3f}", fontsize=7, ha='center', va='bottom')

        plt.xlabel("Value Ranges")
        plt.ylabel("Frequency")
        plt.title("Distribution of vec1 and vec2 Frequencies in Ranges")
        plt.legend()

        plt.savefig(save_path)
        plt.close()
        print(f"Figure saved at {save_path}")


    def attack(self, mode=1, target=None):
        if mode == 1:
            return self._attack_1(target)
        elif mode == 2:
            return self._attack_2()
        elif mode == 3:
            return self._attack_3()
        elif mode == 4:
            return self._attack_4()
        elif mode == 5:
            return self._attack_5()
        else:
            raise ValueError('Invalid mode')
