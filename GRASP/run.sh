# (
#   for defence_sigma in 0.0
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 0 --runs 1 --epochs 500 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit --defence_visual --defence_visual_mode 1
#   done
# ) &


# ----------------------------------------------------------------------------------------------------


# (
#   for defence_sigma in $(seq 0.0 0.1 3.0)
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 0 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit --defence_p 0.1
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.1 3.0)
#   do
#       python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 0 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit --defence_p 0.1
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.1 3.0)
#   do
#       python main.py --gnn gcn --dataset pubmed --lr 0.005 --local_layers 2 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 1 --runs 2 --save_model --epochs 1000 --epochs 5000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit --defence_p 0.1
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.1 3.0)
#   do
#       python main.py --gnn gcn --dataset wikics --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --device 2 --ln --save_model --epochs 1000 --epochs 2000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit --defence_p 0.1
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.1 3.0)
#   do
#       python main.py --gnn gcn --dataset amazon-computer --hidden_channels 512 --lr 0.001 --runs 2 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device 5 --save_model --epochs 1000 --epochs 5000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit --defence_p 0.1
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.1 3.0)
#   do
#       python main.py --gnn gcn --dataset amazon-photo --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device 2 --ln --res --save_model --epochs 1000 --epochs 5000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit --defence_p 0.1
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.1 3.0)
#   do
#       python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 4 --res --save_model --epochs 1500 --epochs 5000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit --defence_p 0.1
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.1 3.0)
#   do
#       python main.py --gnn gcn --dataset coauthor-physics --hidden_channels 64 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 4 --res --save_model --epochs 1500 --epochs 5000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit --defence_p 0.1
#   done
# ) &


# GCN ----------------------------------------------------------------------------------------------------


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 0 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 0 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 1 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 1 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset wikics --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --device 4 --ln --save_model --epochs 1000 --epochs 2000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset wikics --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --device 5 --ln --save_model --epochs 1000 --epochs 2000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset pubmed --lr 0.005 --local_layers 2 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 0 --runs 2 --save_model --epochs 1000 --epochs 5000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset pubmed --lr 0.005 --local_layers 2 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 0 --runs 2 --save_model --epochs 1000 --epochs 5000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset amazon-computer --hidden_channels 512 --lr 0.001 --runs 2 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device 1 --ln --save_model --epochs 1000 --epochs 5000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset amazon-computer --hidden_channels 512 --lr 0.001 --runs 2 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device 1 --ln --save_model --epochs 1000 --epochs 5000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset amazon-photo --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device 2 --ln --res --save_model --epochs 1000 --epochs 8000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset amazon-photo --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device 2 --ln --res --save_model --epochs 1000 --epochs 8000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 4 --ln --res --save_model --epochs 1500 --epochs 7000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 4 --ln --res --save_model --epochs 1500 --epochs 7000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset coauthor-physics --hidden_channels 64 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 5 --ln --res --save_model --epochs 1500 --epochs 8000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.5 5.0)
#   do
#       python main.py --gnn gcn --dataset coauthor-physics --hidden_channels 64 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 6 --ln --res --save_model --epochs 1500 --epochs 8000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit
#   done
# )


# GAT ----------------------------------------------------------------------------------------------------


# (
#   for defence_sigma in $(seq 0.0 10.0 200.0)
#   do
#       python main.py --gnn gat --dataset wikics --hidden_channels 512 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 0.0 --dropout 0.7 --device 0 --ln --res --save_model --epochs 1000 --epochs 2000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 10.0 200.0)
#   do
#       python main.py --gnn gat --dataset wikics --hidden_channels 512 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 0.0 --dropout 0.7 --device 0 --ln --res --save_model --epochs 1000 --epochs 2000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 10.0 200.0)
#   do
#       python main.py --gnn gat --dataset amazon-computer --hidden_channels 64 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-5 --dropout 0.5 --device 1 --ln --save_model --epochs 1000 --epochs 5000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 10.0 200.0)
#   do
#       python main.py --gnn gat --dataset amazon-computer --hidden_channels 64 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-5 --dropout 0.5 --device 1 --ln --save_model --epochs 1000 --epochs 5000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 10.0 200.0)
#   do
#       python main.py --gnn gat --dataset amazon-photo --hidden_channels 64 --lr 0.001 --runs 2 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device 2 --ln --res --save_model --epochs 1000 --epochs 8000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 10.0 200.0)
#   do
#       python main.py --gnn gat --dataset amazon-photo --hidden_channels 64 --lr 0.001 --runs 2 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device 2 --ln --res --save_model --epochs 1000 --epochs 8000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 10.0 200.0)
#   do
#       python main.py --gnn gat --dataset coauthor-cs --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 1 --weight_decay 5e-4 --dropout 0.3 --device 4 --ln --res --save_model --epochs 1500 --epochs 7000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 10.0 200.0)
#   do
#       python main.py --gnn gat --dataset coauthor-cs --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 1 --weight_decay 5e-4 --dropout 0.3 --device 4 --ln --res --save_model --epochs 1500 --epochs 7000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 10.0 200.0)
#   do
#       python main.py --gnn gat --dataset coauthor-physics --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-4 --dropout 0.7 --device 5 --bn --res --save_model --epochs 1500 --epochs 8000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 10.0 200.0)
#   do
#       python main.py --gnn gat --dataset coauthor-physics --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-4 --dropout 0.7 --device 6 --bn --res --save_model --epochs 1500 --epochs 8000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5 --defence_limit
#   done
# ) &


# GCN w/o limit ----------------------------------------------------------------------------------------------------


# (
#   for defence_sigma in $(seq 0.0 0.2 1.0)
#   do
#       python main.py --gnn gcn --dataset wikics --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --device 0 --ln --save_model --epochs 1000 --epochs 2000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.2 1.0)
#   do
#       python main.py --gnn gcn --dataset wikics --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --device 0 --ln --save_model --epochs 1000 --epochs 2000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.2 1.0)
#   do
#       python main.py --gnn gcn --dataset amazon-computer --hidden_channels 512 --lr 0.001 --runs 2 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device 1 --ln --save_model --epochs 1000 --epochs 5000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.2 1.0)
#   do
#       python main.py --gnn gcn --dataset amazon-computer --hidden_channels 512 --lr 0.001 --runs 2 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device 1 --ln --save_model --epochs 1000 --epochs 5000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.2 1.0)
#   do
#       python main.py --gnn gcn --dataset amazon-photo --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device 2 --ln --res --save_model --epochs 1000 --epochs 10000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.2 1.0)
#   do
#       python main.py --gnn gcn --dataset amazon-photo --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device 2 --ln --res --save_model --epochs 1000 --epochs 10000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.2 1.0)
#   do
#       python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 4 --ln --res --save_model --epochs 1500 --epochs 7000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.2 1.0)
#   do
#       python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 4 --ln --res --save_model --epochs 1500 --epochs 7000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.2 1.0)
#   do
#       python main.py --gnn gcn --dataset coauthor-physics --hidden_channels 64 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 5 --ln --res --save_model --epochs 1500 --epochs 10000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.0 0.2 1.0)
#   do
#       python main.py --gnn gcn --dataset coauthor-physics --hidden_channels 64 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 6 --ln --res --save_model --epochs 1500 --epochs 10000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1.5
#   done
# ) &


# ----------------------------------------------------------------------------------------------------


# (
#   for defence_sigma in 0.0
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 2 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit
#   done
# ) &


wait && echo "All tasks are finished."


# ----------------------------------------------------------------------------------------------------


# (
#   for defence_sigma in 0.0
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 1 --runs 1 --ln --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit --defence_visual --defence_visual_mode 1
#   done
# ) &


# (
#   for defence_sigma in 0.0
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 0 --runs 1 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit --defence_visual --defence_visual_mode 1
#   done
# ) &


# (
#   for defence_sigma in 0.0
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 1 --runs 1 --ln --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit --defence_visual --defence_visual_mode 1
#   done
# ) &


# (
#   for defence_sigma in 0.0
#   do
#       python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 1 --runs 1 --ln --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit --defence_visual --defence_visual_mode 1
#   done
# ) &


# (
#   for defence_sigma in 0.0
#   do
#       python main.py --gnn gcn --dataset amazon-photo --hidden_channels 256 --lr 0.001 --runs 1 --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device 4 --ln --res --save_model --epochs 2000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 0 --defence_limit --defence_visual --defence_visual_mode 2
#   done
# ) &


# python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 0 --runs 1 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma 0.0 --defence_k 0 --defence_limit --defence_visual --defence_visual_mode 1


# python main.py --gnn gcn --dataset amazon-photo --hidden_channels 256 --lr 0.001 --runs 1 --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device 2 --ln --res --save_model --epochs 2000 --defence_auc 1.00 --defence_positive --defence_sigma 0.0 --defence_k 0 --defence_limit --defence_visual --defence_visual_mode 1


# wait && echo "All tasks are finished."
