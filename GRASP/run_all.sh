#!/bin/bash


# (
#   for defence_auc in $(seq 1.00 -0.01 0.91)
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 3 --save_model --epochs 500 --epochs 3000 --defence_auc "$defence_auc" --defence_positive --defence_sigma 0.0
#   done
# ) &


# (
#   for defence_auc in $(seq 0.90 -0.01 0.81)
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 2 --runs 3 --save_model --epochs 500 --epochs 3000 --defence_auc "$defence_auc" --defence_positive --defence_sigma 0.0
#   done
# ) &


# (
#   for defence_auc in $(seq 0.80 -0.01 0.71)
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 3 --runs 3 --save_model --epochs 500 --epochs 3000 --defence_auc "$defence_auc" --defence_positive --defence_sigma 0.0
#   done
# ) &


# (
#   for defence_auc in $(seq 1.00 -0.01 0.91)
#   do
#       python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 3 --save_model --epochs 500 --epochs 3000 --defence_auc "$defence_auc" --defence_positive --defence_sigma 0.0
#   done
# ) &


# (
#   for defence_auc in $(seq 0.90 -0.01 0.81)
#   do
#       python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 2 --runs 3 --save_model --epochs 500 --epochs 3000 --defence_auc "$defence_auc" --defence_positive --defence_sigma 0.0
#   done
# ) &


# (
#   for defence_auc in $(seq 0.80 -0.01 0.71)
#   do
#       python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 3 --runs 3 --save_model --epochs 500 --epochs 3000 --defence_auc "$defence_auc" --defence_positive --defence_sigma 0.0
#   done
# ) &


# (
#   for defence_auc in $(seq 0.55 -0.05 0.50)
#   do
#       python main.py --gnn gcn --dataset amazon-computer --hidden_channels 512 --lr 0.001 --runs 1 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device 5 --ln --save_model --epochs 1000 --epochs 5000 --defence_auc "$defence_auc" --defence_positive --defence_sigma 0.0
#   done
# ) &


# (
#   for defence_auc in $(seq 0.45 -0.05 0.40)
#   do
#       python main.py --gnn gcn --dataset amazon-computer --hidden_channels 512 --lr 0.001 --runs 1 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device 5 --ln --save_model --epochs 1000 --epochs 5000 --defence_auc "$defence_auc" --defence_positive --defence_sigma 0.0
#   done
# ) &


# (
#   for defence_auc in $(seq 0.93 -0.01 0.84)
#   do
#       python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --lr 0.001 --runs 1 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 2 --ln --res --save_model --epochs 1500 --epochs 5000 --defence_auc "$defence_auc" --defence_positive --defence_sigma 0.0
#   done
# ) &


# (
#   for defence_auc in $(seq 0.83 -0.01 0.74)
#   do
#       python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --lr 0.001 --runs 1 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 3 --ln --res --save_model --epochs 1500 --epochs 5000 --defence_auc "$defence_auc" --defence_positive --defence_sigma 0.0
#   done
# ) &


# (
#   for defence_auc in $(seq 0.73 -0.01 0.64)
#   do
#       python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --lr 0.001 --runs 1 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 4 --ln --res --save_model --epochs 1500 --epochs 5000 --defence_auc "$defence_auc" --defence_positive --defence_sigma 0.0
#   done
# ) &


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------


# (
#   for defence_sigma in $(seq 0.1 0.1 1.0)
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1
#   done
# ) &


# (
#   for defence_sigma in $(seq 1.1 0.1 2.0)
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 2 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1
#   done
# ) &


# (
#   for defence_sigma in $(seq 2.1 0.1 3.0)
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 3 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1
#   done
# ) &


# (
#   for defence_sigma in $(seq 3.1 0.1 4.0)
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 4 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1
#   done
# ) &


# (
#   for defence_sigma in $(seq 4.1 0.1 5.0)
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 5 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 1
#   done
# )


# (
#   for defence_sigma in $(seq 0.1 0.1 1.0)
#   do
#       python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 3
#   done
# ) &


# (
#   for defence_sigma in $(seq 1.1 0.1 2.0)
#   do
#       python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 2 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 3
#   done
# ) &


# (
#   for defence_sigma in $(seq 2.1 0.1 3.0)
#   do
#       python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 3 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 3
#   done
# ) &


# (
#   for defence_sigma in $(seq 3.1 0.1 4.0)
#   do
#       python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 4 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 3
#   done
# ) &


# (
#   for defence_sigma in $(seq 4.1 0.1 5.0)
#   do
#       python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 5 --runs 2 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma" --defence_k 3
#   done
# )


# (
#   for defence_sigma in $(seq 1.0 0.1 2.0)
#   do
#       python main.py --gnn gcn --dataset wikics --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --device 4 --ln --save_model --epochs 1000 --epochs 2000  --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in $(seq 2.1 0.1 3.0)
#   do
#       python main.py --gnn gcn --dataset wikics --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --device 4 --ln --save_model --epochs 1000 --epochs 2000  --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.6 0.1 1.0)
#   do
#       python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --lr 0.001 --runs 1 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 0 --ln --res --save_model --epochs 1500 --epochs 10000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in $(seq 1.1 0.1 1.5)
#   do
#       python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --lr 0.001 --runs 1 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 2 --ln --res --save_model --epochs 1500 --epochs 10000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in $(seq 1.6 0.1 2.0)
#   do
#       python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --lr 0.001 --runs 1 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 3 --ln --res --save_model --epochs 1500 --epochs 10000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.9 0.1 1.2)
#   do
#       python main.py --gnn gcn --dataset amazon-computer --hidden_channels 512 --lr 0.001 --runs 1 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device 5 --ln --save_model --epochs 1000 --epochs 7000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.5 0.1 1.4)
#   do
#       python main.py --gnn gcn  --dataset squirrel --lr 0.01 --local_layers 4 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --device 0 --runs 10 --bn --res --save_model --epochs 500 --epochs 1000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in $(seq 1.5 0.1 2.4)
#   do
#       python main.py --gnn gcn  --dataset squirrel --lr 0.01 --local_layers 4 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --device 2 --runs 10 --bn --res --save_model --epochs 500 --epochs 1000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in $(seq 2.5 0.1 3.0)
#   do
#       python main.py --gnn gcn  --dataset squirrel --lr 0.01 --local_layers 4 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --device 3 --runs 10 --bn --res --save_model --epochs 500 --epochs 1000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in $(seq 0.1 0.2 0.9)
#   do
#       python main.py --gnn gcn --dataset chameleon --lr 0.005 --local_layers 5 --hidden_channels 512 --weight_decay 0.001 --dropout 0.2 --device 0 --runs 10 --save_model --epochs 200 --epochs 1000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in $(seq 1.1 0.2 1.9)
#   do
#       python main.py --gnn gcn --dataset chameleon --lr 0.005 --local_layers 5 --hidden_channels 512 --weight_decay 0.001 --dropout 0.2 --device 2 --runs 10 --save_model --epochs 200 --epochs 1000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in $(seq 2.1 0.2 2.9)
#   do
#       python main.py --gnn gcn --dataset chameleon --lr 0.005 --local_layers 5 --hidden_channels 512 --weight_decay 0.001 --dropout 0.2 --device 3 --runs 10 --save_model --epochs 200 --epochs 1000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# )


# (
#   for defence_sigma in 3.0
#   do
#       python main.py --gnn gcn --dataset chameleon --lr 0.005 --local_layers 5 --hidden_channels 512 --weight_decay 0.001 --dropout 0.2 --device 3 --runs 10 --save_model --epochs 200 --epochs 1000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# )


# (
#   for defence_sigma in $(seq 10 10 50)
#   do
#       python main.py --gnn gcn --dataset amazon-photo --hidden_channels 256 --lr 0.001 --runs 1 --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device 0 --ln --res --save_model --epochs 1000 --epochs 7000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in $(seq 60 10 100)
#   do
#       python main.py --gnn gcn --dataset amazon-photo --hidden_channels 256 --lr 0.001 --runs 1 --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device 2 --ln --res --save_model --epochs 1000 --epochs 7000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in $(seq 110 10 150)
#   do
#       python main.py --gnn gcn --dataset amazon-photo --hidden_channels 256 --lr 0.001 --runs 1 --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device 3 --ln --res --save_model --epochs 1000 --epochs 7000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in $(seq 160 10 200)
#   do
#       python main.py --gnn gcn --dataset amazon-photo --hidden_channels 256 --lr 0.001 --runs 1 --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device 4 --ln --res --save_model --epochs 1000 --epochs 7000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# )


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------


# (
#   for defence_sigma in "${defence_sigma_values_1[@]}"
#   do
#       python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 3 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_auc in "${defence_auc_values_2[@]}"
#   do
#       python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 2 --runs 3 --save_model --epochs 500 --epochs 3000 --defence_auc "$defence_auc" --defence_positive --defence_sigma 0.0
#   done
# ) &


# (
#   for defence_sigma in "${defence_sigma_values_2[@]}"
#   do
#       python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 2 --runs 3 --save_model --epochs 500 --epochs 3000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_auc in "${defence_auc_values_3[@]}"
#   do
#       python main.py --gnn gcn --dataset pubmed --lr 0.005 --local_layers 2 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device 4 --runs 3 --save_model --epochs 500 --epochs 4000 --defence_auc "$defence_auc" --defence_positive --defence_sigma 0.0
#   done
# ) &


# (
#   for defence_sigma in "${defence_sigma_values_3[@]}"
#   do
#       python main.py --gnn gcn --dataset pubmed --lr 0.005 --local_layers 2 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device 5 --runs 3 --save_model --epochs 500 --epochs 4000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_auc in "${defence_auc_values_4[@]}"
#   do
#       python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 6 --ln --res --save_model --epochs 1500  --epochs 5000 --defence_auc "$defence_auc" --defence_positive --defence_sigma 0.0
#   done
# ) &


# (
#   for defence_sigma in "${defence_sigma_values_4[@]}"
#   do
#       python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --lr 0.001 --runs 2 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 7 --ln --res --save_model --epochs 1500  --epochs 5000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in "${defence_sigma_values_5[@]}"
#   do
#       python main.py --gnn gcn --dataset amazon-computer --hidden_channels 512 --lr 0.001 --runs 2 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device 3 --ln --save_model --epochs 1000 --epochs 7000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


# (
#   for defence_sigma in "${defence_sigma_values_6[@]}"
#   do
#       python main.py --gnn gcn --dataset amazon-photo --hidden_channels 256 --lr 0.001 --runs 2 --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device 3 --ln --res --save_model --epochs 1000 --epochs 7000 --defence_auc 1.00 --defence_positive --defence_sigma "$defence_sigma"
#   done
# ) &


wait


echo "All tasks are finished."
