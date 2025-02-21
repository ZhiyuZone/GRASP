## homophilic datasets

# Computer
python main.py --gnn gcn --dataset amazon-computer --hidden_channels 512 --epochs 1000 --lr 0.001 --runs 5 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device 0 --ln --save_model
python main.py --gnn sage --dataset amazon-computer --hidden_channels 64 --epochs 1000 --lr 0.001 --runs 5 --local_layers 4 --weight_decay 5e-5 --dropout 0.3 --device 2 --ln --save_model
python main.py --gnn gat --dataset amazon-computer --hidden_channels 64 --epochs 1000 --lr 0.001 --runs 5 --local_layers 2 --weight_decay 5e-5 --dropout 0.5 --device 3 --ln --save_model

# Photo
python main.py --gnn gcn --dataset amazon-photo --hidden_channels 256 --epochs 1000 --lr 0.001 --runs 5 --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device 5 --ln --res --save_model
python main.py --gnn sage --dataset amazon-photo --hidden_channels 64 --epochs 1000 --lr 0.001 --runs 5 --local_layers 6 --weight_decay 5e-5 --dropout 0.2 --device 6 --ln --res --save_model
python main.py --gnn gat --dataset amazon-photo --hidden_channels 64 --epochs 1000 --lr 0.001 --runs 5 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device 7 --ln --res --save_model

# CS
python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --epochs 1500 --lr 0.001 --runs 5 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 5 --ln --res --save_model
python main.py --gnn sage --dataset coauthor-cs --hidden_channels 512 --epochs 1500 --lr 0.001 --runs 5 --local_layers 2 --weight_decay 5e-4 --dropout 0.5 --device 6 --ln --res --save_model
python main.py --gnn gat --dataset coauthor-cs --hidden_channels 256 --epochs 1500 --lr 0.001 --runs 5 --local_layers 1 --weight_decay 5e-4 --dropout 0.3 --device 7 --ln --res --save_model

# Physics
python main.py --gnn gcn --dataset coauthor-physics --hidden_channels 64 --epochs 1500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 5 --ln --res --save_model
python main.py --gnn sage --dataset coauthor-physics --hidden_channels 64 --epochs 1500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.7 --device 6 --bn --res --save_model
python main.py --gnn gat --dataset coauthor-physics --hidden_channels 256 --epochs 1500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.7 --device 7 --bn --res --save_model

# WikiCS
python main.py --gnn gcn --dataset wikics --hidden_channels 256 --epochs 1000 --lr 0.001 --runs 5 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --device 0 --ln --save_model
python main.py --gnn sage --dataset wikics --hidden_channels 256 --epochs 1000 --lr 0.001 --runs 5 --local_layers 2 --weight_decay 0.0 --dropout 0.7 --device 2 --ln --save_model
python main.py --gnn gat --dataset wikics --hidden_channels 512 --epochs 1000 --lr 0.001 --runs 5 --local_layers 2 --weight_decay 0.0 --dropout 0.7 --device 3 --ln --res --save_model

# Cora
python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3  --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 5 --save_model
python main.py --gnn sage --dataset cora --lr 0.001 --local_layers 3  --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 2 --runs 5 --save_model
python main.py --gnn gat --dataset cora --lr 0.001 --local_layers 3  --hidden_channels 512 --weight_decay 5e-4 --dropout 0.2 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 3 --runs 5 --res --save_model

# CiteSeer
python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 0 --runs 5 --save_model
python main.py --gnn sage --dataset citeseer --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 0.01 --dropout 0.2 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 2 --runs 5 --save_model
python main.py --gnn gat --dataset citeseer --lr 0.001 --local_layers 3 --hidden_channels 256 --weight_decay 0.01 --dropout 0.5 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --seed 123 --device 3 --runs 5 --res --save_model

# PubMed
python main.py --gnn gcn --dataset pubmed --lr 0.005 --local_layers 2 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device 5 --runs 5 --save_model
python main.py --gnn sage --dataset pubmed --lr 0.005 --local_layers 4 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device 6 --runs 5 --save_model
python main.py --gnn gat --dataset pubmed --lr 0.01 --local_layers 2 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device 7 --runs 5 --save_model

## heterophilic datasets

# Amazon-Ratings
python main.py --gnn gcn --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 3 --local_layers 4 --weight_decay 0.0 --dropout 0.5 --device 4 --bn --res --save_model
python main.py --gnn sage --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 3 --local_layers 9 --weight_decay 0.0 --dropout 0.5 --device 5 --bn --res --save_model
python main.py --gnn gat --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 3 --local_layers 4 --weight_decay 0.0 --dropout 0.5 --device 7 --bn --res --save_model

# Minesweeper
python main.py --gnn gcn --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 5 --local_layers 12 --weight_decay 0.0 --dropout 0.2 --metric rocauc --device 0 --bn --res --save_model
python main.py --gnn sage --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 5 --local_layers 15 --weight_decay 0.0 --dropout 0.2 --metric rocauc --device 2 --bn --res --save_model
python main.py --gnn gat --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 5 --local_layers 15 --weight_decay 0.0 --dropout 0.2 --metric rocauc --device 3 --bn --res --save_model

# Roman-Empire
python main.py --gnn gcn --dataset roman-empire --pre_linear --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 3 --local_layers 9 --weight_decay 0.0 --dropout 0.5 --device 5 --bn --res --save_model
python main.py --gnn sage --dataset roman-empire --pre_linear --hidden_channels 256 --epochs 2500 --lr 0.001 --runs 3 --local_layers 9 --weight_decay 0.0 --dropout 0.3 --device 6 --bn --save_model
python main.py --gnn gat --dataset roman-empire --pre_linear --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 3 --local_layers 10 --weight_decay 0.0 --dropout 0.3 --device 7 --bn --res --save_model

# Questions
python main.py --gnn gcn --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 3 --local_layers 10 --weight_decay 0.0 --dropout 0.3 --metric rocauc --device 5 --res --save_model
python main.py --gnn sage --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 3 --local_layers 6 --weight_decay 0.0 --dropout 0.2 --metric rocauc --device 6 --ln --save_model
python main.py --gnn gat --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 3 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --metric rocauc --device 7 --ln --res --save_model

# Squirrel
python main.py --gnn gcn  --dataset squirrel --lr 0.01 --local_layers 4 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --device 0 --runs 10 --bn --res --save_model
python main.py --gnn sage  --dataset squirrel --lr 0.01 --local_layers 3 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --device 2 --runs 10 --bn --res --save_model
python main.py --gnn gat  --dataset squirrel --lr 0.005 --local_layers 7 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.5 --device 3 --runs 10 --bn --res --save_model

# Chameleon
python main.py --gnn gcn --dataset chameleon --lr 0.005 --local_layers 5 --hidden_channels 512 --weight_decay 0.001 --dropout 0.2 --device 0 --runs 10 --epochs 200 --save_model
python main.py --gnn sage --dataset chameleon --lr 0.01 --local_layers 4 --hidden_channels 256 --weight_decay 0.001 --dropout 0.7 --device 2 --runs 10 --epochs 200 --bn --res --save_model
python main.py --gnn gat --dataset chameleon --lr 0.01 --local_layers 2 --hidden_channels 256 --weight_decay 0.001 --dropout 0.7 --device 3 --runs 10 --epochs 200 --bn --res --save_model
