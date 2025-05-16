# ENZYMES
python train.py --datadir=data --bmname=ENZYMES --cuda=3 --max-nodes=100 --num-classes=6

# ENZYMES - Diffpool
python train.py --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --cuda=1 --num-classes=6 --method=soft-assign

# DD
python train.py --datadir=data --bmname=DD --cuda=0 --max-nodes=500 --epochs=1000 --num-classes=2

# DD - Diffpool
python train.py --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=64 --output-dim=64 --cuda=1 --num-classes=2 --method=soft-assign


cd diffpool-master-egat ; conda activate pytorch




tmux copy-mode -t exp
tmux send-keys -t exp C-c




# ENZYMES
python train.py --datadir data --bmname ENZYMES --num-classes 6 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 600 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 0 --pooling_size_real 8 --method "soft-assign" --device "cuda:0"
python train.py --datadir data --bmname ENZYMES --num-classes 6 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 600 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 0 --pooling_size_real 8 --method "soft-assign-det" --device "cuda:0"
python train.py --datadir data --bmname ENZYMES --num-classes 6 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 600 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 0 --pooling_size_real 8 --method "ice" --device "cuda:0"
python train.py --datadir data --bmname ENZYMES --num-classes 6 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 600 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 0 --pooling_size_real 8 --method "soft-assign-det-ice" --device "cuda:0"
python train.py --datadir data --bmname ENZYMES --num-classes 6 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 600 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 0 --pooling_size_real 8 --method "soft-assign-det-ice-svdonly" --device "cuda:0"
python train.py --datadir data --bmname ENZYMES --num-classes 6 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 600 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 0 --pooling_size_real 8 --method "soft-assign-det-ice-svd" --device "cuda:0"


# Synthie
python train.py --datadir data --bmname Synthie --num-classes 4 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 200 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "soft-assign" --device "cuda:0"
python train.py --datadir data --bmname Synthie --num-classes 4 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 200 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "soft-assign-det" --device "cuda:0"
python train.py --datadir data --bmname Synthie --num-classes 4 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 200 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "ice" --device "cuda:0"
python train.py --datadir data --bmname Synthie --num-classes 4 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 200 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "soft-assign-det-ice" --device "cuda:0"
python train.py --datadir data --bmname Synthie --num-classes 4 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 200 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "soft-assign-det-ice-svdonly" --device "cuda:0"
python train.py --datadir data --bmname Synthie --num-classes 4 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 200 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "soft-assign-det-ice-svd" --device "cuda:0"


# DD
python train.py --datadir data --bmname DD --num-classes 2 --max-nodes 3000 --hidden-dim 64 --output-dim 64 --num_epochs 20 --batch_size 10 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 100 --method "soft-assign" --device "cuda:0"
python train.py --datadir data --bmname DD --num-classes 2 --max-nodes 3000 --hidden-dim 64 --output-dim 64 --num_epochs 20 --batch_size 10 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 100 --method "soft-assign-det" --device "cuda:0"
python train.py --datadir data --bmname DD --num-classes 2 --max-nodes 3000 --hidden-dim 64 --output-dim 64 --num_epochs 20 --batch_size 10 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 100 --method "ice" --device "cuda:0"
python train.py --datadir data --bmname DD --num-classes 2 --max-nodes 3000 --hidden-dim 64 --output-dim 64 --num_epochs 20 --batch_size 10 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 100 --method "soft-assign-det-ice" --device "cuda:0"
python train.py --datadir data --bmname DD --num-classes 2 --max-nodes 3000 --hidden-dim 64 --output-dim 64 --num_epochs 20 --batch_size 10 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 100 --method "soft-assign-det-ice-svdonly" --device "cuda:0"
python train.py --datadir data --bmname DD --num-classes 2 --max-nodes 3000 --hidden-dim 64 --output-dim 64 --num_epochs 20 --batch_size 10 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 100 --method "soft-assign-det-ice-svd" --device "cuda:0"


# NCI1
python train.py --datadir data --bmname NCI1 --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 200 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "soft-assign" --device "cuda:0"
python train.py --datadir data --bmname NCI1 --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 200 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "soft-assign-det" --device "cuda:0"
python train.py --datadir data --bmname NCI1 --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 200 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "ice" --device "cuda:0"
python train.py --datadir data --bmname NCI1 --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 200 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "soft-assign-det-ice" --device "cuda:0"
python train.py --datadir data --bmname NCI1 --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 200 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "soft-assign-det-ice-svdonly" --device "cuda:0"
python train.py --datadir data --bmname NCI1 --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 200 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "soft-assign-det-ice-svd" --device "cuda:0"


# PROTEINS
python train.py --datadir data --bmname PROTEINS --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "soft-assign" --device "cuda:0"
python train.py --datadir data --bmname PROTEINS --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 0 --pooling_size_real 8 --method "soft-assign-det" --device "cuda:0"
python train.py --datadir data --bmname PROTEINS --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "ice" --device "cuda:0"
python train.py --datadir data --bmname PROTEINS --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 0 --pooling_size_real 8 --method "soft-assign-det-ice" --device "cuda:0"
python train.py --datadir data --bmname PROTEINS --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 0 --pooling_size_real 8 --method "soft-assign-det-ice-svdonly" --device "cuda:0"
python train.py --datadir data --bmname PROTEINS --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 0 --pooling_size_real 8 --method "soft-assign-det-ice-svd" --device "cuda:0"


# MUTAG
python train.py --datadir data --bmname Mutagenicity --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "soft-assign" --device "cuda:0"
python train.py --datadir data --bmname Mutagenicity --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 0 --pooling_size_real 8 --method "soft-assign-det" --device "cuda:0"
python train.py --datadir data --bmname Mutagenicity --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "ice" --device "cuda:0"
python train.py --datadir data --bmname Mutagenicity --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 0 --pooling_size_real 8 --method "soft-assign-det-ice" --device "cuda:0"
python train.py --datadir data --bmname Mutagenicity --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 0 --pooling_size_real 8 --method "soft-assign-det-ice-svdonly" --device "cuda:0"
python train.py --datadir data --bmname Mutagenicity --num-classes 2 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 0 --pooling_size_real 8 --method "soft-assign-det-ice-svd" --device "cuda:0"

# COLLAB
#python train.py --datadir data --bmname COLLAB --num-classes 3 --max-nodes 1000 --hidden-dim 30 --output-dim 30 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "soft-assign" --device "cuda:0"
python train.py --datadir data --bmname COLLAB --num-classes 3 --max-nodes 1000 --hidden-dim 64 --output-dim 64 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "soft-assign" --device "cuda:0"
python train.py --datadir data --bmname COLLAB --num-classes 3 --max-nodes 1000 --hidden-dim 64 --output-dim 64 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "soft-assign-det" --device "cuda:0"
python train.py --datadir data --bmname COLLAB --num-classes 3 --max-nodes 1000 --hidden-dim 64 --output-dim 64 --num_epochs 100 --batch_size 30 --linkpred --lr 0.001 --edge_pool_weight 0.5 --X_concat_Singular 1 --use_simple_gat 1 --egat_hidden_dims 128 128 --egat_dropout 0.6 --egat_alpha 0.2 --egat_num_heads 1 3 --DSN 1 --pooling_size_real 8 --method "ice" --device "cuda:0"
