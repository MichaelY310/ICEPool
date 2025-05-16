python trainer_sep_args.py --dataset IMDB-BINARY --conv Edgefeat-GAT --batch_size 128 --final_dropout 0.0 --hidden_dim 64 --svdpool --egat_dropout 0.3 --egat_alpha 0.2 --num_ce_aggs 100 --num_svd_pools 100 --gpu 4

python trainer_sep_args.py --dataset IMDB-MULTI --conv Edgefeat-GAT --batch_size 128 --final_dropout 0.5 --hidden_dim 64 --lr-schedule --svdpool --egat_dropout 0.5 --egat_alpha 0.2 --num_ce_aggs 1 --num_svd_pools 2 --gpu 4

python trainer_sep_args.py --dataset COLLAB --conv Edgefeat-GAT --batch_size 128 --final_dropout 0.0 --hidden_dim 128 --svdpool --egat_dropout 0.5 --egat_alpha 0.2 --num_ce_aggs 100 --num_svd_pools 100 --gpu 4

python trainer_sep_args.py --dataset MUTAG --conv Edgefeat-GAT --batch_size 128 --final_dropout 0.5 --hidden_dim 64 --lr-schedule --svdpool --egat_dropout 0.5 --egat_alpha 0.2 --num_ce_aggs 100 --num_svd_pools 100 --gpu 4

python trainer_sep_args.py --dataset PROTEINS --conv Edgefeat-GAT --batch_size 128 --final_dropout 0.0 --hidden_dim 64 --svdpool --egat_dropout 0.5 --egat_alpha 0.2 --num_ce_aggs 100 --num_svd_pools 100 --gpu 4

python trainer_sep_args.py --dataset NCI1 --conv Edgefeat-GAT --batch_size 128 --final_dropout 0.5 --hidden_dim 64 --lr-schedule --svdpool --egat_dropout 0.5 --egat_alpha 0.2 --num_ce_aggs 100 --num_svd_pools 100 --gpu 4

python trainer_sep_args.py --dataset DD --conv Edgefeat-GAT --batch_size 128 --final_dropout 0.0 --hidden_dim 64 --lr-schedule --svdpool --egat_dropout 0.5 --egat_alpha 0.2 --num_ce_aggs 100 --num_svd_pools 100 --gpu 4
