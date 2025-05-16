import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorboardX
from tensorboardX import SummaryWriter

import argparse
import os
import pickle
import random
import shutil
import time

import cross_val
import encoders
import gen.feat as featgen
import gen.data as datagen
from graph_sampler import GraphSampler
import load_data
import util

import warnings
from train import train, evaluate
NUM_TESTS = 10
def prepare_data(graphs, args, test_graphs=None, max_nodes=0):
    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1 - args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx: test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx:]
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs),
          '; Num testing graphs: ', len(test_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
          max([G.number_of_nodes() for G in graphs]), ', '
                                                      "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
          ', '
          "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    if args.method == 'ice' or args.method == 'soft-assign-det-ice':
        # calculate the actual max_num_nodes as well as max_num_nodes_after_pool
        max_nodes = 1

    # minibatch
    print("======================================")
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(test_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
        dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim


def syn_community1v2(args, writer=None, export_graphs=False):
    print("========== syn_community1v2 ========")
    # data
    graphs1 = datagen.gen_ba(range(40, 60), range(4, 5), 500,
                             featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float)))
    for G in graphs1:
        G.graph['label'] = 0
    if export_graphs:
        util.draw_graph_list(graphs1[:16], 4, 4, 'figs/ba')

    graphs2 = datagen.gen_2community_ba(range(20, 30), range(4, 5), 500, 0.3,
                                        [featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))])
    for G in graphs2:
        G.graph['label'] = 1
    if export_graphs:
        util.draw_graph_list(graphs2[:16], 4, 4, 'figs/ba2')

    graphs = graphs1 + graphs2

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(graphs, args)
    if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
            args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
            bn=args.bn, linkpred=args.linkpred, assign_input_dim=assign_input_dim).to(args.device)
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
                                           args.num_gc_layers, bn=args.bn).to(args.device)
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                                         args.num_gc_layers, bn=args.bn).to(args.device)

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
          writer=writer)


def syn_community2hier(args, writer=None):
    print("========== syn_community2hier ========")
    # data
    feat_gen = [featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))]
    graphs1 = datagen.gen_2hier(1000, [2, 4], 10, range(4, 5), 0.1, 0.03, feat_gen)
    graphs2 = datagen.gen_2hier(1000, [3, 3], 10, range(4, 5), 0.1, 0.03, feat_gen)
    graphs3 = datagen.gen_2community_ba(range(28, 33), range(4, 7), 1000, 0.25, feat_gen)

    for G in graphs1:
        G.graph['label'] = 0
    for G in graphs2:
        G.graph['label'] = 1
    for G in graphs3:
        G.graph['label'] = 2

    graphs = graphs1 + graphs2 + graphs3

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(graphs, args)

    if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
            args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
            bn=args.bn, linkpred=args.linkpred, args=args, assign_input_dim=assign_input_dim).to(args.device)
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
                                           args.num_gc_layers, bn=args.bn, args=args,
                                           assign_input_dim=assign_input_dim).to(args.device)
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                                         args.num_gc_layers, bn=args.bn, args=args).to(args.device)
    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
          writer=writer)


def pkl_task(args, feat=None):
    print("========== pkl_task ========")
    with open(os.path.join(args.datadir, args.pkl_fname), 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    graphs = data[0]
    labels = data[1]
    test_graphs = data[2]
    test_labels = data[3]

    for i in range(len(graphs)):
        graphs[i].graph['label'] = labels[i]
    for i in range(len(test_graphs)):
        test_graphs[i].graph['label'] = test_labels[i]

    if feat is None:
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)
        for G in test_graphs:
            featgen_const.gen_node_features(G)

    train_dataset, test_dataset, max_num_nodes = prepare_data(graphs, args, test_graphs=test_graphs)
    model = encoders.GcnEncoderGraph(
        args.input_dim, args.hidden_dim, args.output_dim, args.num_classes,
        args.num_gc_layers, bn=args.bn).to(args.device)
    train(train_dataset, model, args, test_dataset=test_dataset)
    evaluate(test_dataset, model, args, 'Validation')


def benchmark_task(args, writer=None, feat='node-label'):
    print("========== benchmark_task ========")
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)

    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in graphs[0].node[0]:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                G.node[u]['feat'] = np.array(G.node[u]['label'])
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = \
        prepare_data(graphs, args, max_nodes=args.max_nodes)
    if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
            args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
            bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
            assign_input_dim=assign_input_dim).to(args.device)
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(args.device)
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(args.device)

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
          writer=writer)
    evaluate(test_dataset, model, args, 'Validation')


def benchmark_task_val(args, writer=None, feat='node-label', log_dir=None):
    print("========== benchmark_task_val ========")
    all_vals = []
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)

    example_node = util.node_dict(graphs[0])[0]

    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in example_node:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    all_results = []
    for i in range(NUM_TESTS):

        print(f"============ test {i} ==============")

        train_dataset, val_dataset, max_num_nodes, max_num_clusters_fordet, input_dim, assign_input_dim = \
            cross_val.prepare_val_data(graphs, args, i, max_nodes=args.max_nodes)
        if args.method == 'ice':
            print('Method: ice')
            model = encoders.ICE_SoftPoolingGcnEncoder(
                max_num_nodes,
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, linkpred=args.linkpred, assign_input_dim=assign_input_dim,
                use_simple_gat=args.use_simple_gat, egat_hidden_dims=args.egat_hidden_dims,
                egat_dropout=args.egat_dropout, egat_alpha=args.egat_alpha, egat_num_heads=args.egat_num_heads,
                pooling_size_real=args.pooling_size_real, DSN=args.DSN, device=args.device, args=args).to(args.device)

        elif args.method == 'soft-assign':

            print('Method: soft-assign')
            model = encoders.SoftPoolingGcnEncoder(
                max_num_nodes,
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, dropout=args.dropout, linkpred=args.linkpred,
                assign_input_dim=assign_input_dim, device=args.device, args=args).to(args.device)
        elif args.method == 'soft-assign-det':

            print('Method: soft-assign-det')
            model = encoders.DET_SoftPoolingGcnEncoder(
                max_num_nodes, max_num_clusters_fordet,
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, dropout=args.dropout, linkpred=args.linkpred,
                assign_input_dim=assign_input_dim, device=args.device, args=args).to(args.device)
        elif args.method == 'soft-assign-det-ice':

            print('Method: soft-assign-det-ice')
            model = encoders.ICE_DET_SoftPoolingGcnEncoder(
                max_num_nodes, max_num_clusters_fordet,
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, linkpred=args.linkpred, assign_input_dim=assign_input_dim,
                use_simple_gat=args.use_simple_gat, egat_hidden_dims=args.egat_hidden_dims,
                egat_dropout=args.egat_dropout, egat_alpha=args.egat_alpha, egat_num_heads=args.egat_num_heads,
                pooling_size_real=args.pooling_size_real, DSN=args.DSN, device=args.device, args=args).to(args.device)
        elif args.method == 'soft-assign-det-ice-svdonly':

            print('Method: soft-assign-det-ice-svdonly')
            model = encoders.SVD_DET_SoftPoolingGcnEncoder(
                max_num_nodes, max_num_clusters_fordet,
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, linkpred=args.linkpred, assign_input_dim=assign_input_dim,
                edge_pool_weight=args.edge_pool_weight, X_concat_Singular=args.X_concat_Singular,
                device=args.device, args=args).to(args.device)
        elif args.method == 'soft-assign-det-ice-svd':

            print('Method: soft-assign-det-ice-svd+ce')
            model = encoders.ICE_SVD_DET_SoftPoolingGcnEncoder(
                max_num_nodes, max_num_clusters_fordet,
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, linkpred=args.linkpred, assign_input_dim=assign_input_dim,
                use_simple_gat=args.use_simple_gat, egat_hidden_dims=args.egat_hidden_dims,
                egat_dropout=args.egat_dropout, egat_alpha=args.egat_alpha, egat_num_heads=args.egat_num_heads,
                pooling_size_real=args.pooling_size_real, DSN=args.DSN,
                edge_pool_weight=args.edge_pool_weight, X_concat_Singular=args.X_concat_Singular,
                device=args.device, args=args).to(args.device)
        elif args.method == 'base-set2set':
            print('Method: base-set2set')
            model = encoders.GcnSet2SetEncoder(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(args.device)
        else:
            print('Method: base')
            model = encoders.GcnEncoderGraph(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(args.device)

        model, train_accs, val_accs, test_accs, best_val_result = train(train_dataset, model, args,
                                                                        val_dataset=val_dataset, test_dataset=None,
                                                                        writer=writer)
        all_vals.append(np.array(val_accs))

        all_results.append({"train_accs": train_accs, "val_accs": val_accs, "test_accs": test_accs,
                            "best_val_result": best_val_result["acc"], "args": args})

        os.makedirs(f"{log_dir}", exist_ok=True)
        with open(f"{log_dir}/full_result", 'wb') as f:
            pickle.dump(all_results, f)

    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))