"""This module provides various functions used to read/write and generate the data structures used for Path Learn"""

import networkx as nx
import random as rnd
import numpy as np
import pandas as pd
import os


def find_single_paths(G, node, lim, paths_lim=float('inf')):
    """
    :param G: A NetworkX graph.
    :param node: A node v.
    :param lim: Maximum number of steps.
    :param paths_lim: Maximum number of paths.
    :return: All paths up to lim steps, starting from node v.
    """

    paths = []
    to_extend = [[node]]

    while to_extend and len(paths) < paths_lim:

        cur_path = to_extend.pop(0)
        paths.append(cur_path)

        if len(cur_path) < 2 * lim + 1:
            for neigh in G[cur_path[-1]]:
                if neigh not in cur_path:
                    for rel_id in G[cur_path[-1]][neigh]:
                        ext_path = list(cur_path)
                        ext_path.append(rel_id)
                        ext_path.append(neigh)
                        to_extend.append(ext_path)

    return paths[1:]


def has_circles(path):
    """
    :param path: A sequence of node/edges
    :return: True if the path contains circles
    """

    nodes = set()
    for i, n in enumerate(path):
        if i % 2 == 0:
            if n in nodes:
                return True
            else:
                nodes.add(n)
    return False


def find_paths_between(G, start, end, length):
    """
    Finds all paths up to a given length between two nodes.

    :param G: NetworkX graph.
    :param start: Start node.
    :param end: End node.
    :param length: Maximum path length.
    :return: A set with all paths up to *length* fron *start* to *ends*.
    """

    if length % 2 == 0:
        length1 = length / 2
        length2 = length / 2
    else:
        length1 = int(length / 2) + 1
        length2 = int(length / 2)

    paths1 = find_single_paths(G, start, length1)
    paths2 = find_single_paths(G, end, length2)

    path2_ind = {}
    for path2 in paths2:
        if path2[-1] not in path2_ind:
            path2_ind[path2[-1]] = []
        path2_ind[path2[-1]].append(path2)

    full_paths = set()

    if end in G[start]:
        for edge_id in G[start][end]:
            full_paths.add((start, edge_id, end))

    for path1 in paths1:

        try:
            ext_paths = path2_ind[path1[-1]]
        except:
            ext_paths = []

        for ext_path in ext_paths:
            full_path = tuple(path1 + list(reversed(ext_path[0:-1])))
            if not has_circles(full_path):
                full_paths.add(full_path)

    return full_paths


def find_pair_paths(G, all_pairs, length, rev_rel=None):
    """
    Finds paths beetween a collection of node pairs.

    :param G: NetworkX graph.
    :param all_pairs: A collection of node pairs.
    :param length: Maximum path length
    :param rev_rel: Type of reverse relation, for directed graphs.
    :return: A two level dictionary with the paths for each pair.
    """

    T = {}

    if not rev_rel:
        rev_rel = all_pairs[0][1]

    for count, pair in enumerate(all_pairs):

        print('finding paths: ' + str(count) + '/' + str(len(all_pairs)))

        start = pair[0]
        rel = pair[1]
        end = pair[2]

        rev_pair = (end, rev_rel, start)

        if start not in T:
            T[start] = {}
        if end not in T[start]:
            T[start][end] = set()

        # paths = nx.all_simple_paths(G,start,end,length)
        paths = find_paths_between(G, start, end, length)

        for path in paths:
            path_ext = []
            dirty = False
            for i in range(1, len(path) - 1):
                if i % 2 == 0:
                    path_ext.append(path[i])
                else:
                    step_rel = G[path[i - 1]][path[i + 1]][path[i]]['type']
                    step_pair = (path[i - 1], step_rel, path[i + 1])
                    if (step_pair[0] != pair[0] or step_pair[1] != pair[1] or step_pair[2] != pair[2]) and (
                            step_pair[0] != rev_pair[0] or step_pair[1] != rev_pair[1] or step_pair[2] !=
                            rev_pair[2]):
                        path_ext.append(step_rel)
                    else:
                        dirty = True
                        break
            if not dirty:
                T[start][end].add(tuple(path_ext))

    print('returning')
    return T


def find_single_pair_paths(G, pair, length, rev_rel=None):
    """
    Finds all paths for a single pair of nodes.

    :param G: NetworkX graph.
    :param pair: The node pair.
    :param length: Maximum length.
    :param rev_rel: Reverse relation type, for directed graphs.
    :return: All paths up to *length( between the pair.
    """

    if not rev_rel:
        rev_rel = pair[1]

    start = pair[0]
    rel = pair[1]
    end = pair[2]

    rev_pair = [end, rev_rel, start]

    paths = find_paths_between(G, start, end, length)
    paths_out = set()

    for path in paths:
        path_ext = []
        dirty = False
        for i in range(1, len(path) - 1):
            if i % 2 == 0:
                path_ext.append(path[i])
            else:
                step_rel = G[path[i - 1]][path[i + 1]][path[i]]['elabel']
                step_pair = [path[i - 1], step_rel, path[i + 1]]
                if step_pair != pair and step_pair != rev_pair:
                    path_ext.append(step_rel)
                else:
                    dirty = True
                    break
        if not dirty:
            paths_out.add(tuple(path_ext))

    return paths_out


def add_paths(G, pair_set, steps, rev_rel, T):
    """
    Adds new paths to path dictionary T.

    :param G: Networkx graph.
    :param pair_set: A collection of pairs.
    :param steps: Maximum path length.
    :param rev_rel: Reverse relation type, for directed graphs.
    :param T: A path dictionary T.
    :return: A path dictionary T that includes paths for pairs in pair_set.
    """

    T_new = find_pair_paths(G, pair_set, steps, rev_rel)
    for u in T_new:
        if u not in T:
            T[u] = {}
        for v in T_new[u]:
            if v not in T[u]:
                T[u][v] = set()
            for path in T_new[u][v]:
                T[u][v].add(path)
    return T


def add_Ts(T0,T1):
    """
    Merges two path dictionaries.

    :param T0: A path dictionary.
    :param T1: A path dictionary.
    :return: A merged path dictionary.
    """

    for u in T1:
        if u not in T0:
            T0[u] = {}
        for v in T1[u]:
            if v not in T0[u]:
                T0[u][v] = set()
            for path in T1[u][v]:
                T0[u][v].add(path)
    return T0


def graph_to_files(G, path):
    """
    Saves graph to file.

    :param G: NetworkX graph.
    :param path: Folder path.
    """

    nodes = list(G.nodes)
    node_types = [G.nodes[n]['type'] for n in nodes]
    for type in set(node_types):
        print('writing type '+str(type))
        node_feats = []
        nodes_of_type = [n for n in G if G.nodes[n]['type']==type]
        for n in nodes_of_type:
            node_feats.append([n] + (G.nodes[n]['features'] if 'features' in G.nodes[n] else [0]))
        column_names = ['id'] + ['feat_'+str(i) for i in range(len(node_feats[0])-1)]
        pd.DataFrame(node_feats,columns=column_names).fillna(0).to_csv(path+'/nodes/'+str(type)+'.csv',index=False)
    print('writing relations')
    edges = list(G.edges)
    edge_feats = []
    for e in edges:
        edge_feats.append([e[0], e[1], G[e[0]][e[1]][int(e[2])]['type']] + (G[e[0]][e[1]][int(e[2])]['features'] if 'features' in G[e[0]][e[1]][int(e[2])] else [0]))
    column_names = ['src', 'dst', 'type'] + ['feat_' + str(i) for i in range( max( [len(ef) for ef in edge_feats] ) - 3)]
    pd.DataFrame(edge_feats, columns=column_names).fillna(0).to_csv(path+'/relations/relations.csv',index=False)


def graph_from_files(path):
    """
    Reads graph from file.

    :param path: folder path.
    :return: NetworkX graph.
    """

    G = nx.MultiDiGraph()
    for file in os.listdir(path+'/nodes'):
        print('loading '+file)
        node_type = file.split('.')[-2]
        nodes = pd.read_csv(path+'/nodes/'+file,dtype={0: str})
        for i, row in nodes.iterrows():
            row = list(row)
            G.add_node(str(row[0]),type=node_type,features=row[1:])
    print('loading relations')
    edges = pd.read_csv(path+'/relations/relations.csv',dtype={0: str,1: str,2: str})
    for i, row in edges.iterrows():
        row = list(row)
        G.add_edge(str(row[0]),str(row[1]),type=str(row[2]),features=row[3:])
    return G


# def make_small_data(G):
#     nodes = np.array(list(G.nodes))
#     node_types = np.array([G.nodes[n]['type'] for n in nodes])
#     sel_nodes = set()
#     for type in set(node_types):
#         print('writing type ' + str(type))
#         node_feats = []
#         for n in nodes[node_types == type][0:10000]:
#             sel_nodes.add(n)
#             node_feats.append([n] + G.nodes[n]['features'])
#         column_names = ['id'] + ['feat_' + str(i) for i in range(len(node_feats[0]) - 1)]
#         pd.DataFrame(node_feats, columns=column_names).fillna(0).to_csv('../data/small/' + '/nodes/' + str(type) + '.csv', index=False)
#     print('writing relations')
#     edges = list(G.edges)
#     edge_feats = []
#     for e in edges:
#         if e[0] in sel_nodes and e[1] in sel_nodes:
#             edge_feats.append([e[0], e[1], G[e[0]][e[1]][e[2]]['type']] + G[e[0]][e[1]][e[2]]['features'])
#     column_names = ['src', 'dst', 'type'] + ['feat_' + str(i) for i in range(max([len(ef) for ef in edge_feats]) - 3)]
#     pd.DataFrame(edge_feats, columns=column_names).fillna(0).to_csv('../data/small/' + '/relations/relations.csv', index=False)


def types_from_files(path):
    """
    Gets node/edge types of graph files.

    :param path: folder path.
    :return: dict with node/edge types.
    """

    node_types = []
    for file in os.listdir(path+'/nodes'):
        node_types.append(file.split('.')[0])
    edge_types = list(set(pd.read_csv(path+'/relations/relations.csv').iloc[:,2].astype(str)))
    return {'node_types': node_types, 'edge_types': edge_types}


def add_neg_samples(G, pos_pairs, samp_size, steps):
    """
    Performs negative sampling.

    :param G: NetworkX graph.
    :param pos_pairs: A collection of pairs with edges.
    :param samp_size: Negative samples per existing edge.
    :param steps: Max length of random walk.
    :return: A list of positive and negative node pairs and their labels.
    """

    pairs = []
    labels = []
    print(pos_pairs[0])
    tail_type = G.nodes[pos_pairs[0][2]]['type']
    print(tail_type)
    tails = set([n for n in G.nodes if G.nodes[n]['type'] == tail_type])
    for i, (head,rel_id,tail) in enumerate(pos_pairs):
        print('neg samples {}/{}'.format(i, len(pos_pairs)))
        pos = set(G[head])
        near = set(nx.ego_graph(G, head, steps).nodes) - pos
        #near = set([n for n in subG if G.nodes[n] == tail_type]) - pos
        # for j in range(1,steps):
        #     near -= set(nx.ego_graph(G, head, steps-j).nodes)
        near_samp = rnd.sample(near, min(len(near), samp_size))
        far = tails - pos - near
        far_samp = rnd.sample(far, min(len(far), samp_size))
        pairs.append([head, rel_id, tail])
        labels.append(1)
        for tail in near_samp:
            pairs.append([head, rel_id, tail])
            labels.append(0)
        for tail in far_samp:
            pairs.append([head, rel_id, tail])
            labels.append(0)
    return pairs, labels


def find_node_types(G, edge_type):
    """
    :param G: NetworkX graph.
    :param edge_type: Edge type.
    :return: Node types that correspond to the edge type.
    """

    for e in G.edges:
        if G[e[0]][e[1]][e[2]]['type'] == edge_type:
            u, v = e[0], e[1]
            break
    utype = G.nodes[u]['type']
    vtype = G.nodes[v]['type']
    try:
        if int(utype) > int(vtype):
            return utype, vtype
        else:
            return vtype, utype
    except:
        return utype, vtype


def find_candidate_type(G, edge_type, src_node):
    """
    :param G: NetworkX graph.
    :param edge_type: An edge type.
    :param src_node: A source node.
    :return: The node type that is connecter with edge_type to src_node.
    """

    stype = G.nodes[src_node]['type']
    for e in G.edges:
        if G[e[0]][e[1]][e[2]]['type'] == edge_type:
            u,v = e[0], e[1]
            break
    utype = G.nodes[u]['type']
    vtype = G.nodes[v]['type']
    return vtype if stype==utype else utype


def filter_pairs(test_pairs, test_labels, pair_filter):
    '''
    :param test_pairs: Node pairs.
    :param test_labels: Labels.
    :param pair_filter:  Filter set.
    :return: test_pairs and test_labels that do not exist in pair_filter
    '''
    new_pairs = []
    new_labels = []
    for pair, label in zip(test_pairs,test_labels):
        if tuple(pair) not in pair_filter:
            new_pairs.append(pair)
            new_labels.append(label)
    return test_pairs, test_labels


def make_train_data(G, edge_type, ntr, nvl, nts, steps=3, neg=5):
    """
    :param G: NetworkX graph.
    :param edge_type: Edge type.
    :param ntr: Number of positive training pairs.
    :param nvl: Number of positive validation pairs.
    :param nvs: Number of positive test pairs.
    :param steps: Maximum path length.
    :param neg: Negative samples pre positive egde.
    :return: train pairs, train labels, validation pairs, validation edges, path dictionary T
    """

    utype, vtype = find_node_types(G, edge_type)
    sel_edges = rnd.sample([[e[0],edge_type,e[1]] for e in G.edges if G[e[0]][e[1]][e[2]]['type'] == edge_type
                               and G.nodes[e[0]]['type'] == utype and G.nodes[e[1]]['type'] == vtype], ntr+nvl+nts)
    train_pairs = sel_edges[0:ntr]
    train_pairs, train_labels = add_neg_samples(G,train_pairs, neg, steps)
    T_tr = find_pair_paths(G, train_pairs, steps)
    val_pairs = sel_edges[ntr:ntr+nvl]
    val_pairs, val_labels = add_neg_samples(G, val_pairs, neg, steps)
    T_vl = find_pair_paths(G, val_pairs, steps)
    test_pairs = sel_edges[ntr + nvl:ntr + nvl + nts]
    test_pairs, test_labels = add_neg_samples(G, test_pairs, neg, steps)
    pair_filter = set([(u, r, v) for u, r, v in train_pairs + val_pairs])
    test_pairs, test_labels = filter_pairs(test_pairs, test_labels, pair_filter)
    T_ts = find_pair_paths(G, test_pairs, steps)
    T = add_Ts(T_ts, add_Ts(T_tr, T_vl))
    return train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels, T


