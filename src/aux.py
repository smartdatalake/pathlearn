import csv
import networkx as nx
import sys
import numpy as np
import time
import pandas as pd
import random as rnd
#from ampligraph.datasets import load_fb15k
import traceback
from datetime import datetime
#import mp_counts
from itertools import product
#import preprocessing as prc
#from statsmodels.stats.proportion import proportion_confint
#from matplotlib import pyplot as plt
import pickle
from multiprocessing import Pool
from scipy.stats import beta
import re


def get_node_id(node):
    return node


def get_rel_id(G,u,v):
    return G.get_edge_data(u,v)['type']


# def find_all_paths(G, lim):
#
#     #T = dict()
#     #lens = []
#     #node_list = list(G.nodes)
#     #for s in range(len(G)):
#     #    for t in range(s+1,len(G)):
#     #        lens.append(len(list(nx.all_simple_paths(G,node_list[s],node_list[t],lim))))
#     #        print(str(s+1) + '/' + str(len(G.nodes))+ ':' + str(np.sum(lens)))
#
#     #return lens
#
#     T = dict()
#     lens = []
#     node_list = list(G.nodes)
#
#     for n in node_list:
#         print(n)
#         T[n] = {}
#
#         paths = find_single_paths(G, n, lim)
#         for path in paths:
#             if path[-1] not in T[n]:
#                 T[n][path[-1]] = []
#             path_dict = {}
#             nodes = []
#             for i in range(1, len(path) - 1):
#                 nodes.append(get_node_id(path[i]))
#             rels = []
#             for i in range(0, len(path)-1):
#                 rels.append(get_rel_id(G, path[i], path[i+1]))
#
#             #starts with rel ends with rel
#             path_ext = []
#             node = False
#             rel_index = 0
#             node_index = 0
#             while rel_index<len(rels):
#                 if not node:
#                     path_ext.append(rels[rel_index])
#                     rel_index += 1
#                     node = True
#                 else:
#                     path_ext.append(nodes[node_index])
#                     node_index += 1
#                     node = False
#
#             T[n][path[-1]].append(path_ext)
#     return T


def make_synth(na=1000, nb=100, nr=100):
    G = nx.Graph()
    id = 0

    for i in range(na):
        G.add_node(id)
        G.node[id]['type'] = 'a'
        id += 1

    for i in range(na):
        G.add_node(id)
        G.node[id]['type'] = 'c'
        id += 1

    for i in range(int(nb / 2)):
        G.add_node(id)
        G.node[id]['type'] = 'b'
        G.node[id]['subtype'] = 'b1'
        id += 1

    for i in range(int(nb / 2)):
        G.add_node(id)
        G.node[id]['type'] = 'b'
        G.node[id]['subtype'] = 'b2'
        id += 1

    for b_id in range(int(2 * na), int(2 * na + nb / 2)):
        for _ in range(nr):
            a_id = np.random.randint(0, na)
            c_id = np.random.randint(na, 2 * na)
            G.add_edge(a_id, b_id, type=0)
            G.add_edge(b_id, c_id, type=1)
            G.add_edge(a_id, c_id, type=2)

    for b_id in range(int(2 * na + nb / 2), int(2 * na + nb)):
        for _ in range(nr):
            a_id = np.random.randint(0, na)
            c_id = np.random.randint(na, 2 * na)
            G.add_edge(a_id, b_id, type=0)
            G.add_edge(b_id, c_id, type=1)

   # nx.draw(G, with_labels = True)

    return G


def sample_full(ones, zeros, all_rel_ids, neg_factor):
    for rel_id in all_rel_ids:
        if len(zeros[rel_id]) < neg_factor*len(ones):
            return False
    return True


def has_edge_type(edges, targ_rel):
    for edge in edges:
        if edges[edge]['type'] == targ_rel:
            return True
    return False


def has_other_edge_types(edges,targ_rel):
    for edge in edges:
        if edges[edge]['type'] != targ_rel:
            return True
    return False


def init(na=1000,nb=100,nr=20,rel=2,path_lim=2):
    G = make_synth(na,nb,nr)
    T = find_all_paths(G,path_lim)
    labels = make_one_sets(G, path_lim, rel)
    return G,T,labels


def make_str(X):
    X_str = []
    for x in X:
        X_str.append((str(x[0]), str(x[1]), str(x[2])))
    return X_str


def get_ranks(model, X_test):
    ranks = np.zeros(len(X_test))
    for ind, x in enumerate(X_test):
        if(type(model)==PathE):
            ranks[ind] = pd.DataFrame(model.predict_train(x))[0].rank(ascending=False, method='first')[0]
        else:
            ranks[ind] = pd.DataFrame(model.predict_train(make_str(x)))[0].rank(ascending=False, method='first')[0]
    return ranks


def make_graph_from_triples(triples):
    triples = np.vstack([triples['train'], triples['valid'], triples['test']])
    G = nx.MultiGraph()
    rel_map = {}
    node_map = {}
    rel_id = 0
    node_id = 0
    for triple in triples:
        if triple[0] not in node_map:
            node_map[triple[0]] = node_id
            G.add_node(node_id)
            node_id += 1
        if triple[2] not in node_map:
            node_map[triple[2]] = node_id
            G.add_node(node_id)
            node_id +=1
        if triple[1] not in rel_map:
            rel_map[triple[1]] = rel_id
            rel_id += 1
        G.add_edge(node_map[triple[0]],node_map[triple[2]],type=rel_map[triple[1]])
    return G,rel_map,node_map


def check_paths(G,T,triples):

    for triple in triples:

        T_paths = T[triple[0]][triple[2]]
        nx_paths = nx.all_simple_paths(G,triple[0],triple[2],4)

        T_nodes=list()
        nx_nodes=list()

        for path in T_paths:
            path_nodes = []
            for i,v in enumerate(path):
                if i%2 == 1:
                    path_nodes.append(v)
            T_nodes.append(tuple(path_nodes))

        for path in nx_paths:
            print(path)
            if len(path)>2:
                nx_nodes.append(tuple(path[1:-1]))

        print(T_nodes)
        print(nx_nodes)

        for pn in T_nodes:
            if pn in nx_nodes:
                nx_nodes.remove(pn)
            else:
                raise Exception('problem')
        if len(nx_nodes)>0:
            raise Exception('problem')


def add_zeros_and_paths(G, one_triples, filter_zeros, filter_ones, near_zeros_num=20, far_zeros_num=20, steps=3, triples_limit=float('+Inf'), rev_rel=None):

    zero_sets = {triple: {} for triple in one_triples}
    filter_zeros = set(filter_zeros)
    all_test_labels = []

    for i,triple in enumerate(one_triples):
        if i == triples_limit:
            break
        print('adding zeros: '+str(i)+'/'+str(len(one_triples)))

        all_test_labels.append(triple)

        nzs = random_walk_sample(G, triple[0], steps, near_zeros_num, 10)
        nzs = set([(triple[0], triple[1], nz) for nz in nzs if (triple[0], triple[1], nz) not in filter_zeros and (triple[0], triple[1], nz) not in filter_ones])
        all_test_labels += list(nzs)

        fzs = rnd.sample(list(G.nodes), far_zeros_num)
        fzs = set([(triple[0], triple[1], fz) for fz in fzs if (triple[0], triple[1], fz) not in filter_zeros and (triple[0], triple[1], fz) not in filter_ones])
        all_test_labels += list(fzs)

        zero_sets[triple]['near_zeros'] = nzs
        zero_sets[triple]['far_zeros'] = fzs

    T_test = find_sampled_paths(G, all_test_labels, steps, rev_rel)

    return zero_sets, T_test, set(all_test_labels)


def remove_edge(G,triple,rev_rel):

    start,rel,end = triple
    edge_buf = {}

    try:
        edges = G[start][end]
    except:
        return edge_buf
    edge_id=None
    for id in edges:
        if edges[id]['type']==rel:
            edge_id = id
    if not edge_id:
        return edge_buf

    edge_buf['nodes'] = (start,end)
    edge_buf['id'] = edge_id
    edge_buf['atts'] = G[start][end][edge_id]
    G.remove_edge(start,end,edge_id)

    if rev_rel:
        rev_edges = G[end][start]
        for rev_id in rev_edges:
            if rev_edges[rev_id]['type'] == rev_rel:
                rev_edge_id = rev_id
        edge_buf['rev_id'] = rev_edge_id
        edge_buf['rev_atts'] = G[end][start][rev_edge_id]
    G.remove_edge(end, start, rev_edge_id)

    return edge_buf


def restore_edge(G,edge_buf):

    if 'nodes' not in edge_buf:
        return

    start,end = edge_buf['nodes']
    id = edge_buf['id']
    edge_atts = edge_buf['atts']

    G.add_edge(start,end,id)
    for att in edge_atts:
        G[start][end][id][att] = edge_atts[att]

    if 'rev_atts' in edge_buf:
        rev_edge_atts = edge_buf['rev_atts']
        rev_id = edge_buf['rev_id']
        G.add_edge(end, start, rev_id)
        for att in rev_edge_atts:
            G[end][start][rev_id][att] = rev_edge_atts[att]


def make_data_from_triples(triples,targ_rel,step_lim=2,train_size=0.8,val_size=0.1,train_near_zeros=8,train_far_zeros=4,test_near_zeros=150,test_far_zeros=50):
    G, rels, nodes = make_graph_from_triples(triples)
    rev_rel = None
    labels = make_one_sets(G, step_lim, targ_rel, rev_rel, train_size,val_size)
    all_ones = set(list(labels['train']['ones'])+list(labels['val']['ones'])+list(labels['test']['ones']))
    T = {}
    train_set, T_train, train_zeros = add_zeros_and_paths(G, labels['train']['ones'], filter_zeros=set(),
                                                          filter_ones=all_ones, near_zeros_num=train_near_zeros,
                                                          far_zeros_num=train_far_zeros, steps=step_lim,
                                                          rev_rel=rev_rel)
    T = add_Ts(T, T_train)
    val_set, T_val, val_zeros = add_zeros_and_paths(G, labels['val']['ones'], filter_zeros=train_zeros,
                                                    filter_ones=all_ones, near_zeros_num=test_near_zeros,
                                                    far_zeros_num=test_far_zeros, steps=step_lim, rev_rel=rev_rel)
    T = add_Ts(T, T_val)
    test_set, T_test, _ = add_zeros_and_paths(G, labels['test']['ones'],
                                              filter_zeros=set(list(train_zeros) + list(val_zeros)),
                                              filter_ones=all_ones, near_zeros_num=test_near_zeros,
                                              far_zeros_num=test_far_zeros, steps=step_lim, rev_rel=rev_rel)
    T = add_Ts(T, T_test)
    return G, T, rels, train_set, val_set, test_set, labels['train']['ones_ext']


def make_strings(triples):
    string_triples = []
    for triple in triples:
        string_triples.append([str(triple[0]),str(triple[1]),str(triple[2])])
    return string_triples


def mrr_hits(model,test_set,prnt = False):

    try:
        _ = model.Wv[0]
        model_type = 'pathe'
    except:
        try:
            _ = model.lg
            model_type = 'mpc'
        except:
            try:
                _ = model.entities
                model_type = 'pme'
            except:
                model_type = 'ampli'

    ranks = np.zeros(len(test_set))
    sizes = np.zeros(len(test_set))
    rel_ranks = np.zeros(len(test_set))
    for i, triple in enumerate(test_set):

        #print(str(i)+'/'+str(len(test_set)))

        try:
            near_zero_cands = test_set[triple]['near_zeros']
            far_zero_cands = test_set[triple]['far_zeros']
        except Exception as e:
            if print:
                print('error, first try: ' + str(e))
                traceback.print_exc()
            continue
        zeros = np.array(list(near_zero_cands)+list(far_zero_cands))
        zero_probs = np.zeros(len(zeros))

        try:
            if model_type == 'pathe':
                prob_one = model(*triple).detach().numpy()
            elif model_type == 'ampli':
                prob_one = model.predict_train((str(triple[0]), str(triple[1]), str(triple[2])))[0]
            elif model_type =='mpc':
                prob_one = model.predict_train(triple)
            elif model_type == 'pme':
                prob_one = model.predict_train(triple)
        except Exception as e:
            if print:
                print('error, second try: ' + str(e))
                traceback.print_exc()
            continue

        for j, zero_triple in enumerate(zeros):

            try:
                if model_type == 'pathe':
                    zero_probs[j] = model(*zero_triple).detach().numpy()
                elif model_type == 'ampli':
                    zero_probs[j] = model.predict_train((str(zero_triple[0]), str(zero_triple[1]), str(zero_triple[2])))[0]
                elif model_type == 'mpc':
                    zero_probs[j] = model.predict_train(zero_triple)
                elif model_type == 'pme':
                    zero_probs[j] = model.predict_train(triple)
            except Exception as e:
                if print:
                    print('error, third try: ' + str(e))
                    traceback.print_exc()
                zero_probs[j] = None

        sizes[i] = np.sum(~np.isnan(zero_probs))
        ranks[i] = np.sum(prob_one<zero_probs)+ int((np.sum(prob_one==zero_probs)/2)) + 1
        rel_ranks[i] = ranks[i]/sizes[i]

    mrr = np.mean(ranks[sizes>0])
    rmrr = np.mean(ranks[sizes>0])
    hits10 = np.mean(ranks[sizes>0]<10)

    return mrr, hits10, ranks, sizes


def make_conf_mat(model,labels):
    min_len = min(len(labels['ones']), len(labels['near_zeros']), len(labels['far_zeros']))
    triples_stacked = list(labels['ones'])[0:min_len] + list(labels['near_zeros'])[0:min_len] + list(
        labels['far_zeros'])[0:min_len]
    labels_stacked = np.hstack([np.ones(min_len), np.zeros(min_len), np.zeros(min_len)])
    preds = np.zeros(min_len * 3)

    for index, triple in enumerate(triples_stacked):
        preds[index] = 0 #model.forward(*triple)

    conf_mat = np.zeros((2,2))

    for ind,pred in enumerate(preds):
        conf_mat[int(labels_stacked[ind]),int(np.round(pred))] += 1

    return conf_mat


def try_fb_edges():
    fb = load_fb15k()
    results = {}
    test_sets ={}
    for i in range(100):
        try:
            print(i)
            _,T,rels,train_set,_,test_set,_ = make_data_from_triples(fb,i)
            print(list(rels.keys())[i])
            mpc = mp_counts.MpCount(train_set,T)
            res = mrr_hits(mpc,test_set)
            print(res[0])
            results[i] = res
            test_sets[i] = test_set
        except KeyboardInterrupt:
            return results,test_sets
        except Exception as e:
            traceback.print_exc()
    return results,test_sets


def boot_mean(mr1,mr2,r=1000):

    mr1 = np.array(mr1)
    mr2 = np.array(mr2)
    diff0 = abs(mr1.mean()-mr2.mean())
    len1 = len(mr1)
    pool =  np.hstack([mr1,mr2])
    diffs = np.zeros(r)

    for i in range(r):
        s_pool = np.random.choice(pool,len(pool))
        s_mr1 = s_pool[0:len1]
        s_mr2 = s_pool[len1:]
        diffs[i] = abs(s_mr1.mean()-s_mr2.mean())

    plt.hist(diffs,100)
    return np.sum(diffs>diff0)/r


def boot_pair_mean(mr1,mr2,r=1000):

    mr1 = np.array(mr1)
    mr2 = np.array(mr2)
    diffs = mr1-mr2
    diff0 = np.mean(diffs)

    rdiffs = np.zeros(r)
    for i in range(r):
        diff_samp = np.random.choice(diffs,len(diffs))
        rdiffs[i] = np.mean(diff_samp)

    plt.hist(rdiffs,100)
    print(diff0)
    sdifs = sorted(rdiffs)
    lo = sdifs[int(0.025*r)]
    hi = sdifs[int(0.975*r)]

    return np.sum(rdiffs>0)/r,np.sum(rdiffs<0)/r,(lo,hi)

def write_pme_files(triples):
    triples = np.array(triples)
    nodes = sorted(list(set(triples[:,0]).union(set(triples[:,2]))))
    rels = sorted(list(set(triples[:,1])))

    f = open("yelp.node", "wt")
    for node in nodes:
        f.write(str(node) + '\n')
    f.close()

    f = open("relation.node", "wt")
    for rel in rels:
        f.write(str(rel) + '\n')
    f.close()

    f = open("yelp_train.hin", "wt")
    f.write(str(len(triples)) + '\n')
    for triple in triples:
        f.write(str(triple[0])+' '+str(triple[1])+' '+str(triple[2]) + '\n')
    f.close()

    return nodes,rels


def get_pme():
    entities = []
    with open("/home/pant/Desktop/entity2vec.vec", "rt") as f:
        for line in f:
            entities.append([float(n) for n in line.strip('\t\n').split('\t')])
    entities = np.array(entities)
    d = entities.shape[1]
    relations = []
    with open("/home/pant/Desktop/A.vec", "rt") as f:
        for line in f:
            relations.append([float(n) for n in line.strip('\t\n').split('\t')])
    rho = int(len(relations)/d)
    relations = np.array(relations).reshape(rho,d,d)
    return entities,relations


def get_path_nodes(triple, T):

    v,rel,u = triple

    try:
        vu_paths = T[v][u]
    except:
        try:
            vu_paths = T[u][v]
        except:
            print('v,u not int T')
            return

    vu_path_nodes = set()
    for path in vu_paths:
        for i,ele in enumerate(path):
            if i%2==1:
                vu_path_nodes.add(ele)

    return vu_path_nodes


def get_pairs_that_pass_through(G, node, type_triple, step_lim, pairs_lim):

    v_type,rel,u_type = G.nodes[type_triple[0]]['nlabel'],type_triple[1],G.nodes[type_triple[2]]['nlabel']
    all_paths = prc.find_single_paths(G, node, step_lim - 1)
    paths_per_dist = {v_type: {}, u_type: {}}
    for path in all_paths:
        end = path[-1]
        end_type = G.nodes[end]['nlabel']
        end_dist = (len(path) - 1) / 2
        if end_type in paths_per_dist:
            if end_dist not in paths_per_dist[end_type]:
                paths_per_dist[end_type][end_dist] = set()
            paths_per_dist[end_type][end_dist].add(tuple([node for i, node in enumerate(path) if i%2==0]))

    path_triples = []
    for path_length in range(2, step_lim+1):
        for i in range(1, path_length):
            paths_v = paths_per_dist[v_type]
            if i in paths_v:
                paths_vi = paths_v[i]
            else:
                continue
            paths_u = paths_per_dist[u_type]
            if path_length - i in paths_u:
                paths_umi = paths_u[path_length - i]
            else:
                continue
            if len(paths_vi) * len(paths_umi) <= pairs_lim:
                sample_v = paths_vi
                sample_u = paths_umi
            else:
                sample_v = set(rnd.sample(list(paths_vi),min(len(paths_vi),int(pairs_lim ** 0.5))))
                sample_u = set(rnd.sample(list(paths_umi),min(len(paths_umi),int(pairs_lim/len(sample_v)))))

            path_triples = []
            for v_part in sample_v:
                for u_part in sample_u:
                    path_nodes = v_part + u_part
                    if len(path_nodes) - len(set(path_nodes))==1:
                        path_triples.append((v_part[-1], rel, u_part[-1]))

    return set(path_triples)


def filter_circles(pairs,G,T=None,step_lim=4,rev_rel=-1):
    good_pairs = set()
    for count, pair in enumerate(pairs):
        print('filter circles: ' + str(count) + '/' + str(len(pairs)))
        if not T:
            paths = prc.find_paths_between(G, pair[0], pair[2], step_lim)
        else:
            paths = T[pair[0]][pair[2]]
        if len(paths) > 0:
            good_pairs.add(pair)
    return good_pairs


def get_pairs_with_crossing_paths(G, T, triple, step_lim, pairs_lim=1000, rev_rel=1):

    vu_paths_nodes = get_path_nodes(triple,T)

    path_triples = {}
    for node in vu_paths_nodes:
        path_triples[node] = get_pairs_that_pass_through(node,G,triple,step_lim,pairs_lim)

    T_path_triples = {}
    for count1,path_node in enumerate(path_triples):
        print('cross nodes: ' + str(count1) + '/' + str(len(path_triples)))
        for count2,path_triple in enumerate(path_triples[path_node]):
            print('triples of node: ' + str(count2) + '/' + str(len(path_triples[path_node])))
            T_path_triples = prc.add_paths(G, path_triples[path_node], step_lim, rev_rel, T_path_triples)
            path_triples[path_node] = filter_circles(path_triples[path_node],G,T_path_triples,step_lim,rev_rel)

    return path_triples, T_path_triples


def make_node_train_set(G, T, val_set, test_set, step_lim=4, pairs_lim=1000, rev_rel=None):

    sample_triple_for_types = list(test_set)[0]
    if not rev_rel:
        rev_rel = sample_triple_for_types[1]

    node_lot = set()
    for i, triple in enumerate(test_set):
        print('gathering test node lot: ' + str(i) + '/' + str(len(test_set)))
        node_lot = node_lot.union(get_path_nodes(triple,T))

    for i, triple in enumerate(val_set):
        print('gathering val node lot: ' + str(i) + '/' + str(len(val_set)))
        node_lot = node_lot.union(get_path_nodes(triple,T))

    pairs = set()
    for i, node in enumerate(node_lot):
        print('getting pairs for node: ' + str(i) + '/' + str(len(node_lot)))
        pairs = pairs.union(get_pairs_that_pass_through(G,node,sample_triple_for_types,step_lim,pairs_lim))

    #T = preproc.add_paths(G, pairs, step_lim, rev_rel, T)
    pairs = filter_circles(pairs, G, None, step_lim, rev_rel)
    pairs = filter_val_test(pairs,val_set,test_set)
    train_set_v, labs = make_labs(pairs,G)

    return train_set_v, labs, T


def filter_val_test(pairs,val_set,test_set):
    return [pair for pair in pairs if pair not in val_set and pair not in test_set]


def make_labs(pairs,G):
    labs = np.zeros(len(pairs))
    for i,pair in enumerate(pairs):
        labs[i] = 1 if pair[2] in G[pair[0]] else 0
    return pairs,labs


def randomize_train_set(train_set):
    keys = list(train_set.keys())
    values = list(train_set.values())
    random_indices = np.random.choice(list(range(len(keys))),replace=False)
    train_set_rand = {}
    for i,key in enumerate(keys):
        train_set_rand[key] = values[random_indices[i]]
    return train_set


def count_labels(path_triples,train_set,val_set,test_set,G):
    labels = {}
    for path_node in path_triples:
        labels[path_node] = {}
        labels[path_node]['ones'] = 0
        labels[path_node]['total'] = 0

        for path_triple in path_triples[path_node]:
            if path_triple in train_set or path_triple in val_set or path_triple in test_set:
                labels[path_node]['ones'] += 1
            labels[path_node]['total'] += 1
    for n in labels:
        print(str(G.nodes[n]['type']) + ':'+str(labels[n]['ones']) + '/' + str(labels[n]['total']))
    return labels


def get_test_metapath_nums(test_set,T):
    mp_lens = []
    for triple in test_set:
        mp_lens.append(len(T[triple[0]][triple[2]]))
    return np.array(mp_lens)


def make_node_comparissons(G,T,train_set,val_set):

    sample_triple_for_types = list(val_set)[0]
    node_groups = {}
    node_pos_rate = {}
    triple_types = {}
    for i, triple in enumerate(val_set):

        print('calculating group: ' + str(i) + '/' + str(len(val_set)))
        node_groups[triple] = {'pos': [], 'neg': {}}
        triple_types[triple] = (G.nodes[triple[0]]['type'],G.nodes[triple[2]]['type'])
        node_lot = get_path_nodes(triple, T)
        for j, node in enumerate(node_lot):
            print('pos, node: ' + str(j) + '/' + str(len(node_lot)))
            node_groups[triple]['pos'].append(node)
            node_pos_rate[node] = get_node_pos_rate(node,G,triple,4,200)

        for j, triple_nz in enumerate(val_set[triple]['near_zeros']):
            node_groups[triple]['neg'][triple_nz] = []
            triple_types[triple_nz] = (G.nodes[triple_nz[0]]['type'], G.nodes[triple_nz[2]]['type'])
            if triple_types[triple_nz] == triple_types[triple]:
                node_lot = get_path_nodes(triple_nz, T)
                for k, node_nz in enumerate(node_lot):
                    print('triple_nz: '+ str(j) + '/' + str(len(val_set[triple]['near_zeros'])) + ', node: ' + str(k) + '/' + str(len(node_lot)))
                    node_groups[triple]['neg'][triple_nz].append(node_nz)
                    node_pos_rate[node_nz] = get_node_pos_rate(node_nz, G, triple, 4, 200)

        for j, triple_fz in enumerate(val_set[triple]['far_zeros']):
            node_groups[triple]['neg'][triple_fz] = []
            triple_types[triple_fz] = (G.nodes[triple_fz[0]]['type'], G.nodes[triple_fz[2]]['type'])
            if triple_types[triple_fz] == triple_types[triple]:
                node_lot = get_path_nodes(triple_fz, T)
                for k, node_fz in enumerate(node_lot):
                    print('triple_fz: ' + str(j) + '/' + str(len(val_set[triple]['far_zeros'])) + ', node: ' + str(k) + '/' + str(len(node_lot)))
                    node_groups[triple]['neg'][triple_fz].append(node_nz)
                    node_pos_rate[node_fz] = get_node_pos_rate(node_fz, G, triple, 4, 200)

        if i == 10:
            break

    return node_groups, node_pos_rate, triple_types


def get_node_pos_rate(G,node,pair_types,steps,samples):
    pairs = get_pairs_that_pass_through(G, node, pair_types, steps, samples)
    ones = 0
    for pair in pairs:
        if pair[0] in G and pair[2] in G[pair[0]]:
            ones += 1
    return ones,len(pairs)


def copy_node(G,node,Gn):
    Gn.add_node(node)
    attr_filt = set(['nlabel','pos','neg'])
    for attr in G.nodes[node]:
        if attr in attr_filt:
            Gn.nodes[node][attr] = G.nodes[node][attr]


def copy_edges(G,n1,n2,Gn):
    edges = G[n1][n2]
    for edge in edges:
        Gn.add_edge(n1,n2,edge)
        attrs = G[n1][n2][edge]
        for attr in attrs:
            Gn[n1][n2][edge][attr] = G[n1][n2][edge][attr]


def plot_neighborhood(G,node,steps,all_names=False):
    Gn = nx.MultiDiGraph()
    copy_node(G,node,Gn)
    all_Gn_nodes = set([node] + [path[-1] for path in prc.find_single_paths(G, node, steps)])
    for n in all_Gn_nodes:
        copy_node(G,n,Gn)
    for n1 in Gn:
        for n2 in G[n1]:
            if n2 in Gn:
                copy_edges(G,n1,n2,Gn)
    pos = nx.spring_layout(Gn)
    nx.draw(Gn,pos)
    neighs = Gn[node]
    node_labels = nx.get_node_attributes(Gn, 'nlabel')
    nx.draw_networkx_labels(Gn, pos, labels=node_labels)
    #if all_names:
    #    node_labels = nx.get_node_attributes(Gn, 'name')
    #else:
    #    node_labels = {node: G.nodes[node]['name']}
    #nx.draw_networkx_labels(Gn, pos, labels=node_labels)
    #edge_labels = nx.get_edge_attributes(Gn, 'type')
    #nx.draw_networkx_edge_labels(Gn, pos, labels=edge_labels)
    return Gn


def plot_cis(cis,types,rates,nodes,type=None,thresh = 0.5):
    if type:
        ind = (types==type) * (rates[:,1]>0)
    else:
        ind = rates[:, 1] > 0
    cis_ind = cis[ind]
    nodes_ind = nodes[ind]
    rates_ind = rates[ind]
    fig, ax = plt.subplots()
    ax.scatter(cis_ind[:, 0], cis_ind[:, 1])
    cis_sel = cis_ind[cis_ind[:,0]>thresh]
    nodes_sel = nodes_ind[cis_ind[:,0]>thresh]
    rates_sel = rates_ind[cis_ind[:,0]>thresh]
    #ord = np.argsort(rates_sel[:,0])
    ord = np.argsort(cis_sel[:, 0])
    return cis_sel[ord],nodes_sel[ord],rates_sel[ord]


def get_sources(G,num=float('inf')):
    sources = {}
    for i, n1 in enumerate(G.nodes):
        print(str(i) + '/' + str(len(G.nodes)))
        #print(str(len(sources)) + '/' + str(num))
        if len(sources) > num:
            break
        for n2 in G[n1]:
            for edge in G[n1][n2]:
                if G[n1][n2][edge]['elabel'] == 'ART_SRC':
                    if n2 not in sources:
                        sources[n2] = 0
                    sources[n2] += 1

    return sources


def get_subgraph(G,srcs=['S_forbes.com', 'S_bbc.com', 'S_nytimes.com', 'S_yahoo.com', 'S_msn.com']):
    Gn = nx.MultiDiGraph()
    for i, n1 in enumerate(G):
        print(str(i) + '/' + str(len(G.nodes)))
        for n2 in G[n1]:
            for edge in G[n1][n2]:
                if G[n1][n2][edge]['elabel'] == 'ART_SRC' and n2 in srcs:
                    copy_node(G,n1,Gn)
                    for neigh in G[n1]:
                        copy_node(G,neigh,Gn)
                        copy_edges(G,n1,neigh,Gn)
                        add_reverse_edges(Gn,n1,neigh)
    return Gn


def add_reverse_edges(Gn,n1,n2):
    edges = Gn[n1][n2]
    for edge in edges:
        attrs = Gn[n1][n2][edge]
        Gn.add_edge(n2, n1,edge)
        for attr in attrs:
            Gn[n2][n1][edge][attr] = Gn[n1][n2][edge][attr]


def evaluate_path_diff(G,arts,node_dict):
    art_sets = {}
    for j,art in enumerate(arts):
        print(str(j)+ '/' +str(len(arts)))
        paths = prc.find_single_paths(G, art, 2)
        art_sets[art] = []
        for path in paths:
            weights = 1
            for i,n in enumerate(path):
                if i%2==0 and i>0:
                    weights *= node_dict[n]
            art_sets[art].append(weights)
    return art_sets


def evaluate_arts(G,arts,nodes,cis,type_filter=None):
    art_sets = {}
    for j,art in enumerate(arts):
        print(str(j)+ '/' +str(len(arts)))
        art_sets[art] = []
        for neigh in G[art]:
            if not type_filter or G.nodes[neigh]['nlabel']==type_filter:
                art_sets[art].append(cis[nodes==neigh,:])
    return art_sets


def plot_group_cis(group_cis):
    plt.figure()
    for i, ci in enumerate(group_cis):
        #print(ci)
        plt.plot([i,i],[ci[0,0],ci[0,1]],'C0o-')


def make_G_int(G):

    G_int = nx.MultiDiGraph()

    node_id = {}
    id_node = {}
    nid = 0
    nlabel_id = {}
    id_nlabel = {}
    lid = 0
    for node in G:
        if node not in node_id:
            node_id[node] = nid
            id_node[nid] = node
            nid += 1
        if G.nodes[node]['nlabel'] not in nlabel_id:
            nlabel_id[G.nodes[node]['nlabel']] = lid
            id_nlabel[lid] = G.nodes[node]['nlabel']
            lid += 1
        G_int.add_node(node_id[node], nlabel = nlabel_id[G.nodes[node]['nlabel']])
        if nlabel_id[G.nodes[node]['nlabel']] == 0:
            G_int.nodes[node_id[node]]['pos'] = G.nodes[node]['pos']
            G_int.nodes[node_id[node]]['neg'] = G.nodes[node]['neg']

    rel_id = {}
    id_rel = {}
    rid = 0
    for node in G:
        for neigh in G[node]:
            edges = G[node][neigh]
            for edge in edges:
                if G[node][neigh][edge]['elabel'] not in rel_id:
                    rel_id[G[node][neigh][edge]['elabel']] = rid
                    id_rel[rid] = G[node][neigh][edge]['elabel']
                    rid  += 1
                G_int.add_edge(node_id[node],node_id[neigh],edge, elabel=rel_id[G[node][neigh][edge]['elabel']])

    return G_int, node_id, id_node, rel_id, id_rel, nlabel_id, id_nlabel


def get_srcs(G):
    srcs = {}
    for edge in G.edges:
        if G.edges[edge]['elabel'] == 'ART_SRC':
            src = edge[1]
            if src not in srcs:
                srcs[src] = 0
            srcs[src] += 1
    return np.array(list(srcs.keys())), np.array(list(srcs.values()))


def assign_scores(G,fil):
    for i, node in enumerate(G):
        print(str(i) + '/' + str(len(G)))
        if G.nodes[node]['nlabel'] == 'ART':
            row = fil.loc[fil[0]==node]
            G.nodes[node]['scr1'] = row[2]
            G.nodes[node]['scr2'] = row[3]
    return G


def combine_graphs():
    print(0)
    with open('/home/pant/Desktop/PathM/datasets/gdelt_week/20190201_graph.gpickle','rb') as f:
        Gs = get_subgraph(pickle.load(f))
    print(1)
    with open('/home/pant/Desktop/PathM/datasets/gdelt_week/20190202_graph.gpickle', 'rb') as f:
        Gs = nx.compose(Gs,get_subgraph(pickle.load(f)))
    print(2)
    with open('/home/pant/Desktop/PathM/datasets/gdelt_week/20190203_graph.gpickle', 'rb') as f:
        Gs = nx.compose(Gs,get_subgraph(pickle.load(f)))
    print(3)
    with open('/home/pant/Desktop/PathM/datasets/gdelt_week/20190204_graph.gpickle', 'rb') as f:
        Gs = nx.compose(Gs, get_subgraph(pickle.load(f)))
    print(4)
    with open('/home/pant/Desktop/PathM/datasets/gdelt_week/20190205_graph.gpickle', 'rb') as f:
        Gs = nx.compose(Gs, get_subgraph(pickle.load(f)))
    print(5)
    with open('/home/pant/Desktop/PathM/datasets/gdelt_week/20190206_graph.gpickle', 'rb') as f:
        Gs = nx.compose(Gs, get_subgraph(pickle.load(f)))
    print(6)
    with open('/home/pant/Desktop/PathM/datasets/gdelt_week/20190207_graph.gpickle', 'rb') as f:
        Gs = nx.compose(Gs, get_subgraph(pickle.load(f)))
    print(7)

    res = make_G_int(Gs)
    with open('big_graph','wb') as f:
        pickle.dump(res,f)

    return res


def proportion_confint(success, total, confint=0.95):
    quantile = (1 - confint) / 2.
    if success > 0:
        lower = beta.ppf(quantile, success, total - success + 1)
    else:
        lower = 0
    if success < total:
        upper = beta.ppf(1 - quantile, success + 1, total - success)
    else:
        upper = 1
    return lower, upper


def calc_bind_cis(G, nodes, steps, pairs_per_node, pair_types=(0, 1, 2)):
    rates = []
    for i, node in enumerate(nodes):
        print('node: ' + str(i) + '/' + str(len(nodes)))
        rates.append(get_node_pos_rate(G, node, pair_types, steps, pairs_per_node))
    types = [G.nodes[node]['nlabel'] for node in nodes]
    conf_ints = [proportion_confint(rate[0], rate[1]) for rate in rates]
    print('returning')
    return np.array(nodes), np.array(rates), np.array(conf_ints), np.array(types)


def calc_bind_cis_par(G, steps, pairs_per_node, pair_types=(0, 1, 2), workers=30):

    nodes = list(G)

    p = Pool(workers)

    data_splits = []
    slice = int(len(nodes) / workers) + 1
    for i in range(workers):
        node_split = nodes[i * slice:min((i + 1) * slice, len(nodes))]
        data_splits.append((G, node_split, steps, pairs_per_node, pair_types))

    res = p.starmap(calc_bind_cis, data_splits)
    p.close()

    nodes = np.empty((0,))
    rates = np.empty((0,2))
    conf_ints = np.empty((0,2))
    types = np.empty((0,))
    print('aggregating bind cis')
    for r in res:
        nodes = np.hstack((nodes,r[0]))
        rates = np.vstack((rates, r[1]))
        conf_ints = np.vstack((conf_ints, r[2]))
        types = np.hstack((types, r[3]))

    print('writing out')
    with open('dblp_preprocs/dblp_preproc_{}{}{}'.format(pair_types[0],pair_types[1],pair_types[2]), 'wb') as f:
        pickle.dump([G,nodes,rates,conf_ints,types,None],f)

    return np.array(nodes), np.array(rates), np.array(conf_ints), np.array(types)


def make_gdelt_cis_sing():
    with open('big_graph', 'rb') as f:
        G, node_id, id_node, rel_id, id_rel, nlabel_id, id_nlabel = pickle.load(f)
    calc_bind_cis_par(G, 3, 1000, pair_types=(0, 1, 2), workers=40)
    generate_selected_data_structs(workers=40)


import scipy.sparse as sp
import scipy.io


def make_mat(path):
    with open(path, 'rb') as f:
        G, T, train_triples, train_labels, val_triples, val_labels, test_triples, test_labels, cis_dict, ci_feats = pickle.load(f)
    del T
    net = sp.lil_matrix((len(G),len(G)))
    for node in G:
        for neigh in G[node]:
            net[node,neigh] = 1
    node_types = set()
    for node in G.nodes:
        node_types.add(G.nodes[node]['nlabel'])
    group = np.zeros((len(G),4 + len(node_types)))
    for node in G.nodes:
        group[node, 0] = ci_feats[node][0]
        group[node, 1] = ci_feats[node][1]
        group[node, 2] = G.nodes[node]['pos'] if 'pos' in G.nodes[node] else 0
        group[node, 3] = G.nodes[node]['neg'] if 'neg' in G.nodes[node] else 0
        group[node, 4 + G.nodes[node]['nlabel']] = 1
    mat = {'net': net, 'group': group}
    scipy.io.savemat(path+'.mat', mat)


def find_dist_neighs(G, node, dist, paths_lim=float('inf')):
    dist_neighs = []
    to_extend = [[node]]

    while to_extend and len(dist_neighs)<paths_lim:

        cur_path = to_extend.pop(0)
        if len(cur_path) == 2*dist+1:
            dist_neighs.append(cur_path[-1])

        if len(cur_path) < 2*dist+1:
            for neigh in G[cur_path[-1]]:
                if neigh not in cur_path:
                    for rel_id in G[cur_path[-1]][neigh]:
                        ext_path = list(cur_path)
                        ext_path.append(rel_id)
                        ext_path.append(neigh)
                        to_extend.append(ext_path)

    return dist_neighs


def random_walk_paths(G, start, metapath, step_lim, sample_size, try_fac=10):

    if G.nodes[start]['nlabel'] != metapath[0]:
        return None

    paths = list()
    try_count = 0
    try_lim = try_fac * sample_size -1

    while len(paths) < sample_size and try_count < try_lim:

        step_count = 0
        path = [start]

        while step_count < step_lim:

            next_type = metapath[step_count%(len(metapath)-1)+1]
            type_neighs = [neigh for neigh in G[path[-1]] if G.nodes[neigh]['nlabel'] == next_type]

            if len(type_neighs) == 0 :
                paths.append(path)
                break

            else:
                nxt = rnd.sample(type_neighs,1)[0]
                path.append(nxt)
                step_count += 1

                if len(path)==step_lim+1: #rnd.uniform(0,1) < 1/(step_lim - step_count + 1):
                    paths.append(path)
                    break

        try_count += 1

    return paths


def rw_loop(G,nodes,metapaths,steps,samples):
    rws = []
    for i, node in enumerate(nodes):
        print('{}/{} ({})'.format(i, len(nodes), node))
        for metapath in metapaths:
            rws.append(random_walk_paths(G, node, metapath, steps, samples))
    return rws


def remove_edges(G, test_triples, test_labels):
    for (org, rel, art),label in zip(test_triples,test_labels):
        if label == 1:
            G.remove_edge(org, art)
            G.remove_edge(art, org)
    return G


def neg_sample_loop(G, heads, tails, samp_size, rel_id):
    triples = []
    labels = []
    for i, head in enumerate(heads):
        print('{}/{}'.format(i, len(heads)))
        pos = set(G[head])
        near = set(nx.ego_graph(G, head, 3).nodes) - set(nx.ego_graph(G, head, 2).nodes) - pos
        near_samp = rnd.sample(near, min(len(near), samp_size))
        far = tails - pos - near
        far_samp = rnd.sample(far, min(len(far), samp_size))
        for tail in pos:
            triples.append([head, rel_id, tail])
            labels.append(1)
        for tail in near_samp:
            triples.append([head, rel_id, tail])
            labels.append(0)
        for tail in far_samp:
            triples.append([head, rel_id, tail])
            labels.append(0)
    return triples, labels


def make_triples_neg_samp(G, heads, tails, workers, samp_size, rel_id):

    p = Pool(workers)
    data_splits = []
    slice = int(len(heads) / workers) + 1
    for i in range(workers):
        head_split = heads[i * slice:min((i + 1) * slice, len(heads))]
        data_splits.append((G, set(head_split),  set(tails), samp_size, rel_id))

    res = p.starmap(neg_sample_loop, data_splits)
    p.close()

    triples = [trip for part in res for trip in part[0]]
    labels = [lab for part in res for lab in part[1]]

    return triples, labels



    with open('test_set','wb') as f:
        pickle.dump(test_set,f)


def m2v_preds(wv,test_set):
    scores = []
    for org,_,art in test_set:
        sc = wv[str(org)].dot(wv[str(art)])
        scores.append(sc)
    return scores


from gensim.models import Word2Vec


def generate_data_structs(graph_path = 'big_graph', ci_preproc_path='gdelt_samp_preproc', sel_org ='O_european union', rel=1, workers = 50):
    with open(graph_path, 'rb') as f:
        G, node_id, id_node, rel_id, id_rel, nlabel_id, id_nlabel = pickle.load(f)
    pair_types = [(0,0,1),(0,1,2),(0,3,4),(0,4,5)] #src,org,loc,per
    for pt in pair_types:
        calc_bind_cis_par(G, 3, 1000, pair_types=pt, workers=20)


import random


def make_dataset_gdelt():

    with open('gdelt_graph_sp', 'rb') as f:
        G, node_id, id_node, rel_id, id_rel, nlabel_id, id_nlabel = pickle.load(f)

    with open('gdelt_preprocs/gdelt_preproc_034', 'rb') as f:
        G, nodes, rates, cis, types, art_sets = pickle.load(f)

    pers = np.array([node for node in G.nodes if G.nodes[node]['nlabel'] == nlabel_id['PER']])
    len(pers)
    thms = np.array([node for node in G.nodes if G.nodes[node]['nlabel'] == nlabel_id['THM']])
    len(pers)
    locs = np.array([node for node in G.nodes if G.nodes[node]['nlabel'] == nlabel_id['LOC']])
    len(pers)
    arts = np.array([node for node in G.nodes if G.nodes[node]['nlabel'] == nlabel_id['ART']])
    len(arts)
    orgs = np.array([node for node in G.nodes if G.nodes[node]['nlabel'] == nlabel_id['ORG']])
    len(orgs)

    cis_dict = {node: cis[i] for i, node in enumerate(nodes)}
    ci_feats = {node: (cis[i][0], cis[i][1]) for i, node in enumerate(nodes)}

    links = np.array([len(G[loc]) for loc in locs])
    selected = locs[links>100]
    rel = 3

    workers = 30
    triples,labels = make_triples_neg_samp(G, selected, arts, workers, 500, rel)

    triples, labels = list(zip(*random.sample(list(zip(triples, labels)), len(triples))))

    print(triples[1:10])
    print(labels[1:100])

    T = prc.find_paths_par(G, triples, 3, None, {}, workers)

    train_triples = np.array(triples[0:int(0.7 * len(triples))])
    train_labels = np.array(labels[0:int(0.7 * len(triples))])
    val_triples = np.array(triples[int(0.7 * len(triples)):int(0.8 * len(triples))])
    val_labels = np.array(labels[int(0.7 * len(triples)):int(0.8 * len(triples))])
    test_triples = np.array(triples[int(0.8 * len(triples)):len(triples)])
    test_labels = np.array(labels[int(0.8 * len(triples)):len(triples)])

    print('writing out')
    with open('gdelt_preprocs/gdelt_full_loc', 'wb') as f:
        pickle.dump(
            [G, T, train_triples, train_labels, val_triples, val_labels, test_triples, test_labels, cis_dict, ci_feats],
            f)
    with open('gdelt_preprocs/gdelt_full_loc_triples', 'wb') as f:
        pickle.dump(
            [G, train_triples, train_labels, val_triples, val_labels, test_triples, test_labels, cis_dict, ci_feats],
            f)


def make_cis_dblp():
    G, ids, names = prc.make_dblp_graph()
    pair_types = [(0,0,2),(0,1,1)]
    for pt in pair_types:
        calc_bind_cis_par(G, 3, 1000, pair_types=pt, workers=20)


def make_dblp_dataset():

    G, ids, names = prc.make_dblp_graph()

    with open('dblp_preprocs/dblp_preproc_011', 'rb') as f:
        G, nodes, rates, cis, types, art_sets = pickle.load(f)

    auths = np.array([node for node in G.nodes if G.nodes[node]['nlabel'] == 2])
    len(auths)
    papers = np.array([node for node in G.nodes if G.nodes[node]['nlabel'] == 0])
    len(papers)
    venues = np.array([node for node in G.nodes if G.nodes[node]['nlabel'] == 1])
    len(venues)

    cis_dict = {node: cis[i] for i, node in enumerate(nodes)}
    ci_feats = {node: (cis[i][0], cis[i][1]) for i, node in enumerate(nodes)}

    links = np.array([len(G[auth]) for auth in auths])
    selected = venues[:]
    rel = 1
    print(type(selected))
    print(type(papers))

    selected = np.array(random.sample(list(selected),len(selected)))
    papers = np.array(random.sample(list(papers),len(papers)))

    workers = 20
    triples,labels = make_triples_neg_samp(G, selected, papers, workers, 500, rel)

    triples, labels = list(zip(*random.sample(list(zip(triples, labels)), len(triples))))

    print(triples[1:10])
    print(labels[1:100])

    T = prc.find_paths_par(G, triples, 3, None, {}, workers)

    train_triples = np.array(triples[0:int(0.7 * len(triples))])
    train_labels = np.array(labels[0:int(0.7 * len(triples))])
    val_triples = np.array(triples[int(0.7 * len(triples)):int(0.8 * len(triples))])
    val_labels = np.array(labels[int(0.7 * len(triples)):int(0.8 * len(triples))])
    test_triples = np.array(triples[int(0.8 * len(triples)):len(triples)])
    test_labels = np.array(labels[int(0.8 * len(triples)):len(triples)])

    print('writing out')
    with open('dblp_preprocs/ven', 'wb') as f:
        pickle.dump(
            [G, T, train_triples, train_labels, val_triples, val_labels, test_triples, test_labels, cis_dict, ci_feats],
            f)
    with open('dblp_preprocs/ven_triples', 'wb') as f:
        pickle.dump(
            [G, train_triples, train_labels, val_triples, val_labels, test_triples, test_labels, cis_dict, ci_feats],
            f)


def get_all_rws(G, test_triples, test_labels , steps=5, samples=100, metapaths = [[0,1,0],[0,2,0], [0,4,0], [0,5,0]]):
    G = remove_edges(G, test_triples, test_labels)
    workers = 30
    nodes = list(G.nodes)
    p = Pool(workers)
    data_splits = []
    slice = int(len(nodes) / workers) + 1
    for i in range(workers):
        node_split = nodes[i * slice:min((i + 1) * slice, len(nodes))]
        data_splits.append((G,node_split,metapaths,steps,samples))
    res = p.starmap(rw_loop, data_splits)
    p.close()
    rws = [rw for rw in res ]
    walks = []
    for res in rws:
        for meta in res:
            if meta != None:
                for walk in meta:
                    walks.append(walk)
    for walk in walks:
        for i, w in enumerate(walk):
            walk[i] = str(w)
    with open('walks','wb') as f:
        pickle.dump(walks,f)
    return walks


from gensim.models import Word2Vec


def train_m2v():
    with open('gdelt_preprocs/org', 'rb') as f:
        G, T, train_triples, train_labels, val_triples, val_labels, test_triples, test_labels, cis_dict, ci_feats = pickle.load(f)
    rws = get_all_rws(G, test_triples, test_labels)
    model = Word2Vec(rws, size=64, window=3, min_count=0, sg=1, workers=30, iter=100, negative=10, hs=1, alpha=0.1, min_alpha=0.001)
    with open('m2v_org','wb') as f:
        pickle.dump(model.wv,f)

    with open('gdelt_preprocs/per', 'rb') as f:
        G, T, train_triples, train_labels, val_triples, val_labels, test_triples, test_labels, cis_dict, ci_feats = pickle.load(f)
    rws = get_all_rws(G, test_triples, test_labels)
    model = Word2Vec(rws, size=64, window=3, min_count=0, sg=1, workers=30, iter=100, negative=10, hs=1, alpha=0.1, min_alpha=0.001)
    with open('m2v_per','wb') as f:
        pickle.dump(model.wv,f)

    with open('gdelt_preprocs/loc', 'rb') as f:
        G, T, train_triples, train_labels, val_triples, val_labels, test_triples, test_labels, cis_dict, ci_feats = pickle.load(f)
    rws = get_all_rws(G, test_triples, test_labels)
    model = Word2Vec(rws, size=64, window=3, min_count=0, sg=1, workers=30, iter=100, negative=10, hs=1, alpha=0.1, min_alpha=0.001)
    with open('m2v_loc','wb') as f:
        pickle.dump(model.wv,f)


def make_herec_data():

    with open('../../gdelt_graph_sp', 'rb') as f:
        G, node_id, id_node, rel_id, id_rel, nlabel_id, id_nlabel = pickle.load(f)

    with open('../../gdelt_preprocs/per', 'rb') as f:
        G, T, train_triples, train_labels, val_triples, val_labels, test_triples, test_labels, cis_dict, ci_feats = pickle.load(
            f)

    orgs = [node for node in G.nodes if G.nodes[node]['nlabel'] == nlabel_id['ORG']]
    pers = [node for node in G.nodes if G.nodes[node]['nlabel'] == nlabel_id['PER']]
    srcs = [node for node in G.nodes if G.nodes[node]['nlabel'] == nlabel_id['SRC']]
    locs = [node for node in G.nodes if G.nodes[node]['nlabel'] == nlabel_id['LOC']]
    arts = [node for node in G.nodes if G.nodes[node]['nlabel'] == nlabel_id['ART']]

    sel = pers

    ent_map = {node: i+1 for i, node in enumerate(sel)}
    art_map = {node: i+1 for i, node in enumerate(arts)}

    print('sel:{}, arts:{}'.format(len(sel),len(arts)))

    train = []
    for trip,label in zip(train_triples,train_labels):
        train.append([ent_map[trip[0]],art_map[trip[2]],label])
    for trip, label in zip(val_triples, val_labels):
        train.append([ent_map[trip[0]], art_map[trip[2]], label])

    test = []
    test_ones = set()
    for trip, label in zip(test_triples, test_labels):
        test.append([ent_map[trip[0]], art_map[trip[2]], label])
        if label == 1:
            test_ones.add((ent_map[trip[0]]-1,art_map[trip[2]]-1))

    print('writing train')
    with open('../../herec_data/herec_train.txt','wt') as f:
        for pair in train:
            f.write('{}\t{}\t{}\n'.format(pair[0],pair[1],pair[2]))

    print('writing test')
    with open('../../herec_data/herec_test.txt','wt') as f:
        for pair in test:
            f.write('{}\t{}\t{}\n'.format(pair[0],pair[1],pair[2]))

    print('making apa')
    ap = np.zeros((len(arts), len(pers)))
    for i,art in enumerate(arts):
        for j,per in enumerate(pers):
            if per in G[art]:
                if (per, art) not in test_ones:
                    ap[i, j] = 1

    apa = ap.dot(ap.T)
    with open('../../herec_data/apa.txt', 'wt') as f:
        for i in range(apa.shape[0]):
            for j in range(apa.shape[1]):
                f.write('{}\t{}\t{}\n'.format(i + 1, j + 1, apa[i, j]))

    pap = ap.T.dot(ap)
    with open('../../herec_data/pap.txt','wt') as f:
        for i in range(pap.shape[0]):
            for j in range(pap.shape[1]):
                f.write('{}\t{}\t{}\n'.format(i+1,j+1, pap[i,j]))

    print('making aoa')
    ao = np.zeros((len(arts), len(orgs)))
    for i, art in enumerate(arts):
        for j, org in enumerate(orgs):
            if org in G[art]:
                ao[i, j] = 1

    aoa = ao.dot(ao.T)
    with open('../../herec_data/aoa.txt','wt') as f:
        for i in range(aoa.shape[0]):
            for j in range(aoa.shape[1]):
                f.write('{}\t{}\t{}\n'.format(i+1,j+1,aoa[i,j]))

    oao = ao.T.dot(ao)
    with open('../../herec_data/oao.txt', 'wt') as f:
        for i in range(oao.shape[0]):
            for j in range(oao.shape[1]):
                f.write('{}\t{}\t{}\n'.format(i + 1, j + 1, oao[i, j]))


    print('making asa')
    as_ = np.zeros((len(arts), len(srcs)))
    for i, art in enumerate(arts):
        for j, sr in enumerate(srcs):
            if sr in G[art]:
                as_[i, j] = 1
    asa = as_.dot(as_.T)
    with open('../../herec_data/asa.txt','wt') as f:
        for i in range(asa.shape[0]):
            for j in range(asa.shape[1]):
                f.write('{}\t{}\t{}\n'.format(i+1,j+1,asa[i,j]))

    print('making ala')
    al = np.zeros((len(arts), len(locs)))
    for i, art in enumerate(arts):
        for j, loc in enumerate(locs):
            if loc in G[art]:
                al[i, j] = 1

    ala = al.dot(al.T)
    with open('../../herec_data/ala.txt', 'wt') as f:
        for i in range(ala.shape[0]):
            for j in range(ala.shape[1]):
                f.write('{}\t{}\t{}\n'.format(i + 1, j + 1, ala[i, j]))

    lal = al.T.dot(al)
    with open('../../herec_data/lal.txt', 'wt') as f:
        for i in range(lal.shape[0]):
            for j in range(lal.shape[1]):
                f.write('{}\t{}\t{}\n'.format(i + 1, j + 1, lal[i, j]))


def check_valid(firstName, lastName):
    if firstName == '' or lastName == '' \
            or firstName == '#' or lastName == '#' \
            or '*' in firstName or '*' in lastName \
            or firstName == '-' or lastName == '-' \
            or ' ' in firstName or ' ' in lastName \
            or firstName == 'et.' or lastName == 'et.' \
            or firstName == 'et' or lastName == 'et' \
            or firstName == '.' or lastName == '.' \
            or firstName == ':.' or lastName == ':':
        return False
    else:
        return True


def clean_name(name):
    return name.strip(' ').lstrip('.').lstrip('-').strip('\t').strip('\"')


def get_names(auth_list):
    parsed_auths = []
    try:
        auth_list = auth_list.replace('&amp;apos;', ' ')
        auth_list = re.sub(r'\d+', '', auth_list)
        auth_list = re.sub(r',,', ',', auth_list)
        auth_list = auth_list.strip(',')
        auth_list = auth_list.strip(';')
        if '&amp' in auth_list or '(' in auth_list or ')' in auth_list:
            return parsed_auths
        if ';' in auth_list:
            auth_list = auth_list.replace('&', ';')
            auth_list = auth_list.replace(' and ', ';')
            auths = auth_list.split(';')
            for auth in auths:
                if ',' in auth:
                    names = auth.split(',')
                else:
                    names = auth.strip(' ').split(' ')
                firstName = clean_name(names[-1])
                lastName = clean_name(names[0])
                if lastName.endswith('.'):
                    buf = firstName
                    firstName = lastName
                    lastName = buf
                if check_valid(firstName, lastName):
                    parsed_auths.append((firstName, lastName))
        elif ',' in auth_list:
            auth_list = auth_list.replace('&', ',')
            auth_list = auth_list.replace(' and ', ',')
            auths = auth_list.split(',')
            if len(auths[0].strip(' ').split(' ')) > 1:
                for auth in auths:
                    names = auth.strip(' ').split(' ')
                    firstName = clean_name(names[0])
                    lastName = clean_name(names[-1])
                    if lastName.endswith('.'):
                        buf = firstName
                        firstName = lastName
                        lastName = buf
                    if check_valid(firstName, lastName):
                        parsed_auths.append((firstName, lastName))
        elif len(auth_list.strip(' ').split(' ')) == 2:
            names = auth_list.strip(' ').split(' ')
            firstName = clean_name(names[0])
            lastName = clean_name(names[-1])
            if lastName.endswith('.'):
                buf = firstName
                firstName = lastName
                lastName = buf
            if check_valid(firstName, lastName):
                parsed_auths.append((firstName, lastName))
        return parsed_auths
    except:
        traceback.print_exc()
        print(str(auth_list))
        return []


def add_projects(G, proj):
    status = pd.get_dummies(proj['status'])
    programme = pd.get_dummies(proj['programme'])
    topics = proj['topics']
    start = [pd.Timestamp(st).timestamp() if not pd.isna(st) else None for st in proj['startDate']]
    end = [pd.Timestamp(st).timestamp() if not pd.isna(st) else None for st in proj['endDate']]
    fundingScheme = pd.get_dummies(proj['fundingScheme'])
    totalCost = [float(c.replace(',', '.')) if not pd.isna(c) else None for c in proj['totalCost']]
    maxCont = [float(c.replace(',', '.')) if not pd.isna(c) else None for c in proj['ecMaxContribution']]
    for i, row in proj.iterrows():
        node_feats = []
        node_feats += list(status.iloc[i])
        node_feats += list(programme.iloc[i])
        node_feats.append(start[i])
        node_feats.append(end[i])
        node_feats += list(fundingScheme.iloc[i])
        node_feats.append(totalCost[i])
        node_feats.append(maxCont[i])
        G.add_node(int(row['id']), type='project', features=node_feats)
        if topics.iloc[i] not in G:
            G.add_node(topics.iloc[i],type='topic')
        G.add_edge(int(row['id']),topics.iloc[i],type='project-topic')
        G.add_edge(topics.iloc[i],int(row['id']),type='project-topic')
    else:
        return G


def add_organizations(G, org):
    country = org['country']
    country_feat = pd.get_dummies(org['country'])
    city = org['city']
    act_types = pd.get_dummies(org['activityType'])
    roles = pd.get_dummies(org['role'])
    for i, row in org.iterrows():
        if int(row['id']) not in G:
            node_feats = []
            node_feats += list(act_types.iloc[i])
            node_feats += list(country_feat.iloc[i])
            G.add_node(int(row['id']), type='organization', features=node_feats)
        node1 = int(row['projectID'])
        node2 = int(row['id'])
        edge_feats = []
        edge_feats += list(roles.iloc[i])
        edge_feats.append(int(row['endOfParticipation']))
        edge_feats.append(float(row['ecContribution'].replace(',', '.')) if not pd.isna(row['ecContribution']) else None)
        _ = G.add_edge(node1, node2, type='organization-project', features=edge_feats)
        _ = G.add_edge(node2, node1, type='organization-project', features=edge_feats)
        if country.iloc[i] not in G:
            G.add_node(country.iloc[i], type='country')
        if city.iloc[i] not in G:
            G.add_node(city.iloc[i], type='city')
        _ = G.add_edge(int(row['id']), country.iloc[i], type='organization-country')
        _ = G.add_edge(country.iloc[i], int(row['id']), type='organization-country')
        _ = G.add_edge(int(row['id']), city.iloc[i], type='organization-city')
        _ = G.add_edge(city.iloc[i], int(row['id']), type='organization-city')
    return G


def add_fels(G, fel):
    title = pd.get_dummies(fel['title'])
    fund = pd.get_dummies(fel['fundingScheme'])
    miss = 0
    miss1 = 0
    miss2 = 0
    for i, row in fel.iterrows():
        try:
            key = row['firstName'] + ' ' + row['lastName']
            if key not in G:
                node_feats = []
                node_feats += list(title.iloc[i])
                node_feats.append(int(row['sheet']))
                G.add_node( key, type='person', features=node_feats)
            node2 = key
            try:
                node1 = int(row['projectId'])
                edge_feats = list(fund.iloc[i])
                if node1 in G:
                    _ = G.add_edge(node1, node2, type='person-project', features=edge_feats)
                    _ = G.add_edge(node2, node1, type='person-project', features=edge_feats)
            except:
                miss1 += 1
            else:
                try:
                    node1 = int(row['organizationId'])
                    edge_feats = []
                    if node1 in G:
                        _ = G.add_edge(node1, node2, type='person-organization', features=edge_feats)
                        _ = G.add_edge(node2, node1, type='person-organization', features=edge_feats)
                except:
                    miss2 += 1
        except:
            miss += 1
    else:
        print('missed: {},{},{} of {}'.format(miss, miss1, miss2, len(fel)))
        return G


def add_publ(G, publ):
    miss = 0
    new = 0
    topics = publ['topics']
    pubAs = pd.get_dummies(publ['isPublishedAs'])
    legal = pd.get_dummies(publ['legalBasis'])
    last_update = [pd.Timestamp(st).timestamp() if not pd.isna(st) else None for st in publ['lastUpdateDate']]
    for i, row in publ.iterrows():
        try:
            ven_key = str(row['journalTitle'])
            if ven_key not in G:
                node_feats = []
                G.add_node(ven_key, type='journal', features=node_feats)
            pap_key = row['title']
            if pap_key not in G:
                node_feats = []
                node_feats += list(pubAs.iloc[i])
                node_feats += list(legal.iloc[i])
                node_feats.append(last_update[i])
                try:
                    node_feats.append(int(row['publishedYear']))
                except:
                    node_feats.append(None)
                G.add_node(pap_key, type='paper', features=node_feats)
            edge_feats = []
            _ = G.add_edge(ven_key, pap_key, type='paper-journal', features=edge_feats)
            _ = G.add_edge(pap_key, ven_key, type='paper-journal', features=edge_feats)
            proj_key = int(row['projectID'])
            if proj_key in G:
                edge_feats = []
                _ = G.add_edge(pap_key, proj_key, type='paper-project', features=edge_feats)
                _ = G.add_edge(proj_key, pap_key, type='paper-project', features=edge_feats)
            if topics.iloc[i] not in G:
                G.add_node(topics.iloc[i],type='topic')
            _ = G.add_edge(pap_key, topics.iloc[i], type='paper-topic')
            _ = G.add_edge(topics.iloc[i], pap_key, type='paper-topic')
            auth_list = get_names(row['authors'])
            for auth in auth_list:
                auth_key = auth[0] + ' ' + auth[1]
                if auth_key not in G:
                    node_feats = [None] * 6
                    G.add_node(auth_key, type='person', features=node_feats)
                    new += 1
                edge_feats = []
                _ = G.add_edge(auth_key, pap_key, type='paper-person', features=edge_feats)
                _ = G.add_edge(pap_key, auth_key, type='paper-person', features=edge_feats)
        except:
            traceback.print_exc()
            miss += 1
    else:
        print('missed: {}/{}'.format(miss, len(publ)))
        return G


def add_deli(G, deli):
    programme = pd.get_dummies(deli['programme'])
    topics = deli['topics']
    deli_type = pd.get_dummies(deli['deliverableType'])
    last_update = [pd.Timestamp(st).timestamp() if not pd.isna(st) else None for st in deli['lastUpdateDate']]
    miss = 0
    for i, row in deli.iterrows():
        try:
            deli_id = int(row['rcn'])
            if deli_id not in G:
                node_feats = []
                node_feats += list(programme.iloc[i])
                node_feats += list(deli_type.iloc[i])
                node_feats.append(last_update[i])
                G.add_node(deli_id, type='deliverable', features=node_feats)
            proj_id = int(row['projectID'])
            edge_feats = []
            if proj_id in G:
                _ = G.add_edge(proj_id, deli_id, type='project-deliverable', features=edge_feats)
                _ = G.add_edge(deli_id, proj_id, type='project-deliverable', features=edge_feats)
            if topics.iloc[i] not in G:
                G.add_node(topics.iloc[i],type='topic')
            G.add_edge(topics.iloc[i], deli_id, type='topic-deliverable')
            G.add_edge(deli_id, topics.iloc[i], type='topic-deliverable')
        except:
            miss += 1
    else:
        print('missed: {}/{}'.format(miss, len(deli)))
        return G


def add_rep(G, rep):
    language = pd.get_dummies(rep['language'])
    last_update = []
    for st in rep['lastUpdateDate']:
        try:
            last_update.append(pd.Timestamp(st).timestamp())
        except:
            last_update.append(None)
    else:
        programme = pd.get_dummies(rep['programme'])
        topics = rep['topics']
        miss = 0
        for i, row in rep.iterrows():
            try:
                rep_id = int(row['rcn'])
                if rep_id not in G:
                    node_feats = []
                    node_feats += list(programme.iloc[i])
                    node_feats += list(language.iloc[i])
                    node_feats.append(last_update[i])
                    G.add_node(rep_id, type='report', features=node_feats)
                proj_id = int(row['projectID'])
                edge_feats = []
                if proj_id in G:
                    _ = G.add_edge(proj_id, rep_id, type='project-report', features=edge_feats)
                    _ = G.add_edge(rep_id, proj_id, type='project-report', features=edge_feats)
                if topics.iloc[i] not in G:
                    G.add_node(topics.iloc[i], type='topic')
                G.add_edge(topics.iloc[i], rep_id, type='topic-report')
                G.add_edge(rep_id, topics.iloc[i], type='topic-report')
            except:
                miss += 1
        else:
            print('missed: {}/{}'.format(miss, len(rep)))
            return G


def make_cordis(root_dir='/mnt/data/pchronis/cordis/'):
    G = nx.MultiDiGraph()
    print('loading projects')
    proj = pd.read_csv((root_dir + 'cordis-h2020projects.csv'), sep=';')
    G = add_projects(G, proj)
    print('loading organizations')
    org = pd.read_csv((root_dir + 'cordis-h2020organizations.csv'), sep=';')
    G = add_organizations(G, org)
    print('loading fellows')
    fel0 = pd.read_excel((root_dir + 'cordis-h2020-msca-fellows.xlsx'), header=3, sheet_name=0)
    fel0['sheet'] = 0
    fel1 = pd.read_excel((root_dir + 'cordis-h2020-msca-fellows.xlsx'), header=2, sheet_name=1)
    fel1['sheet'] = 1
    pi = pd.read_excel((root_dir + 'cordis-h2020-erc-pi.xlsx'), header=3)
    pi['sheet'] = 2
    pi = pi.rename(columns={'organisationId':'organizationId',  'fundingScheme ':'fundingScheme'})
    fel = fel0.append(fel1).append(pi)
    G = add_fels(G, fel)
    print('loading publications')
    publ = pd.read_csv((root_dir + 'cordis-h2020projectPublications.csv'), sep=';', quotechar='"',
      skipinitialspace=True,
      escapechar='\\',
      error_bad_lines=False)
    G = add_publ(G, publ)
    print('loading deliverables')
    deli = pd.read_csv((root_dir + 'cordis-h2020projectDeliverables.csv'), sep=';')
    G = add_deli(G, deli)
    print('loading reports')
    rep = pd.read_csv((root_dir + 'cordis-h2020reports.csv'), sep=';')
    G = add_rep(G, rep)
    G.remove_node([n for n in G if pd.isna(n)][0])
    return G


def fix_graph(G, ext_node_feats):
    for node in G.nodes:
        feats = list(ext_node_feats[node])
        if 'year' in G.nodes[node]:
            feats.append(G.nodes[node]['year'])
        else:
            feats.append(0)
        G.nodes[node]['features'] = feats
        G.nodes[node]['type'] = G.nodes[node]['nlabel']
    for edge in G.edges:
        G.edges[edge]['features'] = [0, 0, 0]
        G.edges[edge]['type'] = G.edges[edge]['elabel']
    return G


def add_graph_feats(G, ext_node_feats):
    for node in G.nodes:
        feats = list(ext_node_feats[node])
        if 'year' in G.nodes[node]:
            feats.append(G.nodes[node]['year'])
        else:
            feats.append(0)
        G.nodes[node]['features'] = feats
    return G
