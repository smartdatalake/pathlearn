"""This module contains the implementation of PathLearn as well as functions to train and apply it."""

import torch
import numpy as np
import networkx as nx
import traceback
import time
import random
import pickle
from torch import nn
import pandas as pd


class PathL(torch.nn.Module):
    """
    This calss implements the PathLearn model using Pytorch.

    Attributes:

    G : networkx.MultiGraph
        The graph that is modelled. Each node and edge of the graph must have two attributes: 'type': int that contains
        the of the node/edge and 'features': List[float] that contains the features of the node/edge.
    T : Dict[ Any, Dict[ Any, List[ List[ Any ] ] ] ]
        A 2-level dictionary that contains all paths that exist between a pair of nodes. For node pair (u,v), T[u][v]
        contains a list with all paths from u to v that exist in G. Each path is a sequence of edge ids (as used in
        networkx.MultiGraph) and node ids. The path is also represented as a List. If PathL is applied to a node pair
        an entry for the pair must exist in T.
    edge_type : int
        The type of the edge that is modelled.
    n : int
        Number of nodes in G.
    m : int
        Number of edges in G.
    node_inds : Dict[Any, int]
        Map from node ids of G to integers [0,n-1], used for indexing.
    edge_inds : Dict[ (Any,Any,int), int]
        Map from edges of G to integers [n,n+m-1], used for indexing.
    node_type_inds : Dict[Any, int]
        Map from node types to integers [0,rho-1], used for indexing.
    edge_type_inds : Dict[ (Any,Any,int), int]
        Map from edges edge types to integers [rho,rho+ell-1], used for indexing.
    rho : int
        Number of node types in G.
    ell : int
        Number of edge types in G.
    max_feat_dim : int
        Number of features for the node/edge type with the largest number of features.
    max_path_steps : int
        Number of elements (nodes and edges) of the longest path of T.
    W1 : torch.nn.Parameter
        Tensor of size ell+rho+1 x max_feat_dim containing the weights of the features of each node/edge type.
    W2 : torch.nn.Parameter
        Tensor of size ell+rho+1 x 1 containing the constant weights each node/edge type.
    b : torch.nn.Parameter
        The bias of the model (scalar).
    Vf: torch.nn.Tensor
        Tensor of size n+m+1 x max_feat_dim containing the node/edge features. Vf[node_inds[v]] contains the feature
        vector of node v, Vf[edge_inds[e]] contains the feature vector of edge e. The vectors are padded with zeros to
        length max_feat_dim.
    Vc: torch.nn.Tensor
        Tensor of size n+m+1 x 1 containing the node/edge types. Vc[node_inds[v]] contains the type of node v,
        Vf[edge_inds[e]] contains the type of edge e.
    in_params: Dict[str, Union(torch.nn.Parameter, torch.nn.Tensor)]
        If given, used to initialise W1,W2,b,Vf,Vc.
    feat_transform_func: callable
        A function that performs any required feature engineering transformations on the features of the paths, when
        applying the model.
   """

    def __init__(self, G, T, feat_transform_func=None, in_params=None):

        super(PathL, self).__init__()

        self.G = G
        self.T = T
        self.feat_transform_func = feat_transform_func
        self.edge_type = self.get_edge_type(G, T)
        self.n = len(G)
        self.m = len(G.edges)
        self.node_inds = {node: i for i, node in enumerate(G)}
        self.edge_inds = {}
        for i, e in enumerate(G.edges):
            self.edge_inds[(e[0], e[1], e[2])] = self.n + i

        node_types = set(nx.get_node_attributes(G, 'type').values())
        self.rho = len(node_types)
        self.node_type_inds = {type:i for i, type in enumerate(node_types)}

        edge_types = set(nx.get_edge_attributes(G, 'type').values())
        self.ell = len(edge_types)
        self.edge_type_inds = {type: self.rho+i for i, type in enumerate(edge_types)}

        self.max_feat_dim = max([len(G.nodes[node]['features']) for node in G])
        self.max_path_steps = int( max([len(path) for u in T for v in T[u] for path in T[u][v]]) )

        if in_params is None:
            self.W1 = torch.nn.Parameter(torch.zeros(self.rho + self.ell + 1, self.max_feat_dim))
            self.W2 = torch.nn.Parameter(torch.zeros(self.rho + self.ell + 1, 1))
            self.b  = torch.nn.Parameter(torch.zeros(1))
            self.Vf = self.make_feats(G, self.n, self.m, self.max_feat_dim)
            self.Vc = self.make_cats(G, self.n, self.m, self.rho, self.ell)
        else:
            self.W1 = in_params['W1']
            self.W2 = in_params['W2']
            self.b  = in_params['b']
            self.Vf = in_params['Vf']
            self.Vc = in_params['Vc']

    def get_edge_type(self,G,T):
        """
        :param G: Input graph.
        :param T: Path dictionary T.
        :return:  The type of the modelled edge.
        """
        for u in T:
            for v in T[u]:
                try :
                    return G[u][v][0]['type']
                except:
                    pass

    def make_feats(self, G, n, m, feat_dim):
        """
        Creates feature tensor Vf.

        :param G: Input graph.
        :param n: Number of nodes.
        :param m: Number of edges.
        :param feat_dim: Number of features on the node/edge with the most features.
        :return: Tensor of size n+m+1 x max_feat_dim containing the node/edge features. Vf[node_inds[v]] contains the \
        feature vector of node v, Vf[edge_inds[e]] contains the feature vector of edge e. The vectors are padded with \
        zeros to length max_feat_dim.
        """

        Vf = torch.zeros(n + m + 1, feat_dim)
        for node in G:
            node_feats = G.nodes[node]['features']
            Vf[self.node_inds[node],0:len(node_feats)] = torch.Tensor(node_feats)
        for edge in G.edges:
            edge_feats = G.edges[edge]['features']
            Vf[self.edge_inds[edge],0:len(edge_feats)] = torch.Tensor(edge_feats)
        Vf[n+m] = torch.Tensor([0] * feat_dim)
        #for i in range(Vf.shape[1]):
        #    Vf[:,i] = (Vf[:,i] - Vf[:,i].mean()) - Vf[:,i].std()
        return Vf

    def make_cats(self, G, n, m, rho, ell):
        '''
        Creates type tensor Vf.

        :param G: Input graph.
        :param n: Number of nodes.
        :param m: Number of edges.
        :param rho: Number of node types.
        :param ell: Number of edge types.
        :return: Tensor of size n+m+1 x 1 containing the node/edge types. Vc[node_inds[v]] contains the type of node v, Vf[edge_inds[e]] contains the type of edge e.
        '''

        Vc = torch.LongTensor(torch.zeros(n+m+1).long())
        for node in G:
            node_type = self.node_type_inds[G.nodes[node]['type']]
            Vc[self.node_inds[node]] = node_type
        for edge in G.edges:
            edge_type = self.edge_type_inds[G.edges[edge]['type']]
            Vc[self.edge_inds[edge]] = edge_type
        Vc[n] = rho + ell
        return Vc

    def get_batch_paths(self, batch_pairs, T):
        """
        Collects the nodes ands edges for each path for each pair in batch_pairs. Padds with -1 to reach
        dim_size*max_path_steps entries for each pair. When used to index Vf, Vc -1 selects the last element which
        corresponds to a node with all zero features and a unique type used for padding.

        :param batch_pairs: An iterable with node pairs
        :param T: Path dictionary T.
        :return: A 1-d array with all steps of the paths and the maximum number of paths per pair.
        """

        dim_size = max([len(T[v][u]) for v, _, u in batch_pairs])
        all_steps = []
        for v, _, u in batch_pairs:
            trip_steps = []
            paths = T[v][u]
            for path in paths:
                path_steps = []
                ext_path = [v] + list(path) + [u]
                for i in range(1, len(ext_path)-1):
                    if i % 2 == 0:
                        path_steps.append(self.node_inds[ext_path[i]])
                    else:
                        step_edge = (ext_path[i-1],ext_path[i+1],0)
                        path_steps.append(self.edge_inds[step_edge])
                #each path is padded with -1 up to length max_path_steps
                path_steps += (self.max_path_steps - len(path_steps)) * [-1]
                trip_steps += path_steps
            #the entries for each pair are padded with -1 to dim_size*max_path_steps
            trip_steps += int((dim_size - len(trip_steps)/self.max_path_steps)) * [-1] * self.max_path_steps
            all_steps += trip_steps
        return np.array(all_steps), dim_size

    def forward(self, batch_pairs):
        """
        Calculates the scores for each pair using tensor operators available in torch. The tensors used internally are:

        Vf_slc: Contains the features for the edges/nodes of each path for each pair in batch_pairs. The vectors are
                layed out in the 3d tensor as follows: first dimension represents the path, second dimension represents
                the step of the path and 3rd dimension contains the features. The paths for different nodes are stacked
                along the first dimension. max_paths_num entries correspond to each entry of batch_pairs. When a pair
                has < max_paths_num paths the remaining 2d slices are padded with zeros, selected by the -1 values in
                all_steps variable. E.g., Vf_slc[0][1][2] contains the third feature of the second element of the first
                path of the first pair, Vf_slc[max_paths_num+1][1][2] contains the third feature of the second element
                of the second path of the second pair. The tensor's shape is (len(batch_pairs) x max_paths_num,
                max_path_steps, max_feat_dim).

        Vc_slc: Contains the type for the edge/nodes of each path for each pair in batch_pairs. The layout is similar
                Vf_slc: first dimension represents the path, second dimension represents the node/edge, and the entries
                for each pair are padded to max_paths_num in the first dimension and max_path_step in the second. E.g.,
                Vc_slc[0][1] contains an integer that represents the type of the second element of the first path of
                the first pair, Vc_slc[max_paths_num+1][1] contains an integer that represents the type of the second
                element of the second path of the second pair. The tensor's shape is (len(batch_pairs) x max_paths_num,
                max_path_steps)

        W1_slc: Contains the weight for each element of Vf_slc. The weights are selected according to the types of
                Vf_slc.  Has the same shpae and layout as Vf_slc.

        W2_slc: Contains the constant weight, due to the type, for each element in Vc_slc. Has the same shape and layout
                as Vc_slc.

        W2_mask: Used to mask the nodes used for padding in W2_slc.

        :param batch_pairs: An iterable with node pairs.
        :return: A 1-d tensor with a scoreor each pair.
        """

        activ1 = nn.LeakyReLU()
        activ2 = torch.sigmoid

        all_steps, max_paths_num = self.get_batch_paths(batch_pairs, self.T)

        Vc_slc = self.Vc[all_steps]
        Vf_slc = self.Vf[all_steps].reshape(int(len(all_steps) / self.max_path_steps), self.max_path_steps, self.max_feat_dim)
        if self.feat_transform_func:
            Vf_slc = self.feat_transform_func(Vf_slc, Vc_slc, batch_pairs, max_paths_num, self.node_inds, self.edge_inds, self.Vf, self.Vc)
        Vf_slc[all_steps.reshape(-1, self.max_path_steps) == -1, :] = 0
        pd.DataFrame(Vf_slc).to_csv('vf')

        W1_slc = self.W1[Vc_slc].reshape(int(len(all_steps)/self.max_path_steps), self.max_path_steps, self.max_feat_dim)
        W2_slc = self.W2[Vc_slc].reshape(int(len(all_steps)/self.max_path_steps), self.max_path_steps)
        W2_mask = torch.ones(len(all_steps))
        W2_mask[all_steps == -1] = 0
        W2_mask = W2_mask.reshape(int(len(all_steps)/self.max_path_steps), self.max_path_steps)

        feat_prod = Vf_slc * W1_slc
        out = activ2(activ1( (W2_slc * W2_mask + feat_prod.sum(2)).sum(1) ).reshape(len(batch_pairs), max_paths_num).sum(1)+self.b)

        return out

    def predict_train(self, pairs):
        """
        Wraps forward()

        :param pairs: An iterable with node pairs.
        :return: A 1-d tensor with a score for each pair.
        """

        return self.forward(pairs)

    def predict(self, pairs):
        """
        Wraps forward()

        :param pairs: An iterable of node pairs.
        :return: A 1-d numpy array with a score for each pair.
        """

        return self.forward(pairs).detach().numpy()

    def predict_batch(self, pairs, batch_size):
        """
        Calls forward() incrementally.

        :param pairs: An iterable with node pairs
        :param batch_size: Size of increment.
        :return: A 1-d tensor with a score for each pair.
        """

        pred = np.zeros(len(pairs))
        start = 0
        while start < len(pairs):
            end = min(start + batch_size, len(pairs))
            pred[start:end] = self.predict(pairs[start:end])
            start = end
        return pred

    def get_params(self):
        """
        :return: A dictionary with the parameters of the model.
        """

        return {'W1': self.W1.clone(), 'W2': self.W2.clone(), 'b': self.b.clone(), 'Vf': self.Vf.clone(), 'Vc': self.Vc.clone()}

    def add_paths(self, T):
        """
        Adds alla paths from the input path dictionary to the existing path dictionary

        :param T: A path dictionary T.
        """

        for u in T:
            if u not in self.T:
                self.T[u] = {}
            for v in T[u]:
                if v not in self.T[u]:
                    self.T[u][v] = set()
                for path in T[u][v]:
                    self.T[u][v].add(path)


def calc_batch_loss_mse(model, batch, alpha=None):
    """
    :param model: A PathL model.
    :param batch: An iterable with node pairs.
    :param alpha: Negative sample parameter.
    :return: MSE loss.
    """

    pairs, labs = batch

    ones = labs == 1
    zeros = labs == 0

    if np.sum(ones) == 0:
        return False, None

    if alpha:
        loss_pairs = []
        loss_labs = []

        selected = list(np.where(ones)[0]) + list(np.random.choice(np.where(zeros)[0], int(alpha*np.sum(ones)), replace=False))
        for i in selected:
            loss_pairs.append(list(pairs[i]))
            loss_labs.append(labs[i])

        loss_labs = torch.Tensor(loss_labs)
    else:
        loss_pairs = pairs
        loss_labs = torch.Tensor(labs)

    preds = model.predict_train(loss_pairs)
    #print(preds)
    loss = torch.nn.MSELoss(reduction='mean')(preds, loss_labs)
    return True, loss


def calc_batch_loss_cross_ent(model, batch):
    """
    :param model: A PathL model.
    :param batch: An iterable with node pairs.
    :return: Cross entropy Loss.
    """

    pairs, labs = batch

    zero_trips = pairs[labs == 0]
    one_trips = pairs[labs == 1]

    if len(one_trips) == 0:
        return False, None

    zero_preds = model.predict_train(zero_trips)
    one_preds = model.predict_train(one_trips)

    loss_preds = torch.Tensor(torch.zeros((len(one_trips),len(zero_trips)+1)))
    for i, pair in enumerate(one_trips):
        loss_preds[i, 0] = one_preds[i]
        for j, zero_trip in enumerate(zero_trips):
            loss_preds[i, 1+j] = zero_preds[j]
    loss_labs = torch.Tensor(torch.zeros(len(one_trips))).long()
    #print(loss_preds)
    loss = torch.nn.CrossEntropyLoss(reduction='mean')(loss_preds, loss_labs)
    return True, loss


def calc_batch_loss_cross_ent_list(model, batch):
    """
    List-wise Cross-Entropy loss, described in "Baoxu Shi, Tim Weninger, ProjE: Embedding Projection for KnowledgeGraph Completion, AAAi 2017".

    :param model: A PathL model.
    :param batch: An iterable with node pairs.
    :return: List-wise cross entropy loss.
    """

    pairs, labs = batch

    # if len(one_trips) == 0:
    #     return False, None

    all_preds = model.predict_train(pairs)
    ones_mask_norm = torch.Tensor(labs / sum(labs))

    H = torch.exp(all_preds)/torch.exp(all_preds).sum()
    L = - (H * ones_mask_norm).sum()

    return True, L


def calc_batch_loss_cross_ent_val(model, batch, val_step, max_val_size=10000):
    """
    Cross-Entropy loss used for validation. Applies calc_batch_loss_cross_ent incrementally in steps of size val_step

    :param model: A PathL model.
    :param batch: An iterable with node pairs.
    :param val_step: Increment size.
    :param max_val_size: Maximum size for batch.
    :return: Cross entropy loss.
    """

    pairs, labs = batch

    pairs = pairs[0:min(max_val_size, len(pairs))]
    labs = labs[0:min(max_val_size, len(labs))]

    zero_trips = pairs[labs == 0]
    one_trips = pairs[labs == 1]

    if len(one_trips) == 0:
        return False, None

    zero_preds = model.predict_batch(zero_trips,val_step)
    one_preds = model.predict_batch(one_trips,val_step)

    loss_preds = torch.Tensor(torch.zeros((len(one_trips),len(zero_trips)+1)))
    for i, pair in enumerate(one_trips):
        loss_preds[i, 0] = one_preds[i]
        for j, zero_trip in enumerate(zero_trips):
            loss_preds[i, 1+j] = zero_preds[j]
    loss_labs = torch.Tensor(torch.zeros(len(one_trips))).long()
    #print(loss_preds)
    loss = torch.nn.CrossEntropyLoss(reduction='mean')(loss_preds, loss_labs)
    return True, loss


def calc_batch_loss_bce(model, batch):
    """
    :param model: A PathL model.
    :param batch: An iterable with node pairs.
    :return: Binary cross entropy loss
    """

    pairs, labs = batch

    preds = model.predict_train(pairs)

    if not any(labs == 1):
        return False, None

    labs = torch.Tensor(labs)

    #rate = np.sum(zeros)/np.sum(ones)
    #weights = torch.Tensor(torch.ones(len(preds)))
    #weights[ones] = rate

    loss = torch.nn.BCELoss(reduction='mean')(preds,labs)
    return True, loss


def calc_batch_loss_margin(model, batch, margin=1):
    """
    :param model: PathL model.
    :param batch: An iterable with node pairs.
    :param margin: Margin parameter.
    :return: Hinge loss.
    """

    pairs, labs = batch

    preds = model.predict_train(pairs)

    if not any(labs == 1):
        return False, None
    labs = torch.Tensor((labs-0.5)*2)

    #rate = np.sum(zeros)/np.sum(ones)
    #weights = torch.Tensor(torch.ones(len(preds)))
    #weights[ones] = rate
    hinge_loss = margin - torch.mul(preds, labs)
    hinge_loss[hinge_loss < 0] = 0

    return True, torch.mean(hinge_loss)


def make_batches(train_set, train_labs, batch_size):
    """
    Generates the training batches. Performs random shuffling.

    :param train_set: All training pairs.
    :param train_labs: All training labels.
    :param batch_size: Batch size.
    :return: A list with batches.
    """

    rand_inds = random.sample(range(len(train_set)),len(train_set))
    rand_inds = rand_inds[0:len(rand_inds)//batch_size*batch_size]
    #rand_inds = list(range(len(train_set)))
    batches = []
    batch_index = 0
    while batch_index*batch_size < len(rand_inds):
        if (len(rand_inds)>=batch_index*batch_size+batch_size):
            batch_inds = rand_inds[batch_index * batch_size:batch_index * batch_size + batch_size]
        else:
            remain = len(rand_inds) - batch_index * batch_size
            batch_inds = rand_inds[batch_index * batch_size:batch_index * batch_size + remain]
        batch = [train_set[batch_inds], train_labs[batch_inds]]
        batches.append(batch)
        batch_index += 1
    return batches


def calc_val_loss_mr(model, test_set, test_labs, val_step=4096):
    """
    :param model: A PathL model.
    :param test_set: Test pairs.
    :param test_labs: Test labels.
    :param val_step: Increment step fro incremental calculation.
    :return: Mean rank error.
    """

    pred = model.predict_batch(test_set, val_step)
    neg = pred[test_labs == 0]
    pos = pred[test_labs == 1]
    all_ranks = np.zeros(len(pos))
    for i, p in enumerate(pos):
        all_ranks[i] = np.sum(neg > p) + np.sum(neg == p) / 2
    return np.mean(all_ranks)/len(test_set)


def train_model(G, T, train_set, train_labs, val_set, val_labs, epochs=100, batch_size=1024, batches_per_val=20, learning_rate=0.01, loss ='MCE-l', weight_decay=0, lr_step=20, time_lim_hours=None, batch_lim=None, prnt=True, out_path='', feat_transform_func=None):

    train_set, train_labs, val_set, val_labs = np.array(train_set), np.array(train_labs), np.array(val_set), np.array(val_labs)
    model = PathL(G, T, feat_transform_func)
    if loss == 'MSE':
        calc_batch_loss = calc_batch_loss_mse
    elif loss == 'MCE':
        calc_batch_loss = calc_batch_loss_cross_ent
    elif loss == 'MCE-l':
        calc_batch_loss = calc_batch_loss_cross_ent_list
    elif loss == 'BCE':
        calc_batch_loss = calc_batch_loss_bce
    elif loss == 'MAR':
        calc_batch_loss = calc_batch_loss_margin

    optm = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optm, lr_step, 0.1, -1)

    train_losses = []
    train_inds = []
    val_losses = []
    val_mrrs = []
    val_inds = []
    global_batch_count = 1
    start = time.time()
    f = open(out_path+'train_output', 'w')
    min_val_mrr = float('inf')
    try:
        for epoch in range(epochs):
            batches = make_batches(train_set, train_labs, batch_size)
            for batch_num, batch in enumerate(batches):
                if prnt:
                    print('epoch:' + str(epoch) + ', batch:' + str(batch_num) + '/' + str(len(batches)))
                f.write('epoch:' + str(epoch) + ', batch:' + str(batch_num) + '/' + str(len(batches))+'\n')
                done, batch_loss = calc_batch_loss(model, batch)
                if not done:
                    continue
                optm.zero_grad()
                batch_loss.backward()
                optm.step()
                if prnt:
                    print('batch loss:' + str(float(batch_loss)))
                    for param_group in optm.param_groups:
                        print(param_group['lr'])
                f.write('batch loss:' + str(float(batch_loss)) + '\n' + str(model.W1) + '\n' + str(model.W1.grad)  + '\n' + str(model.W2) + '\n' + str(model.W2.grad) + '\n' + str(model.b) + '\n' + str(model.b.grad)+ '\n')
                train_losses.append(float(batch_loss))
                train_inds.append(global_batch_count)
                if batch_num % batches_per_val == 0 and epoch+batch_num > 0:
                    with torch.no_grad():
                        val_mrr = calc_val_loss_mr(model, val_set, val_labs)
                        done, val_loss = calc_batch_loss(model, (val_set, val_labs))
                        val_loss = val_loss.detach().numpy()
                    if prnt:
                        print('------------------------------- val mrr:' + str(val_mrr) + '-------------------------------')
                        print('------------------------------- val loss:' + str(val_loss) + '-------------------------------')
                    f.write('------------------------------- val mrr:' + str(val_mrr) + '-------------------------------\n')
                    f.write('------------------------------- val loss:' + str(val_loss) + '-------------------------------\n')
                    val_mrrs.append(val_mrr)
                    val_losses.append(val_loss)
                    val_inds.append(global_batch_count)
                    if val_mrr<min_val_mrr:
                        min_val_mrr = val_mrr
                        best_model = model.get_params()
                if time_lim_hours and time.time() - start > time_lim_hours * 60 * 60 or batch_lim and global_batch_count == batch_lim:
                    raise Exception('time out')
                global_batch_count += 1
                f.flush()
            scheduler.step()
    except:
        traceback.print_exc()
    finally:
        return PathL(model.G, model.T, model.feat_transform_func, best_model), model, (train_inds, train_losses), (val_inds, val_mrrs, val_losses)


def apply_model(model, pairs, T=None):
    """
    Applies the model on a given set of pairs. If model.T does not contains entries for the fiven set of
    pairs, these entries must be provided in T argument.
    :param model: A PathL model.
    :param pairs: An iterable with node pairs.
    :param T: A path dictionary.
    :return: A score for each node pair.
    """

    if T:
        model.add_paths(T)
    return model.predict(pairs)


