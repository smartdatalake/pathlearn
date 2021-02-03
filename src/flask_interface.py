import path_learn as pl
import preprocessing as prc
import networkx as nx
import pickle
import traceback
import os

from flask import Flask, request
from flask_restx import Api, Resource
from werkzeug.middleware.proxy_fix import ProxyFix

try:
    from feat_transform import feat_transform_func
except:
    feat_transform_func = None

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

api = Api(app)
app.wsgi_app = ProxyFix(app.wsgi_app)
ns = api.namespace('PathLearn', description='REST api for the PathLearn model.')


@ns.route('/lstypes')
@ns.doc(
    description = 'Returns json with node edge types of a graph.',
    params = {
    'graph': '''Path to a directory that contains files representing a graph. The directory must \
                contain a subdirectory named \'nodes\' with .csv files named after the node \
                types of the graph (e.g., type-1.csv), without headers, where the first column is the id of the \
                node and subsequent columns are numerical values representing features. The directory must \
                also contain a subdirectory named \'relations\' with a file named relations.csv, without header,
                with columns the following columns: source node id, destination node id, edge_type, feature-1, feature-2, etc.'''},
    responses = {200: 'Success', 500: 'Internal server error'}
)
class fl_lstypes(Resource):
    def get(self):
        try:
            graph = request.args.get('graph', type=str)
            types = prc.types_from_files(graph)
        except:
            api.abort(500, traceback.format_exc())
        return types, 200


@ns.route('/lspath')
@ns.doc(
    description = 'Returns json with node edge types of a graph.',
    params = {
    'path': 'Path to list.'},
    responses = {200: 'Success', 500: 'Internal server error'}
)
class fl_lsdatasets(Resource):
    def get(self):
        try:
            path = request.args.get('path', type=str)
            os.listdir(path)
        except:
            api.abort(500, traceback.format_exc())
        return os.listdir(path), 200


@ns.route('/preproc')
@ns.doc(
    description = 'Call to generate the data structures used to train the model.',
    params = {
    'graph': '''Path to a directory that contains files representing a graph. The directory must \
                contain a subdirectory named \'nodes\' with .csv files named after the node \
                types of the graph (e.g., type-1.csv), without headers, where the first column is the id of the \
                node and subsequent columns are numerical values representing features. The directory must \
                also contain a subdirectory named \'relations\' with a file named relations.csv, without header,
                with columns the following columns: source node id, destination node id, edge_type, feature-1, feature-2, etc.''',
    'train': 'Number of training edges.',
    'val':   'Number of validation edges.',
    'test':  'Number of test_Edges.',
    'edge':  'Modelled edge type',
    'steps': 'Maximum path length.',
    'neg':   'Number of negative samples.',
    'out':   'Path to write the output data.'},
    responses = {200: 'Success', 500: 'Internal server error'}
)
class fl_preproc(Resource):
    def get(self):
        try:
            graph = request.args.get('graph', type=str)
            train = request.args.get('train', type=int)
            val = request.args.get('val', type=int)
            test = request.args.get('val', type=int)
            edge = request.args.get('edge', type=str)
            steps = request.args.get('steps', type=int)
            neg = request.args.get('neg', type=int)
            out = request.args.get('out', type=str)
            print('reading graph files', flush=True)
            G = prc.graph_from_files(graph)
            print('generating preproc data', flush=True)
            train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels, T = prc.make_train_data(G, edge, train, val, test, steps, neg)
            print('writing out', flush=True)
            with open(out, 'wb') as f:
                pickle.dump([G, train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels, T], f)
        except:
            api.abort(500, traceback.format_exc())
        return 200


@ns.route('/train')
@ns.doc(
    description = 'Call to train the model.',
    params = {
             'data': 'Path to data structure file to use for training.',
             'epochs': 'Epochs for training',
             'batch': 'Batch size',
             'lr': 'Learning rate',
             'val' : 'Batches per val.',
             'wd': 'Weight_decay',
             'lr_st': 'Epochs until learning rate is scheduled to decrease.',

             'out': 'Path to write the resulting model and training/validation errors.'
         },
    responses = {200: 'Success', 500: 'Internal server error'}
)
class fl_train(Resource):
    def get(self):
        try:
            data = request.args.get('data', type=str)
            out = request.args.get('out', type=str)
            config = {}
            epochs = request.args.get('epochs', type=int)
            if epochs:
                config['epochs'] = epochs
            batch_size = request.args.get('batch', type=int)
            if batch_size:
                config['batch_size'] = batch_size
            learning_rate = request.args.get('lr', type=float)
            if learning_rate:
                config['learning_rate'] = learning_rate
            weight_decay = request.args.get('wd', type=float)
            if weight_decay:
                config['weight_decay'] = weight_decay
            lr_step = request.args.get('lr_st', type=int)
            if lr_step:
                config['lr_step'] = lr_step
            batches_per_val = request.args.get('val', type=int)
            if batches_per_val:
                config['batches_per_val'] = batches_per_val

            print('loading data', flush=True)
            with open(data, 'rb') as f:
                G, train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels, T = pickle.load(f)
            print('training', flush=True)
            res = pl.train_model(G, T, train_pairs, train_labels, val_pairs, val_labels,
                                 feat_transform_func=feat_transform_func, **config)
            print('writing out', flush=True)
            with open(out, 'wb') as f:
                pickle.dump(res[0], f)
            val_losses = [float(n) for n in res[3][2]]
        except:
            api.abort(500, traceback.format_exc())
        return {'train_batch_indicess':list(res[2][0]),'train_batch_loss':list(res[2][1]),'val_batch_indicess':list(res[3][0]),'val_batch_loss':val_losses}, 200


@ns.route('/predict')
@ns.doc(
    description = 'Call to apply the model. Returns a json with candidate nodes and scores.',
    params = {'model': 'Path a pickled PathL object.',
              'node': 'The id of the node to predict links for.'},
    responses={200: 'Success ', 500: 'Internal server error'}
)
class fl_predict(Resource):
    def get(self):
        try:
            model_path = request.args.get('model', type=str)
            src_node = request.args.get('node', type=int)
            print('loading model', flush=True)
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print('finding candidates pairs', flush=True)
            cand_type = prc.find_candidate_type(model.G, model.max_path_steps, src_node)
            # candidates are all nodes of the corredct type with distance <= maximum path length
            candidates = [n for n in nx.ego_graph(model.G, src_node, range) if
                          model.G.nodes[n]['type'] == cand_type and n != src_node]
            test_pairs = [[src_node, model.edge_type, cand] for cand in candidates]
            T = prc.find_pair_paths(model.G, test_pairs, (model.max_path_steps + 1) / 2)
            print('applying model', flush=True)
            scores = pl.apply_model(model, test_pairs, T)
        except:
            api.abort(500, traceback.format_exc())
        return {cand: float(scr) for cand, scr in zip(candidates, scores)}, 200


@ns.route('/test')
@ns.doc(
    description = 'Call to test the model. Returns MRR and test l-MCE',
    params = {'model': 'Path a pickled PathL object.',
              'data': 'Path to preprocessed data file.'},
    responses={200: 'Success ', 500: 'Internal server error'}
)
class fl_test(Resource):
    def get(self):
        try:
            model_path = request.args.get('model', type=str)
            data = request.args.get('data', type=int)
            print('loading data', flush=True)
            with open(data, 'rb') as f:
                G, train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels, T = pickle.load(f)
            print('loading model', flush=True)
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print('applying model', flush=True)
            val_mrr = pl.calc_val_loss_mr(model, test_pairs, test_labels)
            done, val_loss = pl.calc_batch_loss(model, (test_pairs, test_labels))
        except:
            api.abort(500, traceback.format_exc())
        return [val_mrr,val_loss], 200