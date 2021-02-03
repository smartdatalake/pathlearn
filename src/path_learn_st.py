import streamlit as st
import numpy as np
import os
import torch
import traceback
import networkx as nx
import altair as alt
import importlib
import json
import sys
import importlib.util

from preprocessing import graph_from_files
from preprocessing import make_train_data
from preprocessing import find_candidate_type
from preprocessing import find_pair_paths
from pickle import dump
from pickle import load
from path_learn import PathL
from path_learn import calc_batch_loss_cross_ent_list
from path_learn import make_batches
from path_learn import calc_val_loss_mr
from path_learn import apply_model
from pandas import DataFrame
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def train_model_st(G, T, train_set, train_labs, val_set, val_labs, epochs=100, batch_size=1024, batches_per_val=20,
                learning_rate=0.01, weight_decay=0, lr_step=20, feat_transform_func=None):
    train_set, train_labs, val_set, val_labs = np.array(train_set), np.array(train_labs), np.array(val_set), np.array(
        val_labs)
    model = PathL(G, T, feat_transform_func)
    calc_batch_loss = calc_batch_loss_cross_ent_list

    optm = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optm, lr_step, 0.1, -1)

    train_losses = []
    train_inds = []
    val_losses = []
    val_mrrs = []
    val_inds = []
    global_batch_count = 1
    min_val_mrr = float('inf')
    prog = st.progress(0.0)
    chart_tr = st.line_chart()
    chart_val = st.line_chart()
    chart_val_mr = st.line_chart()
    val_loss = 0
    try:
        for epoch in range(epochs):
            batches = make_batches(train_set, train_labs, batch_size)
            for batch_num, batch in enumerate(batches):
                prog.progress((epoch*len(batches)+batch_num) / (epochs*len(batches)))
                print('epoch:' + str(epoch) + ', batch:' + str(batch_num) + '/' + str(len(batches)))
                done, batch_loss = calc_batch_loss(model, batch)
                df = DataFrame([{'train loss': float(batch_loss)}], index=[global_batch_count])
                df.index.rename('Batch',inplace=True)
                chart_tr.add_rows(df)
                if not done:
                    continue
                optm.zero_grad()
                batch_loss.backward()
                optm.step()
                print('batch loss:' + str(float(batch_loss)))
                train_losses.append(float(batch_loss))
                train_inds.append(global_batch_count)
                if batch_num % batches_per_val == 0 and epoch + batch_num > 0:
                    with torch.no_grad():
                        val_mrr = calc_val_loss_mr(model, val_set, val_labs)
                        done, val_loss = calc_batch_loss(model, (val_set, val_labs))
                        val_loss = val_loss.detach().numpy()
                    df = DataFrame([{'val loss': float(val_loss)}], index=[global_batch_count])
                    df.index.rename('Batch', inplace=True)
                    chart_val.add_rows(df)
                    df = DataFrame([{'val mr': float(val_mrr)}], index=[global_batch_count])
                    df.index.rename('Batch', inplace=True)
                    chart_val_mr.add_rows(df)
                    val_mrrs.append(val_mrr)
                    val_losses.append(val_loss)
                    val_inds.append(global_batch_count)
                    if val_mrr < min_val_mrr:
                        min_val_mrr = val_mrr
                        best_model = model.get_params()
                global_batch_count += 1
            scheduler.step()
    except:
        traceback.print_exc()
    finally:
        return PathL(model.G, model.T, model.feat_transform_func, best_model), model, (train_inds, train_losses), (val_inds, val_mrrs, val_losses)


@st.cache(allow_output_mutation=True,show_spinner=False)
def load_data(path):
    G = graph_from_files(path)
    edge_types = list(set([G[e[0]][e[1]][e[2]]['type'] for e in G.edges]))
    return G, edge_types


@st.cache(allow_output_mutation=True,show_spinner=False)
def dump_model(model,out_path):
    with open(data_dir + out_path, 'wb') as f:
        dump(model,f)


@st.cache(allow_output_mutation=True,show_spinner=False)
def load_model(model_path):
    with open(data_dir + model_path, 'rb') as f:
        model = load(f)
    return model


@st.cache(allow_output_mutation=True,show_spinner=False)
def load_preproc(data_path):
    with open(data_dir + data_path, 'rb') as f:
        G, train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels, T = load(f)
    return G, train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels, T


def run_preproc(G, edge_type, train,val,test,steps,neg,out):
    train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels, T = make_train_data(G, edge_type, train, val, test, steps, neg)
    with open(out, 'wb') as f:
        dump([G, train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels, T], f)


def preprocessing_page():
    st.title('Preprocessing')

    graph_path = st.selectbox('Graph path', ['<select path>'] + list(os.listdir(data_dir)))

    edge_types = []
    if graph_path != '<select path>':
        G, edge_types = load_data(data_dir + graph_path)
    else:
        st.warning('select path')

    edge_type = st.selectbox('Edge type', edge_types)
    train = st.number_input('Training edges', value=1000, step=1)
    val = st.number_input('Validation edges', value=100, step=1)
    test = st.number_input('Test edges', value=100, step=1)
    steps = st.number_input('Max path length', value=3, step=1)
    neg = st.number_input('Negative sample', value=4, step=1)
    out = st.text_input('Data output file')

    if st.button('run preprocesing', key=None):
        run_preproc(G, edge_type, train, val, test, steps, neg, data_dir + out)


def training_page():

    st.title('Training')

    data_path = st.selectbox('data path', ['<select path>'] + list(os.listdir(data_dir)))

    epochs = st.number_input('epochs', value=5, step=1)
    lr = st.number_input('learning rate', value=0.01, step=0.0001, format="%.4f")
    batch_size = st.number_input('batch size', value=256, step=1)
    batches_per_val = st.number_input('batches per validation', value=20, step=1)
    lr_step = st.number_input('learning rate step epochs', value=3, step=1)
    wd = st.number_input('weight decay', value=0, step=1)

    func_file = st.selectbox('feature transform module', ['<select file>'] + list(os.listdir('src/')))
    if func_file != '<select file>':
        feat_transform = importlib.import_module(func_file[0:-3])
        feat_transform_func = feat_transform.feat_transform_func
    else:
        feat_transform_func =None

    out = st.text_input('model output file')

    if data_path != '<select_path' and epochs != 0 and lr != 0 and batch_size !=0 and batches_per_val != 0 and lr_step!=0 and out != '':
        G, train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels, T = load_preproc(data_path)
        model, _, _, _ = train_model_st(G, T, train_pairs, train_labels, val_pairs, val_labels, epochs=epochs, learning_rate=lr,
                                        batch_size=batch_size, batches_per_val=batches_per_val, lr_step=lr_step, weight_decay=wd,
                                        feat_transform_func=feat_transform_func)
        with open(data_dir + out, 'wb') as f:
            dump(model, f)
    else:
        st.warning('select input')


def testing_page():
    st.title('Testing')

    def mean_rank(pred, test_labs):
        neg = pred[test_labs == 0]
        pos = pred[test_labs == 1]
        all_ranks = np.zeros(len(pos))
        for i, p in enumerate(pos):
            if i % 10000 == 0:
                print('{}/{}'.format(i, len(pos)))
            all_ranks[i] = np.sum(neg > p) + np.sum(neg == p) + 1
        return np.mean(all_ranks), np.mean(all_ranks) / len(test_labs), all_ranks

    def hits_mrr(res, labels, k):

        neg = res[labels == 0]
        pos = res[labels == 1]

        hits = len(set(np.argsort(res)[-k:]).intersection(set(np.where(labels == 1)[0])))
        mrr = 1 / (np.sum(neg > max(pos)) + int(np.sum(neg == max(pos))) + 1 if len(pos) > 0 else 1)

        return hits, mrr

    def hits_mrrs(scores, test_labels, test_pairs, k):

        test_nodes = set(test_pairs[:, 0])
        hits = []
        mrrs = []

        for i, node in enumerate(test_nodes):

            sel = test_pairs[:, 0] == node
            sel_scores = scores[sel]
            sel_labels = test_labels[sel]

            hm = hits_mrr(sel_scores, sel_labels, k)
            hits.append(hm[0])
            mrrs.append(hm[1])

        return np.mean(hits), np.mean(mrrs)

    data_path = st.selectbox('data path ', ['<select path>'] + list(os.listdir(data_dir)))
    model_path = st.selectbox('model path ', ['<select path>'] + list(os.listdir(data_dir)))

    if data_path != '<select path>' and model_path != '<select path>':
        G, train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels, T = load_preproc(data_path)
        model = load_model(model_path)
        scores = apply_model(model, test_pairs)
        scores = np.array(scores)
        test_pairs = np.array(test_pairs)
        test_labels = np.array(test_labels)
        mr = mean_rank(scores, test_labels)[1]
        hits, mrr = hits_mrrs(scores, test_labels, test_pairs,10)
        auc = roc_auc_score(test_labels, scores)
        scores_df = DataFrame([{'Mean Rank': mr, 'Hits@10': hits, 'MRR': mrr, 'AUC': auc}],index=['Score'])
        st.table(scores_df)
        fpr, tpr, _ = roc_curve(test_labels, scores)

        neg = DataFrame({'Negative Scores': scores[test_labels == 0]})
        pos = DataFrame({'Positive Scores': scores[test_labels == 1]})

        hist0 = alt.Chart(neg).mark_bar().encode(
            alt.X('Negative Scores', bin=alt.Bin(maxbins=20)),
            y='count()',
        ).properties(height=400, width=680)
        st.altair_chart(hist0)
        hist1 = alt.Chart(pos).mark_bar().encode(
            alt.X('Positive Scores', bin=alt.Bin(maxbins=20)),
            y='count()',
        ).properties(height=400, width=680).configure_mark(color='orange')
        st.altair_chart(hist1)

        roc_df = DataFrame({
            'True Positive Rate': tpr,
            'False Positive Rate': fpr
        })
        roc = alt.Chart(roc_df).mark_line().encode(
            x='False Positive Rate',
            y='True Positive Rate'
        ).properties(height=400, width=680)
        st.altair_chart(roc)
        del G, train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels, T, model
    else:
        st.warning('select data and model')


def prediction_page():

    def softmax(scores):
        return np.exp(scores) / np.exp(scores).sum()

    st.title('Prediction')

    model_path = st.selectbox('model path', ['<select path>'] + list(os.listdir(data_dir)))
    src_node = st.text_input('Node')
    k = st.number_input('Number of results', value=10, step=1)
    if model_path != '<select path>' and src_node != '':
        model = load_model(model_path)
        steps = (model.max_path_steps+1)/2
        try:
            cand_type = find_candidate_type(model.G, model.edge_type, src_node)
            candidates = [n for n in nx.ego_graph(model.G, src_node, steps) if
                          model.G.nodes[n]['type'] == cand_type and n != src_node and n not in model.G[src_node]]
            test_pairs = [[src_node, model.edge_type, cand] for cand in candidates]
            T_cand = find_pair_paths(model.G, test_pairs, steps)
            scores = apply_model(model, test_pairs, T_cand)
            res = DataFrame({'Candidates': [p[2] for p in test_pairs], 'Scores': scores})
            res.Scores = [round(float(sc),7) for sc in res.Scores]
            res.sort_values(by='Scores',ascending=False,inplace=True)
            res.index = range(1,len(res)+1)
            #st.write(res.iloc[0:k])
            st.table(res.iloc[0:k])
            chart = (
                alt.Chart(res.iloc[0:k])
                    .mark_bar()
                    .encode(alt.Y("Candidates", title="",sort=alt.SortField('Score')),
                            alt.X("Scores", title="", scale=alt.Scale(domain=[max(min(res.Scores.iloc[0:k]),0), max(res.Scores.iloc[0:k])])))
                    .properties(height=400, width=750)
            )
            text = chart.mark_text(align="left", baseline="middle", dx=3).encode(text="Scores")
        except:
            st.warning('Unknown Author')
        #st.altair_chart(chart + text)
    else:
        st.warning('select model')


key = st.text_input('key')

with open(sys.argv[1],'r') as f:
    config = json.load(f)

data_dir = config['data_dir']

if key in config['train-use']:
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio('', ['Preprocessing', 'Training', 'Testing', 'Prediction'])

    if selection == 'Preprocessing':
        preprocessing_page()
    if selection == 'Training':
       training_page()
    if selection == 'Testing':
        testing_page()
    if selection == 'Prediction':
        prediction_page()
elif key in config['use']:
    prediction_page()
elif key!='':
    st.warning('unknown key')


