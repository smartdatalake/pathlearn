import requests

def add_args(url,args):
    for arg in args:
        if url[-1] != '?':
            url += '&'
        url+=arg+'='+str(args[arg])
    return url

def preproc(args):
    url = 'http://localhost:5002/PathLearn/preproc?'
    url = add_args(url,args)
    print(url)
    response = requests.get(url)
    print(response.json())


def list_dir(args):
    url = 'http://localhost:5002/PathLearn/lspath?'
    url = add_args(url,args)
    print(url)
    response = requests.get(url)
    print(response.json())


def list_types(args):
    url = 'http://localhost:5002/PathLearn/lstypes?'
    url = add_args(url,args)
    print(url)
    response = requests.get(url)
    print(response.json())


def train(args):
    url = 'http://localhost:5002/PathLearn/train?'
    url = add_args(url, args)
    print(url)
    response = requests.get(url)
    print(response.json())


list_dir({'path': '/data'})

#list_types({'graph': '/data/dblp_graph_files'})

#preproc({'graph': '/data/dblp_graph_files','train': 100,'val': 10, 'test': 10, 'edge': 0, 'steps': 3, 'neg': 5, 'out': '/data/dblp_graph_files/preproc_data'})

train({'data': '/data/dblp_graph_files/preproc_data', 'data': '/data/dblp_graph_files/model'})

