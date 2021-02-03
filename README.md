# Path Learn

## Overview

This repository contains an implementation of Path Learn model for link prediction in Heterogeneous Information Networks. Path Learn predicts the formation of links between nodes by modelling the paths that connect the nodes on the graph, taking into account their node/edge types and features. The repository contains a Python module that implements the model in PyTorch, a REST API service, that can be run using Docker or as a Python program, and a web application that uses the model, implemented with the Streamlit framework.

## Python Module

The Python module is available in src/path_learn.py. The documentation is available [here](https://smartdatalake.github.io/pathlearn/). The module requires torch and networkx packages.


## REST API

The implementation of the REST API service is available in src/path_learn_flask.py.  

The Docker image is built with the command:

```
sudo docker build -t path_learn_flask .
```

and deployed with the command:


```
sudo docker run --rm --name dock_pl -p <port>:5000 -v <local_data_directory>:/data path_learn_flask
```

The data required for the tasks must be placed in local_data_directory (e.g. "$(pwd)"/data). The API is accesible at localhost:port

To run without Docker, first run the command:

```
export FLASK_APP=src/flask_interface.py
```

to set the module that will be used by Flask, and then run Flask with the command:

```
flask run -h localhost -p <port>
```

The REST API additionaly requires packages flask-restx and pandas.

## Web Application

The Web application is implemented in src/path_learn_st.py and is deployed with the command:


```
streamlit run src/path_learn_st.py <config_file_path>
```

The config_file_path must be the path to a json file with the keys for using the app and the path that will be used to read and store the data (e.g., config.json.example in this repository). The application additionally requires the packages altair, pandas and sklearn.




