FROM ubuntu
MAINTAINER pchronis
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN pip3 install networkx
RUN pip3 install torch
RUN pip3 install flask-restx
RUN pip3 install pandas
ADD src/path_learn.py /home/path_learn.py
ADD src/flask_interface.py /home/flask_interface.py
ADD src/preprocessing.py /home/preprocessing.py
ADD src/feat_transform.py /home/feat_transform.py
ENV FLASK_APP=/home/flask_interface.py
CMD ["run","-h","0.0.0.0"]
ENTRYPOINT ["flask"]
