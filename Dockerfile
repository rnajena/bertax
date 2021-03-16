FROM continuumio/miniconda3
ENV VERSION 0.1
ENV TOOL bertax

RUN conda install -y -c fkretschmer $TOOL=$VERSION && conda clean -a
RUN pip install keras-bert==0.86.0

ENTRYPOINT ["bertax"]