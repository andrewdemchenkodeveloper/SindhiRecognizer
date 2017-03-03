FROM debian:sid

RUN apt-get update
RUN apt-get install -y python python-dev python-pip python-virtualenv
RUN apt-get install -y python-opencv
RUN apt-get install -y python-matplotlib
RUN pip install numpy==1.11.3
RUN pip install python-docx==0.8.6
RUN pip install sklearn==0.0
RUN pip install scipy==0.18.1

WORKDIR /code

CMD ["bash"]
