FROM ubuntu:20.04
FROM python:3.8

MAINTAINER Vadim Ahmetov <vadim32654@gmail.com>

COPY . /apps

RUN pip install -r /apps/requirements.txt

WORKDIR /apps

CMD ["uvicorn", "myapi:app", "--reload"]
