FROM python:3.6

RUN apt-get update && \
    apt-get -y install libev-dev gcc gcc+ python3-dev g++ && \
    rm -rf /var/lib/apt/lists/*

MAINTAINER Imad Toubal


WORKDIR /app
ADD . /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install gunicorn==20.0.4

# set application PORT and expose docker PORT, 80 is what Elastic Beanstalk expects

ENV PORT 80
EXPOSE 80

ENTRYPOINT ["python", "/usr/local/bin/gunicorn", "config.wsgi", "-b 0.0.0.0:80" ,"-w=3", "--chdir=/app"]

