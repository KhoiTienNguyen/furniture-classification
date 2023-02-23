FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install python3 -y
RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3-pip

RUN pip3 install --upgrade pip
RUN apt-get install python3-flask -y

# Copy repo to container
COPY model.py /app/
COPY requirements.txt /app/
COPY best_model.hdf5 /app/


WORKDIR /app/

# Install python requirements
RUN pip3 install -r requirements.txt


# we can expose a port to listen on 
EXPOSE 8081

# Run API
CMD ["python3", "-m", "gunicorn", "--bind", "0.0.0.0:8081", "--workers", "3", "app:app"]