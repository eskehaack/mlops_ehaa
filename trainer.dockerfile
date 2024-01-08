# Base image
FROM  nvcr.io/nvidia/pytorch:22.07-py3

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_ehaa/ mlops_ehaa/
COPY data/ data/
COPY Makefile Makefile

WORKDIR /
RUN make requirements

ENTRYPOINT ["python", "-u", "mlops_ehaa/train_model.py"]