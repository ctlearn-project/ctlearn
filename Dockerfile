FROM nvcr.io/nvidia/tensorflow:24.12-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

FROM python:3.11 AS builder
#RUN pip install ctlearn
# install git for setuptools_scm
RUN apt update \
    && apt install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# add necessary sources, including .git for version info
COPY ./pyproject.toml MANIFEST.in /repo/
COPY ./ctlearn ./repo/ctlearn/
COPY ./.git ./repo/.git/

# build the wheel
RUN python -m pip install --no-cache-dir build \
    && cd repo \
    && python -m build --wheel


# second stage, copy and install wheel
FROM python:3.11
COPY --from=builder /repo/dist /tmp/dist

RUN python -m pip install --no-cache-dir /tmp/dist/* \
    && rm -r /tmp/dist

RUN addgroup --system ctlearn && adduser --system --group ctlearn
USER ctlearn
