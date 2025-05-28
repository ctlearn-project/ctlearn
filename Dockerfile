# Stage 1: Build the ctlearn wheel using a standard Python image
# Stage 1: Build the ctlearn wheel using a standard Python image
FROM python:3.11 AS builder

# Install git (needed for setuptools_scm during build) and build tool
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \

# Install git (needed for setuptools_scm during build) and build tool
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir build

# Copy source code needed for the build
WORKDIR /repo
COPY ./pyproject.toml MANIFEST.in ./
COPY ./ctlearn ./ctlearn/
# If .git is truly needed for versioning by setuptools_scm, copy it. Otherwise, omit.
COPY ./.git ./.git/
RUN pip install --no-cache-dir build

# Copy source code needed for the build
WORKDIR /repo
COPY ./pyproject.toml MANIFEST.in ./
COPY ./ctlearn ./ctlearn/
# If .git is truly needed for versioning by setuptools_scm, copy it. Otherwise, omit.
COPY ./.git ./.git/

# Build the wheel
RUN python -m build --wheel

# Stage 2: Create the final runtime image BASED ON NVIDIA's TF image
# TODO what version to use ? after 24.?? TF 2.14 is not found in the container
FROM nvcr.io/nvidia/tensorflow:24.01-tf2-py3 

# Copy only the built wheel from the builder stage's dist directory
# Build the wheel
RUN python -m build --wheel

# Stage 2: Create the final runtime image BASED ON NVIDIA's TF image
# TODO what version to use ? after 24.?? TF 2.14 is not found in the container
FROM nvcr.io/nvidia/tensorflow:24.01-tf2-py3 

# Copy only the built wheel from the builder stage's dist directory
COPY --from=builder /repo/dist /tmp/dist

# Install the ctlearn wheel using pip from the NVIDIA base image
# Install the ctlearn wheel using pip from the NVIDIA base image
RUN python -m pip install --no-cache-dir /tmp/dist/* \
    && rm -r /tmp/dist

RUN addgroup --system ctlearn && adduser --system --group ctlearn
USER ctlearn