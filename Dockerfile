# Stage 1: Build the ctlearn wheel using a standard Python image
FROM python:3.12 AS builder

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

# Build the wheel
RUN python -m build --wheel

# Stage 2: Use NVIDIA's PyTorch image; installing CTLearn also installs TensorFlow and Keras.
FROM nvcr.io/nvidia/pytorch:25.02-py3
# Copy only the built wheel from the builder stage's dist directory
COPY --from=builder /repo/dist /tmp/dist

# Install the CTLearn wheel and its declared runtime dependencies.
RUN python -m pip install --no-cache-dir /tmp/dist/* \
    && rm -r /tmp/dist

RUN addgroup --system ctlearn && adduser --system --group ctlearn
USER ctlearn