# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3

ARG PYTHON_VERSION=3.12
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SOCIALGRAPH_FM_HOME=/var/lib/socialgraph-fm

RUN apt-get update \
    && apt-get install -y --no-install-recommends python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python3-pip ca-certificates \
    && python${PYTHON_VERSION} -m venv /opt/socialgraph-fm-venv \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/socialgraph-fm-venv/bin:${PATH}"

WORKDIR /opt/socialgraph-fm-gfm
COPY constraints ./constraints
COPY locks ./locks
COPY contracts ./contracts
COPY configs ./configs
COPY runtime-profiles.json ./runtime-profiles.json
COPY install-profiles.json ./install-profiles.json
COPY pyproject.toml ./
COPY src ./src

# The hash lock and binary-only policy bind every resolved artifact to the
# Torch 2.12 / CUDA 13.0 profile and forbid accidental source builds.
RUN python -m pip install --no-cache-dir \
      --require-hashes \
      --only-binary=:all: \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --find-links https://data.pyg.org/whl/torch-2.12.0+cu130.html \
      -r locks/linux-cu130.requirements.txt \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation . \
    && python -m pip check

RUN mkdir -p \
      /var/lib/socialgraph-fm/core-runtime/serving \
      /var/lib/socialgraph-fm/core-runtime/serving-graphs \
      /var/lib/socialgraph-fm/core-runtime/api/dataset-store \
      /var/lib/socialgraph-fm/research \
      /var/lib/socialgraph-fm/socialgraph-global \
    && cp contracts/core-serving-control.json /var/lib/socialgraph-fm/core-runtime/serving/ \
    && cp contracts/core-serving-registry.json /var/lib/socialgraph-fm/core-runtime/serving/ \
    && cp contracts/core-serving-graph-catalog.json /var/lib/socialgraph-fm/core-runtime/serving/

ENTRYPOINT ["python", "-m", "socialgraph_gfm.core.inference_cli"]
# The public image is a component/test image, not a network deployment recipe.
# Keep the service confined to container loopback unless a separately reviewed
# authenticated deployment supplies a different command explicitly.
CMD ["--runtime-root", "/var/lib/socialgraph-fm/core-runtime/serving", "--serving-control", "/var/lib/socialgraph-fm/core-runtime/serving/core-serving-control.json", "--artifact-root", "/var/lib/socialgraph-fm/core-runtime/serving-graphs", "--research-root", "/var/lib/socialgraph-fm/research", "--global-model-root", "/var/lib/socialgraph-fm/socialgraph-global", "--dataset-store-root", "/var/lib/socialgraph-fm/core-runtime/api/dataset-store", "--token-file", "/var/lib/socialgraph-fm/core-runtime/serving/session.token", "--host", "127.0.0.1", "--port", "8766"]
