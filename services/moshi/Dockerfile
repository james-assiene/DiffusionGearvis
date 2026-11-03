# moshi-server (Kyutai STT) built against CUDA, run as a container.
#
# Why a container rather than a host binary: services/stt.sh originally exec'd `moshi-server` from
# the host PATH, which needs both the CUDA toolkit (nvcc) to build and the CUDA runtime libraries
# to run. This box has neither, and `.claude/rules/gpu-services.md` says no system CUDA changes.
# Building in a container and copying the binary out only moves the problem: the binary still
# links libcudart at runtime. So we build AND run in the container, exactly as vLLM does.
#
# CUDA 12.8.1, and it is the ONLY version that works. Both bounds are read from source, not
# guessed — three builds were burned discovering this the slow way:
#   - UPPER: this server 0.6.4 depends on cudarc 0.16.2, whose build.rs hard-codes its supported
#     toolkits and panics with "Unsupported cuda toolkit version" on anything outside them. The
#     list ends at 12.8 (13.2 and 12.9 were both rejected here, empirically).
#   - LOWER: sm_120 (Blackwell / RTX 5090) needs 12.8+; 12.6 and earlier cannot emit kernels this
#     card can run.
# Upper == lower == 12.8. There is no second choice, so do not "modernise" this line.
# Building against 12.8 is fine on the 595.84 driver: CUDA is backward compatible, so the older
# toolkit costs nothing at run time.
#
# Build:  sg docker -c "docker build -t astra-moshi:0.6.4 -f services/moshi/Dockerfile services/moshi"
# Run:    services/stt.sh

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
# python3-dev: moshi-server pulls pyo3, whose build script fails with "no Python 3.x interpreter
# found" on the bare CUDA image. It needs the headers, not just the interpreter.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config libssl-dev cmake curl ca-certificates git \
        libopus-dev python3 python3-dev \
    && rm -rf /var/lib/apt/lists/*

# rustup rather than the distro toolchain: candle's CUDA backend tracks recent stable Rust.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# sm_120 (RTX 5090). Without this candle builds PTX for the default arch list and the kernels are
# JIT-recompiled at every start, or fail outright on Blackwell.
ENV CUDA_COMPUTE_CAP=120

# Version pinned in services/versions.lock [speech] moshi_server. Changing it is its own task.
RUN cargo install --locked --features cuda moshi-server@0.6.4

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
# libpython3.12t64 provides libpython3.12.so.1.0, which pyo3 links against. The `python3` package
# alone is NOT enough — it installs the interpreter but not the shared library, and the binary
# then dies at startup with "error while loading shared libraries". The t64 suffix is Ubuntu
# 24.04's time_t transition, not a typo.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopus0 ca-certificates libpython3.12t64 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.cargo/bin/moshi-server /usr/local/bin/moshi-server

# Weights come from the HF cache mounted at run time; no network egress from this container.
ENV HF_HOME=/root/.cache/huggingface
EXPOSE 8080
ENTRYPOINT ["moshi-server"]
