# vast.ai 用 shogi-lab base image
#
# 方針: 課金前 (イメージ pull = loading 段階) に済ませられるものは全てここに焼き込む。
#   - CUDA 12.9 devel + cuDNN 9 (base image)
#   - TensorRT 10.11 / ONNX Runtime 1.24.2 GPU  (rshogi rescore_psv 用; crates/tools/docs/rescore_psv.md 準拠)
#   - LLVM/clang 21 (tatara の cuda-oxide 用)
#   - Rust: stable 1.95.0 (rshogi pin) + nightly-2026-04-03 (tatara / shogitest pin) + rustfmt/clippy
#   - huggingface_hub[hf_transfer] (教師データ DL 用)
# private repo のコード・モデル・データは含めない (public registry に push 可能)。
# それらは onstart.sh が課金開始後すぐ自動実行する。
#
# build & push:
#   docker build -t <dockerhub_user>/shogi-lab:cuda129-trt1011 .
#   docker push <dockerhub_user>/shogi-lab:cuda129-trt1011

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# --- 基本パッケージ + SSH (vast の SSH 接続に必要) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        git tmux curl wget ca-certificates xz-utils \
        python3-pip python3-venv \
        build-essential pkg-config libssl-dev \
        openssh-server htop lsb-release software-properties-common gnupg \
    && mkdir -p /run/sshd \
    && rm -rf /var/lib/apt/lists/*

# --- LLVM/clang 21 (tatara cuda-oxide 要件: LLVM 21+, clang-21) ---
RUN wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh \
    && chmod +x /tmp/llvm.sh && /tmp/llvm.sh 21 \
    && apt-get update && apt-get install -y --no-install-recommends \
        clang-21 libclang-common-21-dev llvm-21-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/llvm.sh

# --- ONNX Runtime 1.24.2 GPU (ort クレート 2.0.0-rc.12 対応版; バージョン厳守) ---
RUN wget -q -O /tmp/ort.tgz \
        https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-gpu-1.24.2.tgz \
    && tar xzf /tmp/ort.tgz -C /opt && rm /tmp/ort.tgz

# --- TensorRT 10.11 (rescore_psv --onnx-tensorrt 用; ORT 1.24.2 は libnvinfer.so.10 を要求) ---
RUN wget -q -O /tmp/trt.tgz \
        https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.11.0/tars/TensorRT-10.11.0.33.Linux.x86_64-gnu.cuda-12.9.tar.gz \
    && tar xzf /tmp/trt.tgz -C /opt && rm /tmp/trt.tgz \
    # 使わない大物 (サンプル/静的lib/python wheel 以外の余剰) を削ってイメージを軽くする
    && rm -rf /opt/TensorRT-10.11.0.33/data /opt/TensorRT-10.11.0.33/samples \
              /opt/TensorRT-10.11.0.33/lib/*static*.a

# --- Rust toolchains (rshogi: stable 1.95.0 / tatara・shogitest: nightly-2026-04-03) ---
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain 1.95.0 --component rustfmt clippy \
    && /opt/cargo/bin/rustup toolchain install nightly-2026-04-03 \
        --component rustfmt clippy \
    && chmod -R a+rX /opt/rustup /opt/cargo

# --- HuggingFace CLI (教師データ DL; hf_transfer で高速化) ---
RUN pip3 install --no-cache-dir --break-system-packages "huggingface_hub[hf_transfer]"

# --- 環境変数 (rescore_psv.md §セットアップ準拠。cuDNN は base image 同梱の system 配置) ---
ENV ORT_DYLIB_PATH=/opt/onnxruntime-linux-x64-gpu-1.24.2/lib/libonnxruntime.so \
    LD_LIBRARY_PATH=/opt/TensorRT-10.11.0.33/lib:/opt/onnxruntime-linux-x64-gpu-1.24.2/lib:/usr/local/cuda/lib64 \
    PATH=/opt/cargo/bin:/usr/local/cuda/bin:$PATH \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    WORK=/workspace \
    SHOGI_DATA=/workspace/shogi-data

# 対話シェル (tmux / ssh) にも同じ env を通す
RUN printf 'export ORT_DYLIB_PATH=%s\nexport LD_LIBRARY_PATH=%s\nexport PATH=%s\nexport HF_HUB_ENABLE_HF_TRANSFER=1\nexport WORK=/workspace\nexport SHOGI_DATA=/workspace/shogi-data\nexport RUSTUP_HOME=/opt/rustup\nexport CARGO_HOME=/opt/cargo\n' \
        "$ORT_DYLIB_PATH" "$LD_LIBRARY_PATH" '/opt/cargo/bin:/usr/local/cuda/bin:$PATH' \
        > /etc/profile.d/shogi-lab.sh

WORKDIR /workspace
