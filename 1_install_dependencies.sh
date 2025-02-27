#!/bin/bash
set -e

echo "Step 1: 安装 Python 3.6.5"
conda create --prefix /path/to/env python=3.6.5 -y

echo "Step 2: 安装 mamba"
source activate /path/to/env
conda install mamba -c conda-forge -y

echo "Step 3: 安装 torch 及 cudatoolkit"
pip install torch==1.10.1+cu113 torchaudio==0.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html

echo "Step 4: 安装其他包"
mamba install coloredlogs -y
mamba install dataclasses -y
pip install "librosa>=0.8.0"
mamba install "numpy>=1.18.3" -y
mamba install "pandas>=1.0.3" -y
pip install pocketsphinx
pip install "praat-textgrids==1.3.1"
pip install "protobuf==3.19.4"
mamba install pyaudio -y
mamba install "pydantic>=1.7.3" -y
mamba install "pytest==6.2.3" -y
pip install "pytest-timeout==2.1.0"
pip install "soundfile==0.10.3.post1"
pip install "tensorboard==2.8.0"
pip install tqdm

echo "Step 5: 安装训练模型所需的额外环境"
mamba install openpyxl -y
pip install pre-commit
pip install "webrtcvad==2.0.10"

echo "所有依赖项已成功安装"