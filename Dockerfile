FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN pip  uninstall -y dgl
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu118/repo.html
RUN pip install cupy-cuda11x

WORKDIR /app
