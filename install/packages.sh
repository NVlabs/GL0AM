#!/bin/csh
##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the "NVIDIA License".
# To view a copy of this license, visit
# https://github.com/NVlabs/GL0AM/blob/main/LICENSE
#
##############################################################################

pip install scipy
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
cfg=/usr/local/etc/jupyter/jupyter_notebook_config.py
grep -qxF "os.environ['PATH'] += ':/opt/conda/lib'" $cfg || printf "%s\n" "os.environ['PATH'] += ':/opt/conda/lib'" >> $cfg
grep -qxF "os.environ['LD_LIBRARY_PATH'] += ':/opt/conda/lib'" $cfg || printf "%s\n" "os.environ['LD_LIBRARY_PATH'] += ':/opt/conda/lib'" >> $cfg
cat /usr/local/etc/jupyter/jupyter_notebook_config.py
git clone https://github.com/leofang/cupy.git --branch segmented_sort --recursive
cd cupy || exit 1
/usr/local/bin/pip install .
