#!/bin/csh
##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/GL0AM/blob/main/LICENSE
#
##############################################################################

pip install scipy
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
echo "os.environ['PATH'] += ':/opt/conda/lib'" >> /usr/local/etc/jupyter/jupyter_notebook_config.py
echo "os.environ['LD_LIBRARY_PATH'] += ':/opt/conda/lib'" >> /usr/local/etc/jupyter/jupyter_notebook_config.py
cat /usr/local/etc/jupyter/jupyter_notebook_config.py
git clone https://github.com/leofang/cupy.git --branch segmented_sort --recursive
cd cupy
/usr/local/bin/pip install .
