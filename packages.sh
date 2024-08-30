#!/bin/csh
/usr/local/bin/pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
/usr/local/bin/pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
echo "os.environ['PATH'] += ':/opt/conda/lib'" >> /usr/local/etc/jupyter/jupyter_notebook_config.py
echo "os.environ['LD_LIBRARY_PATH'] += ':/opt/conda/lib'" >> /usr/local/etc/jupyter/jupyter_notebook_config.py
cat /usr/local/etc/jupyter/jupyter_notebook_config.py
git clone https://github.com/leofang/cupy.git --branch segmented_sort --recursive
cd cupy
/usr/local/bin/pip install .
