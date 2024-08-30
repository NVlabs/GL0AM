import torch as th
import dgl
from dgl import DGLGraph
import pickle
import numpy as np
import networkx as nx
import sys
import argparse
import glob, os
import re
from datetime import datetime
import math
import multiprocessing as pmp
from timeit import default_timer as timer
import cupy as cp
import gzip
import cupyx

exec(open('GL0AM.cupy').read())

parser = argparse.ArgumentParser()
parser.add_argument('--block', type=str)
parser.add_argument('--graph', type=str, help = 'the DGL translated graph for combo gates')
parser.add_argument('--graph2', type=str, help = 'the DGL translated graph for registers')
parser.add_argument('--graph3', type=str, help='the DGL translated graph for srams')
parser.add_argument('--graph4', type=str, help='the max FO constraint processed combo DGL graph')
parser.add_argument('--hMetisResults', type=str, help='hMetis partitioning results')
parser.add_argument('--partitions', type=int, help='hMetis partitions parameter')
parser.add_argument('--graphPrecompile', type=str, help='graph precompile file for port names')
args = parser.parse_args()

threads_per_block=512

# +
#load the important input files
fileObject = open(args.graph, 'rb') ; bg = pickle.load(fileObject)
fileObject = open(args.graph2, 'rb') ; bg2 = pickle.load(fileObject)
fileObject = open(args.graph3, 'rb') ; bg3 = pickle.load(fileObject)

f = gzip.GzipFile(args.graphPrecompile, "r"); data = np.load(f, allow_pickle=1) ; f.close()
num_of_std_cells=data['num_of_std_cells']
port_names=data['port_index'] 
port_names=dict((k, port_names[k]) for k in range(num_of_std_cells, len(bg.nodes())))
del data


start_compile=timer()
#graph stuff starts here...setup the simulation graph
device = "cuda:0"
fileObject = open(args.graph4, 'rb') ; bg = pickle.load(fileObject)
bg=bg.to("cpu") ; bg2=bg2.to("cpu")

topo_nodes_cpu =  dgl.traversal.topological_nodes_generator(bg)

fromEndpointsToSrc = dgl.topological_nodes_generator(bg, reverse=True)
dummyNodeNum = len(bg.nodes()) + 1000

endpointNodes = cp.asarray(fromEndpointsToSrc[0]).astype(cp.int32)
numHypergraphNodes = endpointNodes.shape[0]
hypergraphVertexID = cp.full( int(endpointNodes.max()+1), dummyNodeNum, dtype = cp.int32)
setIndex_T( (math.ceil(endpointNodes.shape[0]/512),1), (512,1),\
 (endpointNodes,hypergraphVertexID,endpointNodes.shape[0]) )

hMetisFile = args.hMetisResults
f = open(hMetisFile, 'r') ; hMetisResults = []
for line in f:
  hMetisResults.append(int(line.strip()))
hMetisResults = th.cuda.IntTensor(hMetisResults)
f.close()
partitionIDs, counts = th.unique(hMetisResults, return_counts=True)
dummyTensor, hMetisNodeNums = hMetisResults.sort()
assert th.all(partitionIDs == th.cuda.IntTensor([x for x in range(partitionIDs.max()+1)])), "Malformed partitions from Hmetis"
partitionOffsets = th.roll(th.cumsum(counts, dtype=th.int64, dim = 0), 1, 0)
totalHypergraphNodes = partitionOffsets[0].repeat(1)
partitionOffsets[0] = 0
dglNodeNums = fromEndpointsToSrc[0].to(device)[hMetisNodeNums].to("cpu")

partitionedNetlist =[]
for i in range(partitionIDs.size()[0]):
  baseOrigNodes = dglNodeNums[partitionOffsets[i]:partitionOffsets[i] + counts[i]]
  sub0Nodes = dgl.bfs_nodes_generator(bg,baseOrigNodes,reverse=True) ; allSub0Nodes = th.cat(sub0Nodes)
  sub0 = dgl.node_subgraph(bg,allSub0Nodes,relabel_nodes=True,store_ids=True,output_device="cpu")
  partitionedNetlist.append(sub0)


# +
origGraphNumNodes = len(bg.nodes()) ; nodesPerPartition =[] ; stagesPerPartition =[]
postPartitioningTotalNumNodes = 0
for part in partitionedNetlist:
  topo = dgl.traversal.topological_nodes_generator(part) ; stagesPerPartition.append(len(topo))
  partitionNumNodes = len(part.nodes()) ;  nodesPerPartition.append(partitionNumNodes)
  postPartitioningTotalNumNodes += partitionNumNodes
logicDupFactor = postPartitioningTotalNumNodes/origGraphNumNodes
print("Logic Duplication Factor: " + str(logicDupFactor))

numParts = len(partitionedNetlist)
nodesPerPartition = th.cuda.FloatTensor(nodesPerPartition)
stagesPerPartition = th.cuda.FloatTensor(stagesPerPartition)
stdDev = th.std(nodesPerPartition)
print("std deviation of number of Nodes in Partitions: " + str(float(stdDev)))
print("range/average of number of Nodes in Partitions: " + str(int(nodesPerPartition.min())) + ' -- '  + str(int(nodesPerPartition.mean())) + \
 ' -- ' + str(int(nodesPerPartition.max())))
stdDev2 = th.std(stagesPerPartition)
print("std deviation of number of Stages in Partitions: " + str(float(stdDev2)))
print("range/average of number of Stages in Partitions: " + str(int(stagesPerPartition.min())) + ' -- '  + str(int(stagesPerPartition.mean())) + \
 ' -- ' + str(int(stagesPerPartition.max())))
# -

batchGraph = dgl.batch(partitionedNetlist)
assert th.all(th.unique(batchGraph.edata['_ID']) == th.unique(bg.edge_ids(bg.edges()[0],bg.edges()[1]))),\
 'Malformed partitions, missing some edge connections! Simulation will be wrong!'
assert th.all(th.unique(batchGraph.ndata['_ID']) == th.LongTensor([x for x in range(len(bg.nodes()))])),\
 'Malformed partitions, missing some nodes! Simulation will be wrong!'

#need to add some endpoint translation info
timing_endpoints=[] ; list_of_inputs=[]
for key in port_names:
  this_key = int(key)
  if bg.in_degrees(this_key) > 0:
    timing_endpoints.append(this_key)
  else:
    list_of_inputs.append(this_key)
topo_nodes2_cpu = dgl.traversal.topological_nodes_generator(bg2)
topo_nodes3_cpu = dgl.traversal.topological_nodes_generator(bg3)
tensor_timing_endpoints=th.LongTensor(timing_endpoints)
if len(topo_nodes3_cpu) > 1:
  isolated_nodes = ( (bg2.in_degrees() == 0) & (bg2.out_degrees() == 0) & (bg3.in_degrees() == 0) & (bg3.out_degrees() == 0)).nonzero().squeeze(1)
else:
  isolated_nodes = ( (bg2.in_degrees() == 0) & (bg2.out_degrees() == 0)).nonzero().squeeze(1)
isolated_nodes2 = th.where((bg.in_degrees() == 0) & (bg.out_degrees() == 0))[0]
isolated_nodes = th.cat( (isolated_nodes, isolated_nodes2))
potential_timing_endpoints2 = topo_nodes2_cpu[0]
potential_timing_endpoints3 = topo_nodes3_cpu[0]
if len(topo_nodes3_cpu) > 1:
  potential_timing_endpoints = th.unique(th.cat((potential_timing_endpoints2,potential_timing_endpoints3)))
else:
  potential_timing_endpoints = potential_timing_endpoints2
bg.ndata['inputs']=th.zeros(len(bg.nodes()),dtype=th.bool,device="cpu")
bg.ndata['isolated_nodes']=th.zeros_like(bg.ndata['inputs'])
bg.ndata['potential_timing_endpoints']=th.zeros_like(bg.ndata['inputs'])
bg.ndata['inputs'][list_of_inputs]=1
bg.ndata['isolated_nodes'][isolated_nodes]=1
bg.ndata['potential_timing_endpoints'][potential_timing_endpoints]=1
endpoints =(bg.ndata['potential_timing_endpoints'] & (~bg.ndata['isolated_nodes']) & (~bg.ndata['inputs'])).nonzero().squeeze(1)
endpoints=th.cat(( endpoints,tensor_timing_endpoints))
bg.ndata['is_endpoint']=th.zeros_like(bg.ndata['inputs'])
bg.ndata['is_endpoint'][endpoints]=1

partitionSizes = []
for i in range(len(partitionedNetlist)):
  partitionSizes.append(len(partitionedNetlist[i].nodes()))
partitionSizes = th.IntTensor(partitionSizes)
partitionSizesCumsum = th.roll( th.cumsum(partitionSizes, 0), 1, 0 )
partitionSizesCumsumTotal = partitionSizesCumsum[0].repeat(1)
partitionSizesCumsum[0] = 0
partitionSizesCumsum = th.cat( (partitionSizesCumsum,partitionSizesCumsumTotal) )
fromEndpointsToSrcBatched = dgl.topological_nodes_generator(batchGraph, reverse=True)
batchGraphPotentialEndpoints = fromEndpointsToSrcBatched[0]
batchGraphPotentialEndpoints=batchGraphPotentialEndpoints[th.where( ~((batchGraph.in_degrees(batchGraphPotentialEndpoints) == 0) & (batchGraph.out_degrees(batchGraphPotentialEndpoints) == 0)) )[0]]
batchGraph.ndata['mark_new_endpoints'] = th.zeros(len(batchGraph.nodes()), dtype=th.bool)
batchGraph.ndata['mark_new_endpoints'][batchGraphPotentialEndpoints] = 1
endpointsTakenCareOf = batchGraph.ndata['_ID'][batchGraphPotentialEndpoints]
remainingEndpoints=th.LongTensor(list(set(np.array(endpoints)).difference(set(np.array(endpointsTakenCareOf)))))
origIDs, origIDIndexes = batchGraph.ndata['_ID'].sort()
origIDs, origIDCounts = th.unique(batchGraph.ndata['_ID'], return_counts=True)
origIDCountsCumsum = th.roll( th.cumsum(origIDCounts, 0), 1, 0 )
origIDCountsCumsum[0] = 0
batchGraph.ndata['mark_new_endpoints'][origIDIndexes[origIDCountsCumsum[remainingEndpoints]]] = 1
for i in range(len(partitionedNetlist)):
  partitionedNetlist[i].ndata['mark_new_endpoints'] = th.zeros(len(partitionedNetlist[i].nodes()), dtype=th.bool)
  partitionedNetlist[i].ndata['mark_new_endpoints'] = batchGraph.ndata['mark_new_endpoints'][partitionSizesCumsum[i]:partitionSizesCumsum[i+1]]

totalAnnotatedEndpoints = 0
for i in range(len(partitionedNetlist)):
  totalAnnotatedEndpoints += partitionedNetlist[i].ndata['mark_new_endpoints'].sum()
assert totalAnnotatedEndpoints >= endpoints.size()[0], "Needs debug! Not all combo-reg interface points annotated!"

pickle.dump(partitionedNetlist,open(args.block + '.' + str(args.partitions) + ".partitions",'wb'))


 
