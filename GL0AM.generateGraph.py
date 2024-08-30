import torch as th
import dgl
from dgl import DGLGraph
import numpy as np
import networkx as nx
import sys
import argparse
import pickle
import glob, os
import re
import math
import multiprocessing as pmp
from datetime import datetime
import gzip

parser = argparse.ArgumentParser()
parser.add_argument('--block', type=str)
parser.add_argument('--graphPrecompile', type=str)
args = parser.parse_args()
#args = parser.parse_args(['--block', 'qadd_pipe1000', '--graphPrecompile', 'qadd_pipe1000_GraphPrecompile.pkl.gz'])
#data loading, builds the DGL graph from lil_matrix graph
def build_graph(pkl):
  now=datetime.now()
  f = gzip.GzipFile(args.graphPrecompile, "r"); data = np.load(f, allow_pickle=1) ; f.close()
  PinOrder=data['adjacency_pin_ids']
  EdgeTypes=data['adjacency_edge_types'] ; 
  EdgeTypes2 = (EdgeTypes==2) + (EdgeTypes==3) ;   EdgeTypes3 = (EdgeTypes==5) ; EdgeTypes = (EdgeTypes==1) ; 
  net_delays_rise = data['SDF_net_rise'] ; net_delays_fall = data['SDF_net_fall'] ; 
  seq_pushoutSDF = data['seq_pushoutSDF']
  print("pkl loaded")
  print('start create update (register) graph')
  g_full = dgl.from_scipy(EdgeTypes)
  g_full.edata['x'] = th.IntTensor(PinOrder[tuple(g_full.edges()[0]),tuple(g_full.edges()[1])]).squeeze(0)
  #"global variables"
  cell_names=data['cell_index']
  num_of_std_cells=data['num_of_std_cells']
  cell_names=dict((k, cell_names[k]) for k in range(num_of_std_cells))
  port_names=data['port_index'] 
  port_names=dict((k, port_names[k]) for k in range(num_of_std_cells, EdgeTypes.shape[0]))
  cell_names.update(port_names)
  cell_types=data['cell_ids'] ; cell_types=th.LongTensor(cell_types) ; 
  g_full.ndata['cell_type'] = cell_types
  print("graph init done")
  g_update = dgl.from_scipy(EdgeTypes2)
  g_update.edata['x'] = th.IntTensor(PinOrder[tuple(g_update.edges()[0]),tuple(g_update.edges()[1])]).squeeze(0)
  g_update.ndata['seq_pushoutSDF'] = th.Tensor(seq_pushoutSDF)
  tie_node_nums = data['edge_ties_node_nums'] ; tie_pin_positions = data['edge_ties_pin_positions']
  sorted_node_nums, indexes = th.LongTensor(tie_node_nums).sort()
  tied_pin_positions = th.CharTensor(tie_pin_positions)[indexes]
  tie_info_tensor = {}; tie_info_tensor['node_nums'] = sorted_node_nums ; tie_info_tensor['pin_positions'] = tied_pin_positions;
  #start clkTree stuff
  clkTreeNodes = th.where(th.logical_and(g_update.in_degrees() > 0, g_update.out_degrees() > 0))[0]
  reconstructedBufsInvs = clkTreeNodes[th.where(th.logical_or(g_full.ndata['cell_type'][clkTreeNodes] == 24 , g_full.ndata['cell_type'][clkTreeNodes] == 21))[0]]
  reconstructedEdgeSrcs, reconstructedEdgeDsts = g_update.in_edges(reconstructedBufsInvs)
  edge_xs = g_update.edata['x'][g_update.edge_ids(reconstructedEdgeSrcs,reconstructedEdgeDsts)]
  g_full.add_edges(reconstructedEdgeSrcs,reconstructedEdgeDsts,data={'x': edge_xs})
  tree = dgl.traversal.topological_nodes_generator(g_update)
  g_update.ndata['polarityToSrc'] = th.zeros(len(g_update.nodes()), dtype=th.int8)
  adjustedSAIFClkTreeNodes = [] ;
  for layer in tree:
    bufInvLayerNodes = layer[th.where(th.logical_or(g_full.ndata['cell_type'][layer] == 24, g_full.ndata['cell_type'][layer] == 21))[0]]
    bufInvLayerNodes = bufInvLayerNodes[th.where(g_update.in_degrees(bufInvLayerNodes) > 0)[0]]
    #calculate the current layer polarities.
    allParents, allLayerNodes = g_update.in_edges(layer)
    temp = g_full.ndata['cell_type'][allLayerNodes] == 24
    g_update.ndata['polarityToSrc'][allLayerNodes] = (g_update.ndata['polarityToSrc'][allLayerNodes] + temp) % 2
    #propagate updated polarities to all current layer fanouts
    currentDrivers, currentFanouts = g_update.out_edges(allLayerNodes)
    g_update.ndata['polarityToSrc'][currentFanouts] = g_update.ndata['polarityToSrc'][currentDrivers]
    #start edge deletion.
    newEdgeSrcNodes = g_update.in_edges(bufInvLayerNodes)[0]
    newEdgeDstNodes = g_update.out_edges(bufInvLayerNodes)[1]
    newEdgeSrcNodes = th.repeat_interleave(newEdgeSrcNodes, g_update.out_degrees(bufInvLayerNodes))
    out_edge_xs = g_update.edata['x'][g_update.edge_ids(g_update.out_edges(bufInvLayerNodes)[0],g_update.out_edges(bufInvLayerNodes)[1])]
    g_update.add_edges(newEdgeSrcNodes,newEdgeDstNodes,data={'x': out_edge_xs})
    #insert troublesome invs/buffs here to add to list.
    adjustedSAIFClkTreeNodes.append(bufInvLayerNodes[th.where(th.logical_and( g_full.ndata['cell_type'][newEdgeSrcNodes] >= 300, g_full.ndata['cell_type'][newEdgeSrcNodes] < 400))[0]])
    g_update = dgl.remove_edges(g_update,g_update.edge_ids(g_update.in_edges(bufInvLayerNodes)[0],g_update.in_edges(bufInvLayerNodes)[1]))
    g_update = dgl.remove_edges(g_update,g_update.edge_ids(g_update.out_edges(bufInvLayerNodes)[0],g_update.out_edges(bufInvLayerNodes)[1]))
    #flip the celltypes for the clk gate element outputs of inverters
    clkGates = layer[th.where(th.logical_and( g_full.ndata['cell_type'][layer] >= 300, g_full.ndata['cell_type'][layer] < 400))[0]]
    #insert clk gates to list here.
    adjustedSAIFClkTreeNodes.append(clkGates)
    allClkGateInputEdgeSrcs, allClkGateInputEdgeDsts = g_update.in_edges(clkGates)
    allClkGateClockInputs = allClkGateInputEdgeSrcs[th.where(g_update.edata['x'][g_update.edge_ids(allClkGateInputEdgeSrcs, allClkGateInputEdgeDsts)] == 7)[0]]
    allClkGatesInThisLevel = allClkGateInputEdgeDsts[th.where(g_update.edata['x'][g_update.edge_ids(allClkGateInputEdgeSrcs, allClkGateInputEdgeDsts)] == 7)[0]]
    reverseOutputPolarityClkGates = allClkGatesInThisLevel[th.where(g_update.ndata['polarityToSrc'][allClkGateClockInputs] != g_update.ndata['polarityToSrc'][allClkGatesInThisLevel])[0]]
    g_full.ndata['cell_type'][reverseOutputPolarityClkGates] = th.bitwise_xor(g_full.ndata['cell_type'][reverseOutputPolarityClkGates], 2)
  adjustedSAIFClkTreeNodes = th.cat(adjustedSAIFClkTreeNodes) ; adjustedSAIFClkTreeNodePolarities=g_update.ndata['polarityToSrc'][adjustedSAIFClkTreeNodes]
  print('seq and clkTree graph done...')
  #create the combinational graph
  g_full.edata['net_delay_rise'] = th.Tensor(net_delays_rise[tuple(g_full.edges()[0]),tuple(g_full.edges()[1])]).squeeze(0)
  g_full.edata['net_delay_fall'] = th.Tensor(net_delays_fall[tuple(g_full.edges()[0]),tuple(g_full.edges()[1])]).squeeze(0)
  rownums = data['pin4SDF_rownum'] ; xCondDelays = data['condXSDF_LUT'];
  rowattribute = th.LongTensor(rownums[tuple(g_full.edges()[0]),tuple(g_full.edges()[1])]).squeeze(0)
  pin4_data = data['fullSDF_LUT']
  pin4_data = np.array(pin4_data, dtype=object)
  cell_delay_lengths = th.IntTensor( [len(pin4_data[x]) for x in range(len(pin4_data)) ] )
  fullSDF_delays = th.FloatTensor([item for sublist in data['fullSDF_LUT'] for item in sublist]) ; xCondDelays = th.FloatTensor(np.array(xCondDelays))
  end_pointers=th.cumsum(cell_delay_lengths.type(th.IntTensor), dim=0, dtype=th.int32)
  begin_pointers = th.cat( (th.IntTensor([0]), end_pointers) )
  g_full.edata['start_pointers']=begin_pointers[rowattribute.type(th.LongTensor)].type(th.int32)
  g_full.edata['end_pointers']=end_pointers[rowattribute.type(th.LongTensor)].type(th.int32)
  g_full.edata['XCond_pointers']=rowattribute.type(th.int32)
  print("graph created")
  print('start create sram graph')
  g_sram = dgl.from_scipy(EdgeTypes3)
  if g_sram.edges()[0].size()[0] > 0 :
    g_sram.edata['x'] = th.IntTensor(PinOrder[tuple(g_sram.edges()[0]),tuple(g_sram.edges()[1])]).squeeze(0)
  later=datetime.now()
  delta=(later-now).total_seconds()
  print("loading the graph took " + str(delta) + " seconds on the CPU")
  #the fullSDF delays are stored in a 1D Tensor, while the graph node attributes for SDF cell delays just have a pointer value that tells where in the 1D Tensor the SDF delays for the node/cell start and end
  return g_update, g_full, fullSDF_delays, xCondDelays, tie_info_tensor, g_sram, adjustedSAIFClkTreeNodes, adjustedSAIFClkTreeNodePolarities


update_full, prop_full, fullSDF_delays, xCondDelays, tie_info, sram_full, adjustedSAIFClkTreeNodes,adjustedSAIFClkTreeNodePolarities= build_graph(args.graphPrecompile)
fileObject = open(args.block + "_fullSDF_DGLGraph", 'wb')
pickle.dump(prop_full, fileObject)
fileObject = open(args.block + "_update_DGLGraph", 'wb')
pickle.dump(update_full, fileObject)
fileObject = open(args.block + "_sram_DGLGraph", 'wb')
pickle.dump(sram_full, fileObject)
th.save(fullSDF_delays, args.block + "_fullSDF_delays")
th.save(xCondDelays, args.block + "_XCondSDF_delays")
th.save(adjustedSAIFClkTreeNodes, args.block + "_adjustedSAIFClkTreeNodes")
th.save(adjustedSAIFClkTreeNodePolarities, args.block + "_adjustedSAIFClkTreeNodePolarities")
fileObject = open(args.block + "_tie_info", 'wb')
pickle.dump(tie_info, fileObject)
 
