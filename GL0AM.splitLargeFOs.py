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
#An extra step to limit the max FO of nodes in the graph, due to a drawback in the simulation code that uses a for loop for all the fanouts of a gate
#Future work should solve this issue.

parser = argparse.ArgumentParser()
parser.add_argument('--combo_graph', type=str, help = 'the original DGL translated combinational graph')
parser.add_argument('--max_FO_constraint', type=int, default=32, help = "max FO constraint for combo graph during 0 delay sim")
parser.add_argument('--target_combo_graph', type=str, help = 'the max FO constraint altered combinational graph save file')
args = parser.parse_args()

fileObject = open(args.combo_graph, 'rb') ; bg = pickle.load(fileObject)

topo_nodes_cpu =  dgl.traversal.topological_nodes_generator(bg)
start = timer()


def split_FOs(violated):
  global bg
  num_nodes_to_add_per_violated = th.ceil(bg.out_degrees(violated)/max_FO).type(th.int32) -1
  total_nodes_to_add = th.sum(num_nodes_to_add_per_violated)
  orig_cell_types = bg.ndata['cell_type'][violated]
  add_node_data = th.repeat_interleave(orig_cell_types, num_nodes_to_add_per_violated)
  new_node_numbers = th.cuda.LongTensor([len(bg.nodes()) + x for x in range(total_nodes_to_add)])
  in_degrees_of_violated = bg.in_degrees(violated)
  in_edge_repeats_foreach_violated = th.repeat_interleave(in_degrees_of_violated, num_nodes_to_add_per_violated)
  in_edge_receivers = th.repeat_interleave(new_node_numbers,in_edge_repeats_foreach_violated)
  in_edge_sources = [] ; in_edge_xs = [] ; in_edge_net_delay_rise = []; in_edge_net_delay_fall = [];
  in_edge_start_pointers = [];   in_edge_end_pointers = [];   
  for i in range(len(violated)):
    this_node = violated[i];
    this_sources = bg.in_edges(this_node)[0]
    this_xs = bg.edata['x'][bg.edge_ids(bg.in_edges(this_node)[0],bg.in_edges(this_node)[1])]
    this_net_delay_rises = bg.edata['net_delay_rise'][bg.edge_ids(bg.in_edges(this_node)[0],bg.in_edges(this_node)[1])]
    this_net_delay_falls = bg.edata['net_delay_fall'][bg.edge_ids(bg.in_edges(this_node)[0],bg.in_edges(this_node)[1])]
    this_start_pointers = bg.edata['start_pointers'][bg.edge_ids(bg.in_edges(this_node)[0],bg.in_edges(this_node)[1])]
    this_end_pointers = bg.edata['end_pointers'][bg.edge_ids(bg.in_edges(this_node)[0],bg.in_edges(this_node)[1])]
    repeats = num_nodes_to_add_per_violated[i]
    this_sources = this_sources.repeat(repeats) ; this_xs = this_xs.repeat(repeats) ; 
    this_net_delay_rises = this_net_delay_rises.repeat(repeats) ; this_net_delay_falls = this_net_delay_falls.repeat(repeats)
    this_start_pointers = this_start_pointers.repeat(repeats) ; this_end_pointers = this_end_pointers.repeat(repeats)
    in_edge_sources.append(this_sources) ; in_edge_xs.append(this_xs)
    in_edge_net_delay_rise.append(this_net_delay_rises) ; in_edge_net_delay_fall.append(this_net_delay_falls)
    in_edge_start_pointers.append(this_start_pointers) ; in_edge_end_pointers.append(this_end_pointers)
  in_edge_sources=th.cat(in_edge_sources) ; in_edge_xs = th.cat(in_edge_xs)
  in_edge_net_delay_rise=th.cat(in_edge_net_delay_rise) ; in_edge_net_delay_fall = th.cat(in_edge_net_delay_fall)
  in_edge_start_pointers=th.cat(in_edge_start_pointers) ; in_edge_end_pointers = th.cat(in_edge_end_pointers)
  out_edge_receivers = [] ; out_edge_xs = []
  out_edge_net_delay_rise = []; out_edge_net_delay_fall = [];
  out_edge_start_pointers = []; out_edge_end_pointers = [];  
  for i in range(len(violated)):
    this_node = violated[i]
    this_outs = bg.out_edges(this_node)[1]
    this_xs = bg.edata['x'][bg.edge_ids(bg.out_edges(this_node)[0],bg.out_edges(this_node)[1])]
    this_net_delay_rises = bg.edata['net_delay_rise'][bg.edge_ids(bg.out_edges(this_node)[0],bg.out_edges(this_node)[1])]
    this_net_delay_falls = bg.edata['net_delay_fall'][bg.edge_ids(bg.out_edges(this_node)[0],bg.out_edges(this_node)[1])]
    this_start_pointers = bg.edata['start_pointers'][bg.edge_ids(bg.out_edges(this_node)[0],bg.out_edges(this_node)[1])]
    this_end_pointers = bg.edata['end_pointers'][bg.edge_ids(bg.out_edges(this_node)[0],bg.out_edges(this_node)[1])]
    total_outs = bg.out_degrees(this_node)
    cutoff_index = th.floor(th.Tensor([total_outs])/max_FO).type(th.int32) * max_FO
    cutoff_index = cutoff_index - max_FO if total_outs%max_FO==0 else cutoff_index
    this_outs = this_outs[0:cutoff_index] ; this_xs = this_xs[0:cutoff_index]
    this_net_delay_rises = this_net_delay_rises[0:cutoff_index] ; this_net_delay_falls = this_net_delay_falls[0:cutoff_index]
    this_start_pointers = this_start_pointers[0:cutoff_index] ; this_end_pointers = this_end_pointers[0:cutoff_index]
    out_edge_receivers.append(this_outs) ; out_edge_xs.append(this_xs)
    out_edge_net_delay_rise.append(this_net_delay_rises) ; out_edge_net_delay_fall.append(this_net_delay_falls)
    out_edge_start_pointers.append(this_start_pointers) ; out_edge_end_pointers.append(this_end_pointers)
  out_edge_receivers=th.cat(out_edge_receivers) ; out_edge_xs = th.cat(out_edge_xs)
  out_edge_net_delay_rise =th.cat(out_edge_net_delay_rise) ; out_edge_net_delay_fall = th.cat(out_edge_net_delay_fall)
  out_edge_start_pointers = th.cat(out_edge_start_pointers) ; out_edge_end_pointers = th.cat(out_edge_end_pointers)
  out_edge_sources = th.repeat_interleave(new_node_numbers,max_FO)
  out_edge_source_deletes = th.repeat_interleave(violated,num_nodes_to_add_per_violated)
  out_edge_source_deletes = th.repeat_interleave(out_edge_source_deletes, max_FO)
  bg = dgl.add_nodes(bg, int(total_nodes_to_add),data = {'cell_type': add_node_data})
  bg.add_edges(in_edge_sources,in_edge_receivers,data = {'x': in_edge_xs, 'net_delay_rise' : in_edge_net_delay_rise, 'net_delay_fall' : in_edge_net_delay_fall, 'start_pointers' : in_edge_start_pointers, 'end_pointers' : in_edge_end_pointers})
  bg.add_edges(out_edge_sources,out_edge_receivers,data={'x': out_edge_xs, 'net_delay_rise' : out_edge_net_delay_rise, 'net_delay_fall' : out_edge_net_delay_fall, 'start_pointers' : out_edge_start_pointers, 'end_pointers' : out_edge_end_pointers})
  bg = dgl.remove_edges(bg,bg.edge_ids(out_edge_source_deletes,out_edge_receivers))


max_FO=args.max_FO_constraint
device="cuda:0"
bg=bg.to(device)
total_logic_levels = len(topo_nodes_cpu)
for i in range(total_logic_levels-1,0,-1):
  these_nodes = topo_nodes_cpu[i].to(device)
  num_of_FOs = bg.out_degrees(these_nodes)
  violated = these_nodes[th.where(num_of_FOs > max_FO)[0]]
  if violated.size()[0] > 0:
    split_FOs(violated)
    topo_nodes_cpu = dgl.traversal.topological_nodes_generator(bg)

fileObject = open(args.target_combo_graph, 'wb')
pickle.dump(bg, fileObject)
