import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
import pickle
import numpy as np
import networkx as nx
import sys
import argparse
import glob, os
import re
import gc
import math
from datetime import datetime
from timeit import default_timer as timer
import cupy as cp

from pathlib import Path
cd = Path.cwd()
exec(open(str(cd) + '/gatspi4mlcad2025_contest_sim/evalLogic.cupy').read())

#data loading, builds the DGL graph from csr raw graph
def load_graph(pkl):
 now=datetime.now()
 data = np.load(pkl, allow_pickle=1)
 print("pkl loaded")
 print('start create DGL graph')
 g = dgl.graph(('csr', (data['start'], data['items'], [])))
 g.edata['x'] = th.ByteTensor(data['values'])
 g.ndata['celltype'] = th.ShortTensor(data['gatspi_celltypes'])
 #"global variables"
 num_of_gatspi_cells = data['num_of_gatspi_cells'] ; num_of_top_ports = data['num_of_top_ports'] ;
 id2pinAndNet = data['gatspi_cellname_index']; id2port = data['gatspi_port_index'];
 tempPortDict = {key: (value,value) for key,value in id2port.items()}
 tempPinNetDict = {index+num_of_top_ports: value for index, value in enumerate(id2pinAndNet)}
 tempPortDict.update(tempPinNetDict)
 driverPin2id = {value[0]: key for key, value in tempPortDict.items()}
 net2id = {value[1]: key for key, value in tempPortDict.items()}   
 print("graph created")
 later=datetime.now()
 delta=(later-now).total_seconds()
 print("creating the DGL graph took " + str(delta) + " seconds on the CPU")
 return g, num_of_gatspi_cells, num_of_top_ports, tempPortDict, driverPin2id, net2id

def printATimer(functionDescription: str, temp_delta):
 print(functionDescription +  " done in " + f"{temp_delta:.3f}" + ' seconds')

class gatspiSimulateAndCompareTool:
 def __init__(self, topName, graph0FilePath, graph1FilePath, dumpDGLGraph=1, createStdCellLibLUT=1, cycles=50000, PARALLEL_CYCLES = 32, queryNetsListFile=''):
  self.topName = topName ; self.graph0FilePath=graph0FilePath; self.graph1FilePath=graph1FilePath; self.dumpDGLGraph=dumpDGLGraph; 
  self.createStdCellLibLUT = createStdCellLibLUT; self.cycles = cycles; self.PARALLEL_CYCLES = PARALLEL_CYCLES ; self.queryNetsListFile = queryNetsListFile;
  self.THREADS_PER_BLOCK = 512;
  self.g0 = None ; self.id2pinAndNet0 = None; self.driverPin2id0 = None; self.net2id0 = None;
  self.g1 = None ; self.id2pinAndNet1 = None; self.driverPin2id1 = None; self.net2id1 = None;
  self.loop_sg0 = None; self.oldLoopValues0 = None; self.topo_loop_cpu0 = None; self.loopParticipatingNodes0 = None;
  self.loop_sg1 = None; self.oldLoopValues1 = None; self.topo_loop_cpu1 = None; self.loopParticipatingNodes1 = None;
  self.topo_nodes_cpu0 = None; self.topo_nodes_cpu1 = None;
  self.inputNodes0 = None; self.outputs0 = None; self.outputsTotal0 = None;
  self.inputNodes1 = None; self.outputs1 = None; self.outputsTotal1 = None;
  self.cycles32 = math.ceil(self.cycles/self.PARALLEL_CYCLES) * self.PARALLEL_CYCLES
  self.simLoops = int(self.cycles32/self.PARALLEL_CYCLES) ; self.loopsMaxIter = None; 
  self.stdcell_array = None; self.stdcell_offsets = None;
  self.inputsTotal = None; self.currentLogicValue = None;
 
 def build_graphs(self):
  if self.dumpDGLGraph:
   print("creating the DGL graph from raw CSR graph...")
   temp_start = timer() ;
   self.g0, num_of_gatspi_cells0, num_of_top_ports0, self.id2pinAndNet0, self.driverPin2id0, self.net2id0 = load_graph(self.graph0FilePath)
   fileObject = open(self.topName + "_DGLGraph0", 'wb')
   pickle.dump({'g': self.g0,
    'num_of_gatspi_cells' : num_of_gatspi_cells0,
    'num_of_top_ports' : num_of_top_ports0,
    'id2pinAndNet' : self.id2pinAndNet0,
    'driverPin2id' : self.driverPin2id0,
    'net2id' : self.net2id0}, fileObject)
   self.g1, num_of_gatspi_cells1, num_of_top_ports1, self.id2pinAndNet1, self.driverPin2id1, self.net2id1 = load_graph(self.graph1FilePath)
   fileObject = open(self.topName + "_DGLGraph1", 'wb')
   pickle.dump({'g': self.g1,
    'num_of_gatspi_cells' : num_of_gatspi_cells1,
    'num_of_top_ports' : num_of_top_ports1,
    'id2pinAndNet' : self.id2pinAndNet1,
    'driverPin2id' : self.driverPin2id1,
    'net2id' : self.net2id1}, fileObject)
   temp_delta = timer() - temp_start
   printATimer("DGL graph", temp_delta)
  else:
   data = np.load(self.graph0FilePath, allow_pickle=True);
   self.g0 = data['g']; num_of_gatspi_cells0 = data['num_of_gatspi_cells'] ; num_of_top_ports0 = data['num_of_top_ports'] ; 
   self.id2pinAndNet0 = data['id2pinAndNet'] ; self.driverPin2id0 = data['driverPin2id'] ; self.net2id0 = data['net2id'] ;
   data = np.load(self.graph1FilePath, allow_pickle=True);
   self.g1 = data['g']; num_of_gatspi_cells1 = data['num_of_gatspi_cells'] ; num_of_top_ports1 = data['num_of_top_ports'] ; 
   self.id2pinAndNet1 = data['id2pinAndNet'] ; self.driverPin2id1 = data['driverPin2id'] ; self.net2id1 = data['net2id'] ;
  #prune the graph if we want only targeted nets to be compared.
  if self.queryNetsListFile != '':
   print("pruning graph to only nets in the query nets file...")
   fh = open(self.queryNetsListFile, 'r')
   lines = fh.readlines()
   lines = [line.rstrip('\n').strip() for line in lines]
   queryNets = []
   for net in lines:
    thisID = self.net2id0[net] ; queryNets.append(thisID);
   queryNets = th.LongTensor(queryNets)
   newg0, inverse_indices = dgl.khop_out_subgraph(self.g0, queryNets, k=3000) #may need to find a better solution to this other than hardcoding this to a large number (3000)
   tempOuts = newg0.ndata['_ID']
   newg0, inverse_indices = dgl.khop_in_subgraph(self.g0, tempOuts, k=3000)
   id2pinAndNet0_temp = { x: self.id2pinAndNet0[int(newg0.ndata['_ID'][x])] for x in range(len(newg0.nodes()))}
   self.id2pinAndNet0 = id2pinAndNet0_temp
   self.driverPin2id0 = {value[0]: key for key, value in self.id2pinAndNet0.items()}
   self.net2id0 = {value[1]: key for key, value in self.id2pinAndNet0.items()}
   self.g0 = newg0
   
   queryNets = []
   for net in lines:
    thisID = self.net2id1[net] ; queryNets.append(thisID);
   queryNets = th.LongTensor(queryNets)
   newg1, inverse_indices = dgl.khop_out_subgraph(self.g1, queryNets, k=3000)
   tempOuts = newg1.ndata['_ID']
   newg1, inverse_indices = dgl.khop_in_subgraph(self.g1, tempOuts, k=3000)
   id2pinAndNet1_temp = { x: self.id2pinAndNet1[int(newg1.ndata['_ID'][x])] for x in range(len(newg1.nodes()))}
   self.id2pinAndNet1 = id2pinAndNet1_temp
   self.driverPin2id1 = {value[0]: key for key, value in self.id2pinAndNet1.items()}
   self.net2id1 = {value[1]: key for key, value in self.id2pinAndNet1.items()}
   self.g1 = newg1
 
 def build_stdcell_lib(self):
  if self.createStdCellLibLUT:
   print("creating new std cell lib LUT")
   logic_truth_tables = {}
   cells_list=[ ("A2O1A1I", ['C', 'B', 'A2', 'A1'], "int( ((not(bits[3])) and (not(bits[1]))) or ((not(bits[2])) and (not(bits[1]))) or (not(bits[0])) )"), \
   ("A2O1A1O1I", ['D', 'C', 'B', 'A2', 'A1'], "int( ((not(bits[1])) and (not(bits[0]))) or ((not(bits[3])) and (not(bits[2])) and (not(bits[0]))) or ((not(bits[4])) and (not(bits[2])) and (not(bits[0]))) )"), \
   ("AO211", ['C', 'B', 'A2', 'A1'], "int((bits[3] and bits[2]) or bits[1] or bits[0])"), \
   ("AO21", ['B', 'A2', 'A1'], "int((bits[2] and bits[1]) or bits[0])"), \
   ("AO221", ['C', 'B2', 'B1', 'A2', 'A1'], "int((bits[4] and bits[3]) or (bits[2] and bits[1]) or bits[0])"), \
   ("AO222", ['C2', 'C1', 'B2', 'B1', 'A2', 'A1'], "int((bits[5] and bits[4]) or (bits[3] and bits[2]) or (bits[1] and bits[0]))"), \
   ("AO22", ['B2', 'B1', 'A2', 'A1'], "int((bits[3] and bits[2]) or (bits[1] and bits[0]))"), \
   ("AO31", ['B', 'A3', 'A2', 'A1'], "int((bits[3] and bits[2] and bits[1]) or bits[0])"), \
   ("AO322", ['C2', 'C1', 'B2', 'B1', 'A3', 'A2', 'A1'], "int((bits[6] and bits[5] and bits[4]) or (bits[3] and bits[2]) or (bits[1] and bits[0]))"), \
   ("AO32", ['B2', 'B1', 'A3', 'A2', 'A1'], "int((bits[4] and bits[3] and bits[2]) or (bits[1] and bits[0]))"), \
   ("AO331", ['C', 'B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int((bits[6] and bits[5] and bits[4]) or (bits[3] and bits[2] and bits[1]) or (bits[0]))"), \
   ("AO332", ['C2', 'C1', 'B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int((bits[7] and bits[6] and bits[5]) or (bits[4] and bits[3] and bits[2]) or (bits[1] and bits[0]))"), \
   ("AO333", ['C3', 'C2', 'C1', 'B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int((bits[8] and bits[7] and bits[6]) or (bits[5] and bits[4] and bits[3]) or (bits[2] and bits[1] and bits[0]))"), \
   ("AO33", ['B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int((bits[5] and bits[4] and bits[3]) or (bits[2] and bits[1] and bits[0]))"), \
   ("AOI211", ['C', 'B', 'A2', 'A1'], "int(not((bits[3] and bits[2]) or bits[1] or bits[0]))"), \
   ("AOI21", ['B', 'A2', 'A1'], "int(not((bits[2] and bits[1]) or bits[0]))"), \
   ("AOI221", ['C', 'B2', 'B1', 'A2', 'A1'], "int(not((bits[4] and bits[3]) or (bits[2] and bits[1]) or bits[0]))"), \
   ("AOI222", ['C2', 'C1', 'B2', 'B1', 'A2', 'A1'], "int(not((bits[5] and bits[4]) or (bits[3] and bits[2]) or (bits[1] and bits[0])))"), \
   ("AOI22", ['B2', 'B1', 'A2', 'A1'], "int(not((bits[3] and bits[2]) or (bits[1] and bits[0])))"), \
   ("AOI311", ['C', 'B', 'A3', 'A2', 'A1'], "int(not((bits[4] and bits[3] and bits[2]) or bits[1] or bits[0] ))"), \
   ("AOI31", ['B', 'A3', 'A2', 'A1'], "int(not((bits[3] and bits[2] and bits[1]) or bits[0]))"), \
   ("AOI321", ['C', 'B2', 'B1', 'A3', 'A2', 'A1'], "int(not((bits[5] and bits[4] and bits[3]) or (bits[2] and bits[1]) or bits[0] ))"), \
   ("AOI322", ['C2', 'C1', 'B2', 'B1', 'A3', 'A2', 'A1'], "int(not((bits[6] and bits[5] and bits[4]) or (bits[3] and bits[2]) or (bits[1] and bits[0]) ))"), \
   ("AOI32", ['B2', 'B1', 'A3', 'A2', 'A1'], "int(not((bits[4] and bits[3] and bits[2]) or (bits[1] and bits[0])))"), \
   ("AOI331", ['C1', 'B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int(not((bits[6] and bits[5] and bits[4]) or (bits[3] and bits[2] and bits[1]) or bits[0] ))"), \
   ("AOI332", ['C2', 'C1', 'B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int(not((bits[7] and bits[6] and bits[5]) or (bits[4] and bits[3] and bits[2]) or (bits[1] and bits[0]) ))"), \
   ("AOI333", ['C3', 'C2', 'C1', 'B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int(not((bits[8] and bits[7] and bits[6]) or (bits[5] and bits[4] and bits[3]) or (bits[2] and bits[1] and bits[0]) ))"), \
   ("AOI33", ['B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int(not((bits[5] and bits[4] and bits[3]) or (bits[2] and bits[1] and bits[0])))"), \
   ("BUF", ['A'], "int(bits[0])"), \
   ("CKINVDC", ['A'], "int(not(bits[0]))"), \
   ("HB1", ['A'], "int(bits[0])"), \
   ("HB2", ['A'], "int(bits[0])"), \
   ("HB3", ['A'], "int(bits[0])"), \
   ("HB4", ['A'], "int(bits[0])"), \
   ("INV", ['A'], "int(not(bits[0]))"), \
   ("O2A1O1I", ['C', 'B', 'A2', 'A1'], "int( ((not(bits[1])) and (not(bits[0]))) or ((not(bits[3])) and (not(bits[2])) and (not(bits[0]))) )"), \
   ("OA211", ['C', 'B', 'A2', 'A1'], "int((bits[3] or bits[2]) and bits[1] and bits[0])"), \
   ("OA21", ['B', 'A2', 'A1'], "int((bits[2] or bits[1]) and bits[0])"), \
   ("OA221", ['C', 'B2', 'B1', 'A2', 'A1'], "int((bits[4] or bits[3]) and (bits[2] or bits[1]) and bits[0])"), \
   ("OA222", ['C2', 'C1', 'B2', 'B1', 'A2', 'A1'], "int((bits[5] or bits[4]) and (bits[3] or bits[2]) and (bits[1] or bits[0]))"), \
   ("OA22", ['B2', 'B1', 'A2', 'A1'], "int((bits[3] or bits[2]) and (bits[1] or bits[0]))"), \
   ("OA31", ['B1', 'A3', 'A2', 'A1'], "int((bits[3] or bits[2] or bits[1]) and bits[0])"), \
   ("OA331", ['C1', 'B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int((bits[6] or bits[5] or bits[4]) and (bits[3] or bits[2] or bits[1]) and (bits[0]))"), \
   ("OA332", ['C2', 'C1', 'B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int((bits[7] or bits[6] or bits[5]) and (bits[4] or bits[3] or bits[2]) and (bits[1] or bits[0]))"), \
   ("OA333", ['C3', 'C2', 'C1', 'B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int((bits[8] or bits[7] or bits[6]) and (bits[5] or bits[4] or bits[3]) and (bits[2] or bits[1] or bits[0]))"), \
   ("OA33", ['B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int((bits[5] or bits[4] or bits[3]) and (bits[2] or bits[1] or bits[0]))"), \
   ("OAI211", ['C', 'B', 'A2', 'A1'], "int(not((bits[3] or bits[2]) and bits[1] and bits[0]))"), \
   ("OAI21", ['B', 'A2', 'A1'], "int(not((bits[2] or bits[1]) and bits[0]))"), \
   ("OAI221", ['C', 'B2', 'B1', 'A2', 'A1'], "int(not((bits[4] or bits[3]) and (bits[2] or bits[1]) and bits[0]))"), \
   ("OAI222", ['C2', 'C1', 'B2', 'B1', 'A2', 'A1'], "int(not((bits[5] or bits[4]) and (bits[3] or bits[2]) and (bits[1] or bits[0])))"), \
   ("OAI22", ['B2', 'B1', 'A2', 'A1'], "int(not((bits[3] or bits[2]) and (bits[1] or bits[0])))"), \
   ("OAI311", ['C1', 'B1', 'A3', 'A2', 'A1'], "int(not((bits[4] or bits[3] or bits[2]) and bits[1] and bits[0]))"), \
   ("OAI31", ['B', 'A3', 'A2', 'A1'], "int(not((bits[3] or bits[2] or bits[1]) and bits[0]))"), \
   ("OAI321", ['C', 'B2', 'B1', 'A3', 'A2', 'A1'], "int(not((bits[5] or bits[4] or bits[3]) and (bits[2] or bits[1]) and bits[0]))"), \
   ("OAI322", ['C2', 'C1', 'B2', 'B1', 'A3', 'A2', 'A1'], "int(not((bits[6] or bits[5] or bits[4]) and (bits[3] or bits[2]) and (bits[1] or bits[0])))"), \
   ("OAI32", ['B2', 'B1', 'A3', 'A2', 'A1'], "int(not((bits[4] or bits[3] or bits[2]) and (bits[1] or bits[0])))"), \
   ("OAI331", ['C1', 'B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int(not((bits[6] or bits[5] or bits[4]) and (bits[3] or bits[2] or bits[1]) and bits[0]))"), \
   ("OAI332", ['C2', 'C1', 'B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int(not((bits[7] or bits[6] or bits[5]) and (bits[4] or bits[3] or bits[2]) and (bits[1] or bits[0])))"), \
   ("OAI333", ['C3', 'C2', 'C1', 'B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int(not((bits[8] or bits[7] or bits[6]) and (bits[5] or bits[4] or bits[3]) and (bits[2] or bits[1] or bits[0])))"), \
   ("OAI33", ['B3', 'B2', 'B1', 'A3', 'A2', 'A1'], "int(not((bits[5] or bits[4] or bits[3]) and (bits[2] or bits[1] or bits[0])))"), \
   ("AND2", ['B', 'A'], "int(bits[1] and bits[0])"), \
   ("AND3", ['C', 'B', 'A'], "int(bits[2] and bits[1] and bits[0])"), \
   ("AND4", ['D', 'C', 'B', 'A'], "int(bits[3] and bits[2] and bits[1] and bits[0])"), \
   ("AND5", ['E', 'D', 'C', 'B', 'A'], "int(bits[4] and bits[3] and bits[2] and bits[1] and bits[0])"), \
   ("FASN", ['CI', 'B', 'A'], "int(not(bits[2] ^ bits[1] ^ bits[0]))"), \
   ("FACON", ['CI', 'B', 'A'], "int(not((bits[0] and bits[1]) or (bits[2] and (bits[0] ^ bits[1]))))"), \
   ("HASN", ['B', 'A'], "int(not(bits[1] ^ bits[0]))"), \
   ("HACON", ['B', 'A'], "int(not(bits[1] and bits[0]))"), \
   ("MAJI", ['C', 'B', 'A'], "int(not((bits[1] and bits[0]) or (bits[2] and bits[0]) or (bits[1] and bits[2])))"), \
   ("MAJ", ['C', 'B', 'A'], "int((bits[1] and bits[0]) or (bits[2] and bits[0]) or (bits[1] and bits[2]))"), \
   ("NAND2", ['B', 'A'], "int(not(bits[1] and bits[0]))"), \
   ("NAND3", ['C', 'B', 'A'], "int(not(bits[2] and bits[1] and bits[0]))"), \
   ("NAND4", ['D', 'C', 'B', 'A'], "int(not(bits[3] and bits[2] and bits[1] and bits[0]))"), \
   ("NAND5", ['E', 'D', 'C', 'B', 'A'], "int(not(bits[4] and bits[3] and bits[2] and bits[1] and bits[0]))"), \
   ("NOR2", ['B', 'A'], "int(not(bits[1] or bits[0]))"), \
   ("NOR3", ['C', 'B', 'A'], "int(not(bits[2] or bits[1] or bits[0]))"), \
   ("NOR4", ['D', 'C', 'B', 'A'], "int(not(bits[3] or bits[2] or bits[1] or bits[0]))"), \
   ("NOR5", ['E', 'D', 'C', 'B', 'A'], "int(not(bits[4] or bits[3] or bits[2] or bits[1] or bits[0]))"), \
   ("OR2", ['B', 'A'], "int(bits[1] or bits[0])"), \
   ("OR3", ['C', 'B', 'A'], "int(bits[2] or bits[1] or bits[0])"), \
   ("OR4", ['D', 'C', 'B', 'A'], "int(bits[3] or bits[2] or bits[1] or bits[0])"), \
   ("OR5", ['E', 'D', 'C', 'B', 'A'], "int(bits[4] or bits[3] or bits[2] or bits[1] or bits[0])"), \
   ("XNOR2", ['B', 'A'], "int(not(bits[1] ^ bits[0]))"), \
   ("XOR2", ['B', 'A'], "int(bits[1] ^ bits[0])") ]
   
   def numberToBase(n, b, length):
    if n == 0:
     returnal = 0
    digits = []
    while n:
     digits.append(int(n % b))
     n //= b
    returnal = digits[::-1]
    while len(returnal) < length:
     returnal.insert(0,0)
    return returnal
   
   cell_counter = 0
   for cell_info in cells_list:
    cell_name=cell_info[0]
    cell_pins=cell_info[1]
    cell_func=cell_info[2]
    logic_truth_tables[cell_name]={}
    logic_truth_tables[cell_name]['pins']=cell_pins
    logic_truth_tables[cell_name]['cell_id']=cell_counter ; cell_counter+=1 ;
    truth_table = np.zeros(shape=(2**len(logic_truth_tables[cell_name]['pins']),len(logic_truth_tables[cell_name]['pins'])+1), dtype=np.uint8)
    for i in range(2**len(logic_truth_tables[cell_name]['pins'])):
     num_of_pins = len(logic_truth_tables[cell_name]['pins'])
     bits_np = np.array(numberToBase(i,2,num_of_pins))
     bits = list(bits_np)
     bits.reverse()
     output = eval(cell_func)
     bits.reverse()
     bits.append(output)
     truth_table[i] = bits
    logic_truth_tables[cell_name]['truth_table']=truth_table
   
   out_array = []
   for cell_type in logic_truth_tables.keys():
    out_array.append(th.ByteTensor(logic_truth_tables[cell_type]['truth_table'][:,-1]))
   out_offset = th.IntTensor([ x.size()[0] for x in out_array ])
   out_offset = th.roll(th.cumsum(out_offset,  dim=0), 1, 0)
   out_offset[0] = 0
   out_offset = out_offset.type(th.int32)
   out_array = th.concat(out_array)
   th.save( (out_array, out_offset), "MLCADDesignContest2025StdCellLibLUT")
   print("std cell lib LUT done...")
   self.stdcell_array = out_array ; self.stdcell_offsets = out_offset ;
  else:
   self.stdcell_array, self.stdcell_offsets = th.load("MLCADDesignContest2025StdCellLibLUT")
  self.stdcell_array_GPU = cp.asarray(self.stdcell_array)
 
 
 def processLoopsAndIOs(self, g, graphName: str, id2pinAndNet):
  temp_start = timer() ;
  print("checking if " + graphName + " has combinational loops...") 
  nx_g = dgl.to_networkx(g)
  listOfLoops = sorted(nx.simple_cycles(nx_g))
  if len(listOfLoops):
   print(graphName + " has combinational loops, results may not be accurate if net within loop drives a Sequential component")
   g.ndata['logicLevel'] = th.zeros( len(g.nodes()), dtype = th.int16)
   g.ndata['loopsPresent'] = th.zeros( len(g.nodes()), dtype = th.int32)
   allParticipatingLoops = th.concat([th.IntTensor(x) for x in listOfLoops])
   participatingNodes, loopCount = th.unique(allParticipatingLoops, return_counts = True)
   participatingNodes = participatingNodes.type(th.int64)
   g.ndata['loopsPresent'][participatingNodes] = loopCount.type(th.int32)
   #if too many loops, will overflow. So set a max value I guess.
   self.loopsMaxIter = max( int(2 ** th.max(g.ndata['loopsPresent'])), self.cycles )
   brokenEdgeSrc = [] ; brokenEdgeDst = [] ; brokenEdgeX = []
   for loop in listOfLoops:
    brokenEdgeSrc.append(loop[-1]) ; brokenEdgeDst.append(loop[0]) ; 
   brokenEdgeSrc = th.LongTensor(brokenEdgeSrc) ; brokenEdgeDst = th.LongTensor(brokenEdgeDst) ; 
   oldLoopValues = cp.asarray(th.zeros( size=(participatingNodes.size()[0],self.PARALLEL_CYCLES), dtype=th.uint8 ))
   brokenEdgeIDs = th.unique(g.edge_ids(brokenEdgeSrc, brokenEdgeDst))
   brokenEdgeSrc, brokenEdgeDst = g.find_edges(brokenEdgeIDs)
   brokenEdgeX = g.edata['x'][brokenEdgeIDs]
   g = dgl.remove_edges(g,brokenEdgeIDs)
   loop_sg = dgl.node_subgraph(g, participatingNodes)
   topo_loop_cpu = dgl.traversal.topological_nodes_generator(loop_sg)
  topo_nodes_cpu =  dgl.traversal.topological_nodes_generator(g)
  outputs= dgl.topological_nodes_generator(g, reverse=True)[0]
  outputs = outputs[ g.in_degrees(outputs) > 0 ]
  inputNodes = topo_nodes_cpu[0] ; 
  inputNodes = inputNodes[ g.out_degrees(inputNodes) > 0 ]
  inputsToRemove =[]; inputNodes = list(inputNodes)
  if len(listOfLoops):
   for i in inputNodes:
    if i in participatingNodes:
     inputsToRemove.append(i)
  for i in inputsToRemove:
   inputNodes.remove(i)
  inputNodes = cp.asarray(inputNodes)
  #right now we don't use Unconnected outputs, currently using netname "UNCONNECTED" regex to do filtering
  outputs = outputs.tolist() ; toRemove =[]
  for i in outputs:
   netName = id2pinAndNet[i][1]
   if re.search(r"^UNCONNECTED", netName):
    toRemove.append(i) ; continue;
   if len(listOfLoops):
    if i in participatingNodes:
     toRemove.append(i) ; 
  for i in toRemove:
   outputs.remove(i)
  outputs = cp.asarray(outputs, dtype=cp.int32)
  outputsTotal = cp.asarray(th.full(size=(outputs.shape[0],self.cycles32), fill_value=9, dtype=th.uint8))
  g.ndata['celltype'][g.ndata['celltype'] == 999] = 0
  g.ndata['celloffsets'] = self.stdcell_offsets[g.ndata['celltype'].type(th.int32)]
  if len(listOfLoops):
   g = dgl.add_edges(g, brokenEdgeSrc,brokenEdgeDst, {'x' : brokenEdgeX} )
   temp_delta = timer() - temp_start
   printATimer("Loop processing", temp_delta)
   return g, participatingNodes, loop_sg, oldLoopValues, topo_loop_cpu, topo_nodes_cpu, inputNodes, outputs, outputsTotal
  else:
   temp_delta = timer() - temp_start
   printATimer("Loop processing", temp_delta)
   return g, None, None, None, None, topo_nodes_cpu, inputNodes, outputs, outputsTotal
 
 
 def processLoopsAndIOsForGoldenAndCompareNetlists(self):
  self.g0, self.loopParticipatingNodes0, self.loop_sg0, self.oldLoopValues0, self.topo_loop_cpu0, self.topo_nodes_cpu0, \
   self.inputNodes0, self.outputs0, self.outputsTotal0 = self.processLoopsAndIOs(self.g0, "Golden", self.id2pinAndNet0)
  self.g1, self.loopParticipatingNodes1, self.loop_sg1, self.oldLoopValues1, self.topo_loop_cpu1, self.topo_nodes_cpu1, \
   self.inputNodes1, self.outputs1, self.outputsTotal1 = self.processLoopsAndIOs(self.g1, "Compare", self.id2pinAndNet1)
 
 
 def alignIOs(self):
  print("aligning inputs and outputs of the two netlists...")
  temp_start = timer()
  assert len(self.inputNodes0) == len(self.inputNodes1), "The two graphs don't have the same number of input nodes!"
  for i in range(len(self.inputNodes0)):
   thisID = int(self.inputNodes0[i]) ; thisNet = self.id2pinAndNet0[thisID][1] ; 
   alignedInput = self.net2id1[thisNet] ;
   self.inputNodes1[i] = alignedInput
  assert self.outputs1.shape[0] == self.outputs0.shape[0], "The two graphs don't have the same number of output nodes!"
  for i in range(self.outputs0.shape[0]):
   thisID = int(self.outputs0[i]) ; thisNet = self.id2pinAndNet0[thisID][1] ; 
   alignedOutput = self.net2id1[thisNet] ;
   self.outputs1[i] = alignedOutput
  temp_delta = timer() - temp_start
  printATimer("Aligning IOs", temp_delta)
 
 
 def initializeSimAndCreateRandomInputPattern(self):
  self.inputsTotal = cp.asarray(th.ByteTensor(np.random.randint(0,2, (len(self.inputNodes0),self.cycles32))))
 
 
 def cudaArrayfyGraph(self, graphName: str, g, topo_nodes_cpu, loop_sg, loopParticipatingNodes, topo_loop_cpu):
  temp_start = timer()
  nodesPerStage=[]; driversPerGate=[] ; edgeOffsets=[] ; drivers =[]; celltypes = []; pinPositions=[]
  for logicStage in range(1,len(topo_nodes_cpu)):
   theseNodes = topo_nodes_cpu[logicStage]; 
   if loop_sg != None:
    g.ndata['logicLevel'][theseNodes] = logicStage 
   theseDrivers, dummy =  g.in_edges( theseNodes ) ; 
   #this roundabout stuff is done to process the case of one driver driving multiple input pins of the same cell
   toTuple = [(int(theseDrivers[i]), int(dummy[i])) for i in range(theseDrivers.size()[0])] ; toTensor = th.LongTensor(list(set(toTuple)))
   dummy2, shuffleIndex = toTensor[:,1].sort() ; theseDrivers2 = toTensor[:,0][shuffleIndex] ; theseNodes2 = th.unique(dummy2) ;
   nodesPerStage.append(cp.asarray(theseNodes2.type(th.int32)));
   celltypes.append(cp.asarray(g.ndata['celloffsets'][theseNodes2].type(th.int32)));
   in_degs = g.in_degrees(theseNodes2) ; driversPerGate.append(cp.asarray(in_degs.type(th.uint8)));
   theseEdgeOffsets = th.roll(th.cumsum(in_degs,  dim=0), 1, 0) ; theseEdgeOffsets[0] = 0 ; edgeOffsets.append(cp.asarray(theseEdgeOffsets));
   actualDrivers, notUsed, edgeIDs = g.edge_ids(theseDrivers2, dummy2, return_uv=True) ; drivers.append(cp.asarray(actualDrivers.type(th.int32)));
   pinPositions.append(cp.asarray(g.edata['x'][edgeIDs])) ; 
  deepestLoopStage=None; nodesPerStage_loop=None; driversPerGate_loop=None; edgeOffsets_loop=None; drivers_loop=None; celltypes_loop=None; pinPositions_loop=None;
  if loop_sg != None:
   deepestLoopStage = th.max(g.ndata['logicLevel'][loopParticipatingNodes])
   nodesPerStage_loop=[]; driversPerGate_loop=[] ; edgeOffsets_loop=[] ; drivers_loop =[]; celltypes_loop = []; pinPositions_loop=[]
   for logicStage in range(len(topo_loop_cpu)):
    theseNodes = loop_sg.ndata['_ID'][topo_loop_cpu[logicStage]];
    theseDrivers, dummy =  g.in_edges( theseNodes ) ; 
    #this roundabout stuff is done to process the case of one driver driving multiple input pins of the same cell
    toTuple = [(int(theseDrivers[i]), int(dummy[i])) for i in range(theseDrivers.size()[0])] ; toTensor = th.LongTensor(list(set(toTuple)))
    dummy2, shuffleIndex = toTensor[:,1].sort() ; theseDrivers2 = toTensor[:,0][shuffleIndex] ; theseNodes2 = th.unique(dummy2) ;
    nodesPerStage_loop.append(cp.asarray(theseNodes2.type(th.int32)));
    celltypes_loop.append(cp.asarray(g.ndata['celloffsets'][theseNodes2].type(th.int32)));
    in_degs = g.in_degrees(theseNodes2) ; driversPerGate_loop.append(cp.asarray(in_degs.type(th.uint8)));
    theseEdgeOffsets = th.roll(th.cumsum(in_degs,  dim=0), 1, 0) ; theseEdgeOffsets[0] = 0 ; edgeOffsets_loop.append(cp.asarray(theseEdgeOffsets));
    actualDrivers, notUsed, edgeIDs = g.edge_ids(theseDrivers2, dummy2, return_uv=True) ; drivers_loop.append(cp.asarray(actualDrivers.type(th.int32)));
    pinPositions_loop.append(cp.asarray(g.edata['x'][edgeIDs])) ; 
  temp_delta = timer() - temp_start
  printATimer(graphName + ' graph --> cuda arrays', temp_delta)
  return nodesPerStage, driversPerGate, edgeOffsets, drivers, celltypes, pinPositions, \
   deepestLoopStage, nodesPerStage_loop, driversPerGate_loop, edgeOffsets_loop, drivers_loop, celltypes_loop, pinPositions_loop
 
 
 def gatspiSimACycle(self, rangeStart, rangeEnd, nodesPerStage, celltypes, driversPerGate, drivers, pinPositions, edgeOffsets):
  for logicStage in range(rangeStart, rangeEnd):
   theseNodes = nodesPerStage[logicStage] ; theseCelltypes = celltypes[logicStage]; numDrivers = driversPerGate[logicStage];
   theseDrivers = drivers[logicStage] ; thesePinPositions = pinPositions[logicStage]; theseEdgeOffsets = edgeOffsets[logicStage];
   evalLogic( (1,math.ceil(theseNodes.shape[0]/(self.THREADS_PER_BLOCK/self.PARALLEL_CYCLES))), (self.PARALLEL_CYCLES,(self.THREADS_PER_BLOCK/self.PARALLEL_CYCLES)),\
    (self.currentLogicValue,theseNodes,theseCelltypes,numDrivers,theseDrivers,thesePinPositions,theseEdgeOffsets,\
    self.stdcell_array_GPU,theseNodes.shape[0],self.PARALLEL_CYCLES) )  
 
 def gatspiSim(self, graphName: str, g, loop_sg, inputNodes, loopParticipatingNodes, \
 deepestLoopStage, topo_loop_cpu, nodesPerStage_loop, celltypes_loop, driversPerGate_loop, drivers_loop, pinPositions_loop, edgeOffsets_loop, oldLoopValues, \
 topo_nodes_cpu, nodesPerStage, celltypes, driversPerGate, drivers, pinPositions, edgeOffsets, outputsTotal, outputs):
  print("Start " + graphName + ' simulation...')
  temp_start = timer()
  mempool = cp.get_default_memory_pool()
  mempool.free_all_blocks()
  self.currentLogicValue = cp.asarray(th.zeros( size=(g.nodes().shape[0],self.PARALLEL_CYCLES), dtype=th.uint8 ))
  if loop_sg != None:
   for c in range(self.simLoops):
    self.currentLogicValue[inputNodes] = self.inputsTotal[:,c*self.PARALLEL_CYCLES:c*self.PARALLEL_CYCLES+self.PARALLEL_CYCLES]
    self.gatspiSimACycle(0,deepestLoopStage, nodesPerStage, celltypes, driversPerGate, drivers, pinPositions, edgeOffsets)
    loopConverged = 0 ; loopCycles=0 ;
    while ((not loopConverged) and loopCycles<self.loopsMaxIter):
     self.gatspiSimACycle(0,len(topo_loop_cpu), nodesPerStage_loop, celltypes_loop, driversPerGate_loop, drivers_loop, pinPositions_loop, edgeOffsets_loop)
     loopCycles+=1; loopConverged = cp.all(self.currentLogicValue[loopParticipatingNodes] == oldLoopValues) ;
     oldLoopValues = self.currentLogicValue[loopParticipatingNodes] ;
    assert loopConverged, "There are non-convergent combinational loops in your design! Check it!!!"
    self.gatspiSimACycle(0,len(topo_nodes_cpu)-1, nodesPerStage, celltypes, driversPerGate, drivers, pinPositions, edgeOffsets)
    outputsTotal[:,c*self.PARALLEL_CYCLES:c*self.PARALLEL_CYCLES+self.PARALLEL_CYCLES] = self.currentLogicValue[outputs]
  else:
   for c in range(self.simLoops):
    self.currentLogicValue[inputNodes] = self.inputsTotal[:,c*self.PARALLEL_CYCLES:c*self.PARALLEL_CYCLES+self.PARALLEL_CYCLES]
    self.gatspiSimACycle(0,len(topo_nodes_cpu)-1, nodesPerStage, celltypes, driversPerGate, drivers, pinPositions, edgeOffsets)
    outputsTotal[:,c*self.PARALLEL_CYCLES:c*self.PARALLEL_CYCLES+self.PARALLEL_CYCLES] = self.currentLogicValue[outputs]
  temp_delta = timer() - temp_start
  printATimer(graphName + " simulation for " + str(self.cycles32) + ' cycles', temp_delta) 
 
 def simulateGoldenAndCompareGraphs(self):
  nodesPerStage, driversPerGate, edgeOffsets, drivers, celltypes, pinPositions, deepestLoopStage, \
  nodesPerStage_loop, driversPerGate_loop, edgeOffsets_loop, drivers_loop, celltypes_loop, pinPositions_loop = \
  self.cudaArrayfyGraph("Golden", self.g0, self.topo_nodes_cpu0, self.loop_sg0, \
  self.loopParticipatingNodes0, self.topo_loop_cpu0)
  self.gatspiSim("Golden", self.g0, self.loop_sg0, self.inputNodes0, self.loopParticipatingNodes0, \
  deepestLoopStage, self.topo_loop_cpu0, nodesPerStage_loop, celltypes_loop, driversPerGate_loop, drivers_loop, \
  pinPositions_loop, edgeOffsets_loop, self.oldLoopValues0, self.topo_nodes_cpu0, nodesPerStage, celltypes, driversPerGate, \
  drivers, pinPositions, edgeOffsets, self.outputsTotal0, self.outputs0)
  
  nodesPerStage, driversPerGate, edgeOffsets, drivers, celltypes, pinPositions, deepestLoopStage, \
  nodesPerStage_loop, driversPerGate_loop, edgeOffsets_loop, drivers_loop, celltypes_loop, pinPositions_loop = \
  self.cudaArrayfyGraph("Compare", self.g1, self.topo_nodes_cpu1, self.loop_sg1, \
  self.loopParticipatingNodes1, self.topo_loop_cpu1)
  self.gatspiSim("Compare", self.g1, self.loop_sg1, self.inputNodes1, self.loopParticipatingNodes1, \
  deepestLoopStage, self.topo_loop_cpu1, nodesPerStage_loop, celltypes_loop, driversPerGate_loop, drivers_loop, \
  pinPositions_loop, edgeOffsets_loop, self.oldLoopValues1, self.topo_nodes_cpu1, nodesPerStage, celltypes, driversPerGate, \
  drivers, pinPositions, edgeOffsets, self.outputsTotal1, self.outputs1)
 
 def compareResults(self):
  print("start result compare...")
  temp_start = timer()
  if cp.all(self.outputsTotal1==self.outputsTotal0):
   print("results match, valid resynth edit")
  else:
   print("results don't match, invalid resynth edit")
   wrongIDindex, cycle = cp.where(self.outputsTotal1!=self.outputsTotal0)
   wrongID = int(self.outputs0[wrongIDindex[0]]) ; cycle = int(cycle[0])
   sg, inverse_indices = dgl.khop_in_subgraph(self.g0, wrongID, k=len(self.topo_nodes_cpu0))
   wrongIDname = self.id2pinAndNet0[wrongID][0] ; wrongIDnet = self.id2pinAndNet0[wrongID][1]
   sgInputs = dgl.traversal.topological_nodes_generator(sg)[0]
   wrongInputIDs = sg.ndata['_ID'][sgInputs]
   print("output node " + wrongIDname + ' (net ' + wrongIDnet + ') is incorrect. With inputs:') ; print_string = ''
   for i in wrongInputIDs:
    ii = int(i)
    wrongInputName = self.id2pinAndNet0[ii][0]
    inputIndex = int(cp.where(self.inputNodes0 == ii)[0]) ; wrongInputValue = int(self.inputsTotal[inputIndex,cycle]) ;
    print_string += wrongInputName + ' = ' + str(wrongInputValue) + '\t'
   print(print_string)
   rightValue = int(self.outputsTotal0[wrongIDindex[0],cycle]) ; wrongValue = int(self.outputsTotal1[wrongIDindex[0],cycle])
   print("SHOULD BE: " + str(rightValue) + ' BUT IS: ' + str(wrongValue))
  temp_delta = timer() - temp_start
  printATimer("Golden vs Edited comparison", temp_delta)
 
 
 def doEverything(self):
  self.build_graphs()
  self.build_stdcell_lib()
  self.processLoopsAndIOsForGoldenAndCompareNetlists()
  self.alignIOs()
  self.initializeSimAndCreateRandomInputPattern()
  self.simulateGoldenAndCompareGraphs()
  self.compareResults()
