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

parser = argparse.ArgumentParser()
parser.add_argument('--top_name', type=str, help = 'top module name')
parser.add_argument('--graph0FilePath', type=str, help='raw csr graph file path, or stored DGL graph file path for golden netlist')
parser.add_argument('--graph1FilePath', type=str, help='raw csr graph file path, or stored DGL graph file path for resynth netlist')
parser.add_argument('--dumpDGLGraph', type=bool, default=False, help='dump the created DGL graph or not. \
Should be true and ran once when creating the DGL graph from raw CSR graph, from then on can be set to False to simply load the DGL graph')
parser.add_argument('--createStdCellLibLUT', type=bool, default=False, help='compile the std cell library truth tables or not. should be run once for each new technology')
parser.add_argument('--cycles', type=int, default=50000, help='target verification cycles to run')
parser.add_argument('--parallel_sim_cycles', type=int, default=32, choices=[1,2,4,8,16,32,64,128,256], help='# of cycles to be simulated in parallel on GPU')
#args = parser.parse_args()
args = parser.parse_args(['--top_name', 'adder', '--graph0FilePath', './adder.pkl', '--graph1FilePath', \
'./adder.pkl', '--dumpDGLGraph', '1'])

#data loading, builds the DGL graph from csr raw graph
def build_graph(pkl):
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
 print("graph created")
 later=datetime.now()
 delta=(later-now).total_seconds()
 print("creating the DGL graph took " + str(delta) + " seconds on the CPU")
 return g, num_of_gatspi_cells, num_of_top_ports, id2pinAndNet, id2port

if args.dumpDGLGraph:
 print("creating the DGL graph from raw CSR graph...")
 temp_start = timer() ;
 g0, num_of_gatspi_cells0, num_of_top_ports0, id2pinAndNet0, id2port0 = build_graph(args.graph0FilePath)
 fileObject = open(args.top_name + "_DGLGraph0", 'wb')
 pickle.dump({'g': g0,
  'num_of_gatspi_cells' : num_of_gatspi_cells0,
  'num_of_top_ports' : num_of_top_ports0,
  'id2pinAndNet' : id2pinAndNet0,
  'id2port' : id2port0}, fileObject)
 g1, num_of_gatspi_cells1, num_of_top_ports1, id2pinAndNet1, id2port1 = build_graph(args.graph1FilePath)
 fileObject = open(args.top_name + "_DGLGraph1", 'wb')
 pickle.dump({'g': g1,
  'num_of_gatspi_cells' : num_of_gatspi_cells1,
  'num_of_top_ports' : num_of_top_ports1,
  'id2pinAndNet' : id2pinAndNet1,
  'id2port' : id2port1}, fileObject)
 temp_delta = timer() - temp_start
 print("DGL graph done in " + f"{temp_delta:.3f}" + ' seconds')
else:
 data = np.load(args.graph0FilePath, allow_pickle=True);
 g0 = data['g']; num_of_gatspi_cells0 = data['num_of_gatspi_cells'] ; num_of_top_ports0 = data['num_of_top_ports'] ; 
 id2pinAndNet0 = data['id2pinAndNet'] ; id2port0 = data['id2port'] ;
 data = np.load(args.graph1FilePath, allow_pickle=True);
 g1 = data['g']; num_of_gatspi_cells1 = data['num_of_gatspi_cells'] ; num_of_top_ports1 = data['num_of_top_ports'] ; 
 id2pinAndNet1 = data['id2pinAndNet'] ; id2port1 = data['id2port'] ;

if args.createStdCellLibLUT:
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
else:
 out_array, out_offset = th.load("MLCADDesignContest2025StdCellLibLUT")

print("start golden simulation graph setup...")
import cupy as cp
temp_start = timer()
#need to figure out and align the inputs and outputs for both graphs here. If something doesn't match I think the dictionary
#translation will throw an error
port2id0 = {value: key for key, value in id2port0.items()}
driverPin2id0 = {tupleThing[0]: (index+num_of_top_ports0) for index, tupleThing in enumerate(id2pinAndNet0)}
net2id0 = {tupleThing[1]: (index+num_of_top_ports0) for index, tupleThing in enumerate(id2pinAndNet0)}
port2id1 = {value: key for key, value in id2port1.items()}
driverPin2id1 = {tupleThing[0]: (index+num_of_top_ports1) for index, tupleThing in enumerate(id2pinAndNet1)}
net2id1 = {tupleThing[1]: (index+num_of_top_ports1) for index, tupleThing in enumerate(id2pinAndNet1)}

PARALLEL_CYCLES=args.parallel_sim_cycles
cycles32 = math.ceil(args.cycles/PARALLEL_CYCLES) * PARALLEL_CYCLES
simLoops = int(cycles32/PARALLEL_CYCLES)
topo_nodes_cpu0 =  dgl.traversal.topological_nodes_generator(g0)
inputNodes0 = topo_nodes_cpu0[0] ; 
inputNodes0 = inputNodes0[ g0.out_degrees(inputNodes0) > 0 ]
numOfInputNodes0 = inputNodes0.size()[0]
inputNodes0 = cp.asarray(inputNodes0)
#shared inputsTotal
inputsTotal = cp.asarray(th.ByteTensor(np.random.randint(0,2, (numOfInputNodes0,cycles32))))
currentLogicValue = cp.asarray(th.zeros( size=(g0.nodes().shape[0],PARALLEL_CYCLES), dtype=th.uint8 ))
#update currentLogicValue separately for graph1. currentLogicValue is a temporary variable anyway
outputs0= dgl.topological_nodes_generator(g0, reverse=True)[0]
outputs0 = outputs0[ g0.in_degrees(outputs0) > 0 ]
#right now we don't use Unconnected outputs, currently using netname "UNCONNECTED" regex to do filtering
outputs0 = outputs0.tolist() ; toRemove =[]
for i in outputs0:
 netName = id2pinAndNet0[i-num_of_top_ports0][1] if i >= num_of_top_ports0 else id2port0[i]
 if re.search(r"^UNCONNECTED", netName):
  toRemove.append(i) 
for i in toRemove:
 outputs0.remove(i)
outputs0 = cp.asarray(outputs0, dtype=cp.int32)
outputsTotal0 = cp.asarray(th.full(size=(outputs0.shape[0],cycles32), fill_value=9, dtype=th.uint8))
g0.ndata['celltype'][g0.ndata['celltype'] == 999] = 0
g0.ndata['celloffsets'] = out_offset[g0.ndata['celltype'].type(th.int32)]
out_array_GPU = cp.asarray(out_array)

topo_nodes_cpu1 =  dgl.traversal.topological_nodes_generator(g1)
inputNodes1 = topo_nodes_cpu1[0] ; 
inputNodes1 = inputNodes1[ g1.out_degrees(inputNodes1) > 0 ]
numOfInputNodes1 = inputNodes1.size()[0]
inputNodes1 = cp.asarray(inputNodes1)
assert numOfInputNodes0 == numOfInputNodes1, "The two graphs don't have the same number of input nodes!"
for i in range(numOfInputNodes0):
 thisID = int(inputNodes0[i]) ; thisNet = id2pinAndNet0[thisID-num_of_top_ports0][1] if thisID >= num_of_top_ports0 else id2port0[thisID] ; 
 alignedInput = port2id1[thisNet] if thisNet in port2id1.keys() else net2id1[thisNet] ;
 inputNodes1[i] = alignedInput

outputs1= dgl.topological_nodes_generator(g1, reverse=True)[0]
outputs1 = outputs1[ g1.in_degrees(outputs1) > 0 ]
outputs1 = outputs1.tolist() ; toRemove =[]
for i in outputs1:
 netName = id2pinAndNet1[i-num_of_top_ports1][1] if i >= num_of_top_ports1 else id2port1[i]
 if re.search(r"^UNCONNECTED", netName):
  toRemove.append(i) 
for i in toRemove:
 outputs1.remove(i)
outputs1 = cp.asarray(outputs1, dtype=cp.int32)
outputsTotal1 = cp.asarray(th.full(size=(outputs1.shape[0],cycles32), fill_value=9, dtype=th.uint8))
g1.ndata['celltype'][g1.ndata['celltype'] == 999] = 0
g1.ndata['celloffsets'] = out_offset[g1.ndata['celltype'].type(th.int32)]

assert outputs1.shape[0] == outputs0.shape[0], "The two graphs don't have the same number of output nodes!"
for i in range(outputs0.shape[0]):
 thisID = int(outputs0[i]) ; thisNet = id2pinAndNet0[thisID-num_of_top_ports0][1] if thisID >= num_of_top_ports0 else id2port0[thisID] ; 
 alignedOutput = port2id1[thisNet] if thisNet in port2id1.keys() else net2id1[thisNet] ;
 outputs1[i] = alignedOutput

exec(open('evalLogic.cupy').read())

nodesPerStage=[]; driversPerGate=[] ; edgeOffsets=[] ; drivers =[]; celltypes = []; pinPositions=[]
for logicStage in range(1,len(topo_nodes_cpu0)):
 theseNodes = topo_nodes_cpu0[logicStage]; 
 theseDrivers, dummy =  g0.in_edges( theseNodes ) ; 
 #this roundabout stuff is done to process the case of one driver driving multiple input pins of the same cell
 toTuple = [(int(theseDrivers[i]), int(dummy[i])) for i in range(theseDrivers.size()[0])] ; toTensor = th.LongTensor(list(set(toTuple)))
 dummy2, shuffleIndex = toTensor[:,1].sort() ; theseDrivers2 = toTensor[:,0][shuffleIndex] ; theseNodes2 = th.unique(dummy2) ;
 nodesPerStage.append(cp.asarray(theseNodes2.type(th.int32)));
 celltypes.append(cp.asarray(g0.ndata['celloffsets'][theseNodes2].type(th.int32)));
 in_degs = g0.in_degrees(theseNodes2) ; driversPerGate.append(cp.asarray(in_degs.type(th.uint8)));
 theseEdgeOffsets = th.roll(th.cumsum(in_degs,  dim=0), 1, 0) ; theseEdgeOffsets[0] = 0 ; edgeOffsets.append(cp.asarray(theseEdgeOffsets));
 actualDrivers, notUsed, edgeIDs = g0.edge_ids(theseDrivers2, dummy2, return_uv=True) ; drivers.append(cp.asarray(actualDrivers.type(th.int32)));
 pinPositions.append(cp.asarray(g0.edata['x'][edgeIDs])) ; 
temp_delta = timer() - temp_start
print("Golden sim graph done in " + f"{temp_delta:.3f}" + ' seconds')

print("start golden simulation...")
temp_start = timer()
for c in range(simLoops):
 currentLogicValue[inputNodes0] = inputsTotal[:,c*PARALLEL_CYCLES:c*PARALLEL_CYCLES+PARALLEL_CYCLES]
 for logicStage in range(len(topo_nodes_cpu0)-1):
  theseNodes = nodesPerStage[logicStage] ; theseCelltypes = celltypes[logicStage]; numDrivers = driversPerGate[logicStage];
  theseDrivers = drivers[logicStage] ; thesePinPositions = pinPositions[logicStage]; theseEdgeOffsets = edgeOffsets[logicStage];
  evalLogic( (1,math.ceil(theseNodes.shape[0]/(512/PARALLEL_CYCLES))), (PARALLEL_CYCLES,(512/PARALLEL_CYCLES)),\
   (currentLogicValue,theseNodes,theseCelltypes,numDrivers,theseDrivers,thesePinPositions,theseEdgeOffsets,\
   out_array_GPU,theseNodes.shape[0],PARALLEL_CYCLES) )
 outputsTotal0[:,c*PARALLEL_CYCLES:c*PARALLEL_CYCLES+PARALLEL_CYCLES] = currentLogicValue[outputs0]
temp_delta = timer() - temp_start
print("Golden simulation for " + str(cycles32) + ' cycles done in ' + f"{temp_delta:.3f}" + ' seconds')

for c in range(PARALLEL_CYCLES):
 A=[] ; B=[]; C=[] ; printA='' ; printB='' ;  printC='' ; 
 for i in range(31,-1,-1):
  aName = 'a' + '[' + str(i) + ']' ; bName = 'b' + '[' + str(i) + ']' ; cName = 'c' + '[' + str(i) + ']' ; 
  bitIDa = port2id0[aName] ;  bitIDb = port2id0[bName] ;bitIDc = port2id0[cName] ;
  A.append(str(int(currentLogicValue[bitIDa,c]))) ; B.append(str(int(currentLogicValue[bitIDb,c]))) ; C.append(str(int(currentLogicValue[bitIDc,c]))) ; 
 A = "".join(A) ; B = "".join(B) ; C = "".join(C) ; 
 printA += 'a' + '[' + str(31) + ':' + str(0) + ']' + " : " + str(hex(int(A, base=2)))
 printB += 'b' + '[' + str(31) + ':' + str(0) + ']' + " : " + str(hex(int(B, base=2)))
 printC += 'c' + '[' + str(31) + ':' + str(0) + ']' + " : " + str(hex(int(C, base=2)))
 print(printA + ' ' + printB + ' : ' + printC) 

print("start edited simulation graph setup...")
temp_start = timer()
mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()
currentLogicValue = cp.asarray(th.zeros( size=(g1.nodes().shape[0],PARALLEL_CYCLES), dtype=th.uint8 ))

nodesPerStage=[]; driversPerGate=[] ; edgeOffsets=[] ; drivers =[]; celltypes = []; pinPositions=[]
for logicStage in range(1,len(topo_nodes_cpu1)):
 theseNodes = topo_nodes_cpu1[logicStage]; 
 theseDrivers, dummy =  g1.in_edges( theseNodes ) ; 
 #this roundabout stuff is done to process the case of one driver driving multiple input pins of the same cell
 toTuple = [(int(theseDrivers[i]), int(dummy[i])) for i in range(theseDrivers.size()[0])] ; toTensor = th.LongTensor(list(set(toTuple)))
 dummy2, shuffleIndex = toTensor[:,1].sort() ; theseDrivers2 = toTensor[:,0][shuffleIndex] ; theseNodes2 = th.unique(dummy2) ;
 nodesPerStage.append(cp.asarray(theseNodes2.type(th.int32)));
 celltypes.append(cp.asarray(g1.ndata['celloffsets'][theseNodes2].type(th.int32)));
 in_degs = g1.in_degrees(theseNodes2) ; driversPerGate.append(cp.asarray(in_degs.type(th.uint8)));
 theseEdgeOffsets = th.roll(th.cumsum(in_degs,  dim=0), 1, 0) ; theseEdgeOffsets[0] = 0 ; edgeOffsets.append(cp.asarray(theseEdgeOffsets));
 actualDrivers, notUsed, edgeIDs = g1.edge_ids(theseDrivers2, dummy2, return_uv=True) ; drivers.append(cp.asarray(actualDrivers.type(th.int32)));
 pinPositions.append(cp.asarray(g1.edata['x'][edgeIDs])) ; 
mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()
temp_delta = timer() - temp_start
print("Edited sim graph done in " + f"{temp_delta:.3f}" + ' seconds')

print("start edited simulation...")
temp_start = timer()
for c in range(simLoops):
 currentLogicValue[inputNodes1] = inputsTotal[:,c*PARALLEL_CYCLES:c*PARALLEL_CYCLES+PARALLEL_CYCLES]
 for logicStage in range(len(topo_nodes_cpu1)-1):
  theseNodes = nodesPerStage[logicStage] ; theseCelltypes = celltypes[logicStage]; numDrivers = driversPerGate[logicStage];
  theseDrivers = drivers[logicStage] ; thesePinPositions = pinPositions[logicStage]; theseEdgeOffsets = edgeOffsets[logicStage];
  evalLogic( (1,math.ceil(theseNodes.shape[0]/(512/PARALLEL_CYCLES))), (PARALLEL_CYCLES,(512/PARALLEL_CYCLES)),\
   (currentLogicValue,theseNodes,theseCelltypes,numDrivers,theseDrivers,thesePinPositions,theseEdgeOffsets,\
   out_array_GPU,theseNodes.shape[0],PARALLEL_CYCLES) )
 outputsTotal1[:,c*PARALLEL_CYCLES:c*PARALLEL_CYCLES+PARALLEL_CYCLES] = currentLogicValue[outputs1]
temp_delta = timer() - temp_start
print("Edited simulation for " + str(cycles32) + ' cycles done in ' + f"{temp_delta:.3f}" + ' seconds')

for c in range(PARALLEL_CYCLES):
 A=[] ; B=[]; C=[] ; printA='' ; printB='' ;  printC='' ; 
 for i in range(31,-1,-1):
  aName = 'a' + '[' + str(i) + ']' ; bName = 'b' + '[' + str(i) + ']' ; cName = 'c' + '[' + str(i) + ']' ; 
  bitIDa = port2id1[aName] ;  bitIDb = port2id1[bName] ;bitIDc = port2id1[cName] ;
  A.append(str(int(currentLogicValue[bitIDa,c]))) ; B.append(str(int(currentLogicValue[bitIDb,c]))) ; C.append(str(int(currentLogicValue[bitIDc,c]))) ; 
 A = "".join(A) ; B = "".join(B) ; C = "".join(C) ; 
 printA += 'a' + '[' + str(31) + ':' + str(0) + ']' + " : " + str(hex(int(A, base=2)))
 printB += 'b' + '[' + str(31) + ':' + str(0) + ']' + " : " + str(hex(int(B, base=2)))
 printC += 'c' + '[' + str(31) + ':' + str(0) + ']' + " : " + str(hex(int(C, base=2)))
 print(printA + ' ' + printB + ' : ' + printC) 

print("start result compare...")
temp_start = timer()
if cp.all(outputsTotal1==outputsTotal0):
 print("results match, valid resynth edit")
else:
 print("results don't match, invalid resynth edit")
 wrongIDindex, cycle = cp.where(outputsTotal1!=outputsTotal0)
 wrongID = int(outputs0[wrongIDindex[0]]) ; cycle = int(cycle[0])
 sg, inverse_indices = dgl.khop_in_subgraph(g0, wrongID, k=len(topo_nodes_cpu0))
 wrongIDname = id2pinAndNet0[wrongID-num_of_top_ports0][0] if wrongID >= num_of_top_ports0 else id2port0[wrongID]
 sgInputs = dgl.traversal.topological_nodes_generator(sg)[0]
 wrongInputIDs = sg.ndata['_ID'][sgInputs]
 print("output node " + wrongIDname + ' is incorrect. With inputs:') ; print_string = ''
 for i in wrongInputIDs:
  ii = int(i)
  wrongInputName = id2pinAndNet0[ii-num_of_top_ports0][0] if ii >= num_of_top_ports0 else id2port0[ii]
  inputIndex = int(cp.where(inputNodes0 == ii)[0]) ; wrongInputValue = int(inputsTotal[inputIndex,cycle]) ;
  print_string += wrongInputName + ' = ' + str(wrongInputValue) + '\t'
 print(print_string)
 rightValue = int(outputsTotal0[wrongIDindex[0],cycle]) ; wrongValue = int(outputsTotal1[wrongIDindex[0],cycle])
 print("SHOULD BE: " + str(rightValue) + ' BUT IS: ' + str(wrongValue))
temp_delta = timer() - temp_start
print("Golden vs Edited comparison done in " + f"{temp_delta:.3f}" + ' seconds')

