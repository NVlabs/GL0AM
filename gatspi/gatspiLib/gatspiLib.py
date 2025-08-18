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
sys.path.append(str(cd) + '/gatspiLib/../../saif_dumper')
import saif_dumper
exec(open(str(cd) + '/gatspiLib/GATSPI.cupy').read())


#data loading, builds the DGL graph from csr raw graph
def load_graph(pkl):
 now=datetime.now()
 data = np.load(pkl, allow_pickle=1)
 print("pkl loaded")
 print('start create DGL graph')
 g = dgl.graph(('csr', (data['start'], data['items'], [])))
 g.edata['x'] = th.ByteTensor(data['values']) ; g.edata['SDFPointerStart'] = th.LongTensor(data['SDFPointerStart'])
 g.edata['SDFPointerEnd'] = th.LongTensor(data['SDFPointerEnd']) ; g.edata['interconnectDelays'] = th.IntTensor(data['interconnectDelays'])
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
 SDFLUT = th.IntTensor(data['SDFLUT']) ;
 print("SDF Loaded")
 later=datetime.now()
 delta=(later-now).total_seconds()
 print("creating the DGL graph took " + str(delta) + " seconds on the CPU")
 return g, num_of_gatspi_cells, num_of_top_ports, tempPortDict, driverPin2id, net2id, SDFLUT


def printATimer(functionDescription: str, temp_delta):
 print(functionDescription +  " done in " + f"{temp_delta:.3f}" + ' seconds')


def saifEscapedString(string):
 if(re.search("^\\\d",string)):
  string = re.sub(r'^\\','\\\\', string)
 else:
  string = re.sub(r'^\\','', string)
 return string.replace('[', '\[').replace(']', '\]').replace('/', '\/')


class GATSPI:
 PARALLEL_CYCLES=32 ; END_TOKEN = 2140480647 ; DEVICE = "cuda:0"
 def __init__(self, topName, instanceName, graphFilePath, \
  testname, inputTraceFile, testDuration, period, numOfSubchunks, waveformBufferSize, dumpDGLGraph=0, createStdCellLibLUT=0):
  
  self.topName = topName ; self.instanceName = instanceName ; self.graphFilePath=graphFilePath; self.dumpDGLGraph=dumpDGLGraph; 
  self.createStdCellLibLUT = createStdCellLibLUT; self.testname = testname ; self.inputTraceFile = inputTraceFile;
  self.numOfSubchunks = numOfSubchunks ; self.period = period; self.fold_split = None; 
  self.waveformBufferSize = waveformBufferSize ; self.testDuration = testDuration
  self.THREADS_PER_BLOCK = 128;
  self.g = None ; self.id2pinAndNet = None; self.driverPin2id = None; self.net2id = None; self.SDFLUT = None; self.num_of_gatspi_cells = None;
  self.topo_nodes_cpu = None;  
  self.stdcell_array = None; self.stdcell_offsets = None;
  self.TC_master = None; self.T0s_master = None; self.new_waveforms_total = None;
 
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
   ("OA32", ['B2', 'B1', 'A3', 'A2', 'A1'], "int((bits[0] or bits[1]) and (bits[2] or bits[3] or bits[4]))"), \
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
   ("XOR2", ['B', 'A'], "int(bits[1] ^ bits[0])"), \
   ("XNOR3", ['A3', 'A2', 'A1'], "int(not(bits[2] ^ bits[1] ^ bits[0]))"), \
   ("XOR3", ['A3', 'A2', 'A1'], "int(bits[2] ^ bits[1] ^ bits[0])"), \
   ("AO2222", ['D2', 'D1', 'C2', 'C1', 'B2', 'B1', 'A2', 'A1'], "int((bits[7] and bits[6]) or (bits[5] and bits[4]) or (bits[3] and bits[2]) or (bits[1] and bits[0]))"), \
   ("AOAI211", ['C', 'B', 'A2', 'A1'], "int(not(((bits[3] and bits[2]) or bits[1]) and bits[0]))"), \
   ("OAOI211", ['C', 'B', 'A2', 'A1'], "int(not(((bits[3] or bits[2]) and bits[1]) or bits[0]))"), \
   ("IAND2", ['B1', 'A1'], "int(bits[0] and (not bits[1]))"), \
   ("IAOI21", ['B', 'A2', 'A1'], "int(not((bits[2] and bits[1]) or (not bits[0])))"), \
   ("IAOI22", ['B2', 'B1', 'A2', 'A1'], "int((not(bits[1] and bits[0])) and (bits[3] or bits[2]))"), \
   ("IBAO21", ['B', 'A2', 'A1'], "int((bits[2] and bits[1]) or (not bits[0]))"), \
   ("IBOA21", ['B', 'A2', 'A1'], "int((bits[2] or bits[1]) and (not bits[0]))"), \
   ("INOR2", ['B1', 'A1'], "int(not((not bits[1]) or bits[0]))"), \
   ("INAND2", ['B1', 'A1'], "int(not((not bits[1]) and bits[0]))"), \
   ("INOR3", ['B2', 'B1', 'A1'], "int(not((not bits[2]) or bits[1] or bits[0]))"), \
   ("INOR4", ['B3', 'B2', 'B1', 'A1'], "int(not((not bits[3]) or bits[2] or bits[1] or bits[0]))"), \
   ("IIOAI21", ['B', 'A2', 'A1'], "int(not(((not bits[2]) or (not bits[1])) and bits[0]))"), \
   ("IIOAI22", ['B2', 'B1', 'A2', 'A1'], "int(not((not(bits[3] and bits[2])) and (bits[1] or bits[0])))"), \
   ("IBOAI21", ['B', 'A2', 'A1'], "int(not((bits[2] or bits[1]) and (not bits[0])))"), \
   ("IOR2", ['B1', 'A1'], "int(bits[0] or (not bits[1]))"), \
   ("FAS", ['CI', 'B', 'A'], "int(bits[0] ^ bits[1] ^ bits[2])"), \
   ("FACO", ['CI', 'B', 'A'], "int((bits[0] and bits[1]) or (bits[0] and bits[2]) or (bits[1] and bits[2]))"), \
   ("HAS", ['B', 'A'], "int(bits[1] ^ bits[0])"), \
   ("HACO", ['B', 'A'], "int(bits[1] and bits[0])"), \
   ("MAJORITYAOI222", ['C', 'B', 'A'], "int(not((bits[1] and bits[0]) or (bits[2] and bits[0]) or (bits[1] and bits[2])))"), \
   ("MAJORITYAOI22", ['B2', 'B1', 'A2', 'A1'], "int((not (bits[3] and bits[2])) and (bits[1] or bits[0]))"), \
   ("MAJORITYOAI22", ['B2', 'B1', 'A2', 'A1'], "int((not (bits[3] or bits[2])) or (bits[1] and bits[0]))"), \
   ("MUX2", ['S', 'I1', 'I0'], "int((bits[0] and bits[1]) or ((not bits[0]) and bits[2]))"), \
   ("MUX2N", ['S', 'I1', 'I0'], "int(not((bits[0] and bits[1]) or ((not bits[0]) and bits[2])))"), \
   ("MUX3", ['S1', 'S0', 'I2', 'I1', 'I0'], "int((bits[0] and bits[2]) or ((not bits[0]) and ((bits[1] and bits[3]) or ((not bits[1]) and bits[4]))))"), \
   ("MUX3N", ['S1', 'S0', 'I2', 'I1', 'I0'], "int(not((bits[0] and bits[2]) or ((not bits[0]) and ((bits[1] and bits[3]) or ((not bits[1]) and bits[4])))))"), \
   ("MUX4", ['S1', 'S0', 'I3', 'I2', 'I1', 'I0'], "int((bits[0] and ((bits[1] and bits[2]) or ((not bits[1]) and bits[3]))) or ((not bits[0]) and ((bits[1] and bits[4]) or ((not bits[1]) and bits[5]))))"), \
   ("MUX4N", ['S1', 'S0', 'I3', 'I2', 'I1', 'I0'], "int(not((bits[0] and ((bits[1] and bits[2]) or ((not bits[1]) and bits[3]))) or ((not bits[0]) and ((bits[1] and bits[4]) or ((not bits[1]) and bits[5])))))") ]
   
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
   th.save( (out_array, out_offset), "GATSPIStdCellLibLUT")
   print("std cell lib LUT done...")
   self.stdcell_array = out_array ; self.stdcell_offsets = out_offset ;
  else:
   self.stdcell_array, self.stdcell_offsets = th.load("GATSPIStdCellLibLUT")
  self.stdcell_array_GPU = cp.asarray(self.stdcell_array)
 
 
 def build_graphs(self):
  if self.dumpDGLGraph:
   print("creating the DGL graph from raw CSR graph...")
   temp_start = timer() ;
   self.g, self.num_of_gatspi_cells, num_of_top_ports, self.id2pinAndNet, self.driverPin2id, self.net2id, self.SDFLUT = load_graph(self.graphFilePath)
   fileObject = open(self.topName + "_DGLGraph", 'wb')
   pickle.dump({'g': self.g,
    'num_of_gatspi_cells' : self.num_of_gatspi_cells,
    'num_of_top_ports' : num_of_top_ports,
    'id2pinAndNet' : self.id2pinAndNet,
    'driverPin2id' : self.driverPin2id,
    'net2id' : self.net2id,
    'SDFLUT' : self.SDFLUT}, fileObject)
   temp_delta = timer() - temp_start
   printATimer("DGL graph", temp_delta)
  else:
   print("Loading the DGL graph from disk...")
   temp_start = timer() ;
   data = np.load(self.graphFilePath, allow_pickle=True);
   self.g = data['g']; self.num_of_gatspi_cells = data['num_of_gatspi_cells'] ; num_of_top_ports = data['num_of_top_ports'] ; 
   self.id2pinAndNet = data['id2pinAndNet'] ; self.driverPin2id = data['driverPin2id'] ; self.net2id = data['net2id'] ;
   self.SDFLUT = data['SDFLUT']
   temp_delta = timer() - temp_start
   printATimer("DGL graph", temp_delta)
  self.SDFLUT = cp.asarray(self.SDFLUT)
 
 def graph2GPU(self):
  self.g = self.g.to(self.DEVICE)
 
 def loadWaveforms(self):
  temp_start = timer() ; print("Loading the input/pseudo-input waveform pkl file...")
  waveforms = np.load(self.inputTraceFile,allow_pickle=1)
  waveforms = waveforms['waveforms'] ;
  self.fold_split = math.ceil( self.testDuration / (self.PARALLEL_CYCLES*self.numOfSubchunks*self.period) ) *self.period
  assert self.testDuration >= (self.numOfSubchunks-1)*self.PARALLEL_CYCLES*self.fold_split, "Lower number of subchunks, else test duration will OVERFLOW"
  print("Folding the test duration into " + str(self.PARALLEL_CYCLES) + " parallel cycles. Each window will simulate: " + str(self.fold_split) + " time units.")
  waveforms["1'b1"] = np.array([-1,0])
  waveforms["1'b0"] = np.array([0])
  self.g.ndata['waveform_start'] = th.LongTensor( [2*x for x in range(self.PARALLEL_CYCLES)] ).unsqueeze(0).repeat(len(self.g.nodes()), 1).type(th.int64)
  self.g.ndata['waveform_end'] = th.LongTensor( [2*x+2 for x in range(self.PARALLEL_CYCLES)] ).unsqueeze(0).repeat(len(self.g.nodes()), 1).type(th.int64)
  input_w = [] ; node_nums = [] ; waveform_lengths = [0]; new_length =0;
  for pin in list( waveforms.keys() ):
   if pin[1].isalpha():
    adjusted_pinname=re.sub(r'^\\','', pin)
   else:
    adjusted_pinname = pin
   if adjusted_pinname in self.driverPin2id.keys():
    node_nums.append(self.driverPin2id[adjusted_pinname])
    input_w.append(th.LongTensor(waveforms[pin]))
    new_length +=waveforms[pin].shape[0]
    waveform_lengths.append(new_length)
  node_nums = th.LongTensor(node_nums) #node_nums still on the CPU for now
  input_w = th.cat(input_w) #input_w still on the CPU for now
  input_waveform_length_start_pointers=th.LongTensor(waveform_lengths[0:-1], device="cpu")
  input_waveform_length_end_pointers=th.LongTensor(waveform_lengths[1:], device="cpu")
  temp_delta = timer() - temp_start
  printATimer("Initial waveform loading", temp_delta)
  return node_nums, input_w, input_waveform_length_start_pointers, input_waveform_length_end_pointers
 
 def prepWaveformOneSubchunk(self,node_nums,input_w,input_waveform_length_start_pointers,input_waveform_length_end_pointers,subchunkID):
  temp_start = timer() ;
  print("Prepping waveforms for subchunk " + str(subchunkID) )
  if self.new_waveforms_total is not None:
   del self.new_waveforms_total ; self.new_waveforms_total = None;
   th.cuda.empty_cache() ;  mempool = cp.get_default_memory_pool(); mempool.free_all_blocks() ;
  input_waveform_length_start_pointers=cp.asarray(input_waveform_length_start_pointers)
  input_waveform_length_end_pointers=cp.asarray(input_waveform_length_end_pointers)
  input_w=cp.asarray(input_w)
  split_waveform_lengths = cp.zeros( (node_nums.size()[0], self.PARALLEL_CYCLES) , dtype=cp.int64 );
  cudaBlockY = math.ceil(128/self.PARALLEL_CYCLES) ;
  determineSplitWaveformSizes( (math.ceil(node_nums.size()[0]/cudaBlockY),1), (self.PARALLEL_CYCLES,cudaBlockY),\
   (input_w,input_waveform_length_start_pointers,input_waveform_length_end_pointers,split_waveform_lengths,\
   subchunkID,self.PARALLEL_CYCLES,self.END_TOKEN,self.fold_split,node_nums.size()[0]) ) 
  split_waveform_lengths = cp.roll(cp.cumsum(split_waveform_lengths.reshape(-1)), 1, 0)
  organized_waveforms_size = int(split_waveform_lengths[0].repeat(1)) ; split_waveform_lengths[0] = 0;
  split_waveform_lengths = split_waveform_lengths.reshape(-1,self.PARALLEL_CYCLES)
  organized_waveforms = cp.full( organized_waveforms_size, self.END_TOKEN, dtype=cp.int32)
  reorganizeWaveform( (math.ceil(node_nums.size()[0]/cudaBlockY),1), (self.PARALLEL_CYCLES,cudaBlockY),\
   (input_w,input_waveform_length_start_pointers,input_waveform_length_end_pointers,split_waveform_lengths,organized_waveforms,\
   subchunkID,self.PARALLEL_CYCLES,self.END_TOKEN,self.fold_split,node_nums.size()[0]) ) 
  organized_waveforms = cp.concatenate((cp.asarray( [0,self.END_TOKEN] * self.PARALLEL_CYCLES ).astype(cp.int32), organized_waveforms ))
  self.g.ndata['waveform_start'][node_nums] = th.LongTensor(split_waveform_lengths.get() + (2 * self.PARALLEL_CYCLES))
  self.g.ndata['waveform_end'][node_nums] = th.LongTensor( (2 * self.PARALLEL_CYCLES) + \
   cp.concatenate( (split_waveform_lengths.reshape(-1)[1:], cp.array([organized_waveforms_size]).astype(cp.int64)) ).reshape(-1,self.PARALLEL_CYCLES).get() )
  input_waveform_length_start_pointers=th.LongTensor(input_waveform_length_start_pointers.get()) ; 
  input_waveform_length_end_pointers=th.LongTensor(input_waveform_length_end_pointers.get()) ; 
  input_w=th.LongTensor(input_w.get()) ; del split_waveform_lengths ;
  th.cuda.empty_cache();   mempool = cp.get_default_memory_pool(); mempool.free_all_blocks() ;
  self.new_waveforms_total = cp.full( self.waveformBufferSize, self.END_TOKEN, dtype=cp.int32 )
  self.new_waveforms_total[0:organized_waveforms.shape[0]] = organized_waveforms
  append_length = int(organized_waveforms.shape[0]) ; 
  del organized_waveforms ; 
  temp_delta = timer() - temp_start
  printATimer("Prepping waveforms", temp_delta)
  return append_length
 
 def cudaArrayfyGraph(self):
  temp_start = timer()
  print("Translating DGL graph to CUDA arrays... " )
  try:
   topo_nodes = dgl.traversal.topological_nodes_generator(self.g)
  except:
   print("Error during DGL topo sort traversal, a loop detected. Ensure your logic has no combinational loops") ; exit(1)
  topo_nodes = list(topo_nodes) ; self.g.ndata['logicLevel'] = th.zeros( len(self.g.nodes()), dtype = th.int16 )
  self.g.ndata['celltype'][self.g.ndata['celltype'] == 999] = 28
  self.g.ndata['celloffsets'] = (self.stdcell_offsets[self.g.ndata['celltype'].type(th.int32)])
  nodesPerStage=[]; driversPerGate=[] ; edgeOffsets=[] ; drivers =[]; celltypes = []; pinPositions=[] ; netDelays=[];
  delayPointersStart = [] ; delayPointersEnd = []
  for logicStage in range(1,len(topo_nodes)):
   theseNodes = topo_nodes[logicStage]; 
   self.g.ndata['logicLevel'][theseNodes] = logicStage 
   theseDrivers, dummy =  self.g.in_edges( theseNodes ) ; 
   #this roundabout stuff is done to process the case of one driver driving multiple input pins of the same cell
   toTuple = [(int(theseDrivers[i]), int(dummy[i])) for i in range(theseDrivers.size()[0])] ; toTensor = th.LongTensor(list(set(toTuple)))
   dummy2, shuffleIndex = toTensor[:,1].sort() ; theseDrivers2 = (toTensor[:,0][shuffleIndex]) ; theseNodes2 = th.unique(dummy2) ;
   nodesPerStage.append(cp.asarray(theseNodes2.type(th.int32)));
   celltypes.append(cp.asarray(self.g.ndata['celloffsets'][theseNodes2].type(th.int32)));
   in_degs = self.g.in_degrees(theseNodes2) ; driversPerGate.append(cp.asarray(in_degs.type(th.uint8)));
   theseEdgeOffsets = th.roll(th.cumsum(in_degs,  dim=0), 1, 0) ; theseEdgeOffsets[0] = 0 ; edgeOffsets.append(cp.asarray(theseEdgeOffsets).astype(cp.int32));
   actualDrivers, notUsed, edgeIDs = self.g.edge_ids(theseDrivers2, dummy2, return_uv=True) ; drivers.append(cp.asarray(actualDrivers.type(th.int32)));
   pinPositions.append(cp.asarray(self.g.edata['x'][edgeIDs])) ; 
   netDelays.append(cp.asarray(self.g.edata['interconnectDelays'][edgeIDs]))
   delayPointersStart.append(cp.asarray(self.g.edata['SDFPointerStart'][edgeIDs]))
   delayPointersEnd.append(cp.asarray(self.g.edata['SDFPointerEnd'][edgeIDs]))
  temp_delta = timer() - temp_start
  printATimer('Graph --> cuda arrays', temp_delta)
  return nodesPerStage,driversPerGate,edgeOffsets,drivers,celltypes,pinPositions,netDelays,delayPointersStart,delayPointersEnd
 
 def gatspiSimASubchunk(self,append_length,\
  nodesPerStage,driversPerGate,edgeOffsets,drivers,celltypes,pinPositions,netDelays,delayPointersStart,delayPointersEnd):
  temp_start = timer() ; print("Simulating subchunk...")
  logicStages = len(nodesPerStage)
  for logicStage in range(logicStages):
   theseNodes=nodesPerStage[logicStage]; theseCelltypes=celltypes[logicStage]; numDrivers=driversPerGate[logicStage];
   theseDrivers=drivers[logicStage] ; thesePinPositions=pinPositions[logicStage]; theseEdgeOffsets=edgeOffsets[logicStage];
   theseNetDelays=netDelays[logicStage] ; theseDelayPointersStart=delayPointersStart[logicStage]; theseDelayPointersEnd=delayPointersEnd[logicStage]; 
   waveformPointers = cp.asarray(self.g.ndata['waveform_start'])[theseDrivers]   
   output_TC=cp.zeros( ( theseNodes.shape[0], self.PARALLEL_CYCLES ) , dtype=cp.int32 )
   init_vals=cp.full( ( theseNodes.shape[0], self.PARALLEL_CYCLES), -1, dtype=cp.int32 )
   simulateGateTC( ( math.ceil(theseNodes.shape[0]/(self.THREADS_PER_BLOCK/self.PARALLEL_CYCLES)),1 ), (self.PARALLEL_CYCLES,(self.THREADS_PER_BLOCK/self.PARALLEL_CYCLES)),\
    (self.new_waveforms_total,self.SDFLUT,self.stdcell_array_GPU,numDrivers,theseNetDelays,thesePinPositions,\
    theseDelayPointersStart,theseDelayPointersEnd,theseCelltypes,theseEdgeOffsets,waveformPointers,init_vals,output_TC,\
    theseNodes.shape[0],self.PARALLEL_CYCLES,self.END_TOKEN) )
   waveformPointers = cp.asarray(self.g.ndata['waveform_start'])[theseDrivers]
   temp= (output_TC + init_vals + 2 + ((output_TC + init_vals)%2)).astype(cp.int64)
   output_pointers=( cp.roll( cp.cumsum(temp.reshape(-1)), 1, 0 ) ).reshape(-1, self.PARALLEL_CYCLES)
   stage_length = int(output_pointers[0,0]) ; output_pointers[0,0] = 0
   self.g.ndata['waveform_start'][th.IntTensor(cp.asnumpy(theseNodes))] = th.LongTensor(cp.asnumpy(output_pointers +append_length))
   append_length += stage_length
   if ( append_length >= self.waveformBufferSize) :
    print("max pointer value reached! Consider a bigger subchunk number. exiting..." + str(append_length) )
    sys.exit()
   output_pointers = cp.asarray(self.g.ndata['waveform_start'][th.IntTensor(cp.asnumpy(theseNodes))])
   simulateGateWaveforms( ( math.ceil(theseNodes.shape[0]/(self.THREADS_PER_BLOCK/self.PARALLEL_CYCLES)),1 ), (self.PARALLEL_CYCLES,(self.THREADS_PER_BLOCK/self.PARALLEL_CYCLES)),\
    (self.new_waveforms_total,self.SDFLUT,self.stdcell_array_GPU,numDrivers,theseNetDelays,thesePinPositions,\
    theseDelayPointersStart,theseDelayPointersEnd,theseCelltypes,theseEdgeOffsets,waveformPointers,output_pointers,\
    theseNodes.shape[0],self.PARALLEL_CYCLES,self.END_TOKEN) )
   self.g.ndata['waveform_end'][th.IntTensor(cp.asnumpy(theseNodes))] = th.LongTensor(cp.asnumpy(output_pointers + 2 + output_pointers%2))
  temp_delta = timer() - temp_start
  printATimer('One Subchunk Simulation', temp_delta)
  return append_length
 
 def update_saif(self, subchunkID):
  temp_start = timer() ;
  if subchunkID==0:
   self.TC_master=cp.zeros( len(self.g.nodes()) , dtype=cp.uint32 ) ; 
   self.T0s_master = cp.zeros( len(self.g.nodes()) , dtype=cp.uint64  ) ; 
  if subchunkID == (self.numOfSubchunks - 1):
   subchunk_duration = int( self.testDuration - ( int(self.fold_split) * self.PARALLEL_CYCLES * subchunkID ) )
  else:
   subchunk_duration = int( self.PARALLEL_CYCLES * int(self.fold_split) )
  print("subchunk " + str(subchunkID) + " duration is " + str(subchunk_duration))
  subchunkStart = int(self.fold_split) * self.PARALLEL_CYCLES * subchunkID ; subchunkEnd = subchunkStart + subchunk_duration ;
  cudaBlockY = math.ceil(128/self.PARALLEL_CYCLES) ; 
  TCs = cp.zeros( (len(self.g.nodes()),self.PARALLEL_CYCLES) , dtype=cp.uint32 ) ; 
  T0s = cp.zeros( (len(self.g.nodes()),self.PARALLEL_CYCLES) , dtype=cp.uint64 ) ;
  calculateSAIF( (math.ceil(self.TC_master.shape[0]/cudaBlockY),1), (self.PARALLEL_CYCLES,cudaBlockY), \
   (self.new_waveforms_total,cp.asarray(self.g.ndata['waveform_start']),cp.asarray(self.g.ndata['waveform_end']),\
   T0s,TCs,self.PARALLEL_CYCLES,len(self.g.nodes()),self.fold_split,subchunk_duration,subchunkID,self.END_TOKEN) )
  T0s_temp = cp.sum(T0s, axis=1) ; TCs_temp =cp.sum(TCs, axis=1) ;
  self.TC_master = self.TC_master + TCs_temp.astype(cp.uint32) ; self.T0s_master = self.T0s_master + T0s_temp ;
  temp_delta = timer() - temp_start
  printATimer("Updating SAIF values for subchunk " + str(subchunkID), temp_delta)
 
 def simAllSubchunks(self,node_nums,input_w,input_waveform_length_start_pointers,input_waveform_length_end_pointers,\
  nodesPerStage,driversPerGate,edgeOffsets,drivers,celltypes,pinPositions,netDelays,delayPointersStart,delayPointersEnd):
  for chunk_id in range(self.numOfSubchunks):
   append_length=self.prepWaveformOneSubchunk(node_nums,input_w,input_waveform_length_start_pointers,input_waveform_length_end_pointers,chunk_id)
   self.gatspiSimASubchunk(append_length,nodesPerStage,driversPerGate,edgeOffsets,drivers,celltypes,pinPositions,netDelays,delayPointersStart,delayPointersEnd)
   self.update_saif(chunk_id)
  print("ALL subchunks done simulating...")
 
 def dumpSAIF(self):
  temp_start = timer() ; print("Starting SAIF file dump...")
  dumper = saif_dumper.PySaifDumper()
  dumper.load_from_dict(self.id2pinAndNet)
  dumper.create_saif_file(self.topName, self.testname, self.testDuration, self.instanceName,self.TC_master.get(),self.T0s_master.get())
  temp_delta = timer() - temp_start
  printATimer("SAIF file dump", temp_delta)
 
 
 def doEverything(self):
  self.build_stdcell_lib()
  self.build_graphs()
  node_nums, input_w, input_waveform_length_start_pointers, input_waveform_length_end_pointers = self.loadWaveforms()
  nodesPerStage,driversPerGate,edgeOffsets,drivers,celltypes,pinPositions,netDelays,delayPointersStart,delayPointersEnd=\
   self.cudaArrayfyGraph()
  self.simAllSubchunks(node_nums,input_w,input_waveform_length_start_pointers,input_waveform_length_end_pointers,\
  nodesPerStage,driversPerGate,edgeOffsets,drivers,celltypes,pinPositions,netDelays,delayPointersStart,delayPointersEnd) 
  self.dumpSAIF()
