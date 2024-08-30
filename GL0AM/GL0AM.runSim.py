import torch as th
import dgl
from dgl import DGLGraph
import pickle
import numpy as np
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
import logging


def update_saif():
  global TC_master, IG_master, T0s_master, T1s_master, bg, listOfClkTreeComponents, pushoutDelaysClkTreeRise, pushoutDelaysClkTreeFall
  g = bg;  
  start_saif_dump = timer()
  SAIF_duration = int(args.duration)
  if subchunk_id ==0:
    TC_master=cp.zeros( ( len(g.nodes()) ) , dtype=cp.uint32) ;  IG_master=cp.zeros( ( len(g.nodes()) ) , dtype=cp.uint32)
  #the TC count for the level 0 nodes
  T0T1TXTimes = cp.zeros( ( len(g.nodes()), args.num_splits, 5 ) , dtype=cp.uint64)
  #look up saif T0s
  start_saif_t0 = timer()
  if subchunk_id==0:
    T0s_master=cp.zeros(TC_master.shape[0], dtype=cp.uint64) ; T1s_master=cp.zeros(TC_master.shape[0], dtype=cp.uint64)
  if subchunk_id == (args.num_of_sub_chunks - 1):
    subchunk_duration = int( SAIF_duration - ( int(fold_split) * args.num_splits * subchunk_id ) )
  else:
    subchunk_duration = int( args.num_splits * int(fold_split) )
  logging.info("subchunk duration is " + str(subchunk_duration))
  if subchunk_id == (args.num_of_sub_chunks - 1):
    subchunkStart = int(fold_split) * args.num_splits * subchunk_id ; 
    calculateSAIFLastSubchunk( (math.ceil(T0T1TXTimes.shape[0]/16),1), (512,1), \
     (new_waveforms_total,cp.asarray(bg.ndata['waveform_start']),cp.asarray(bg.ndata['waveform_end']),T0T1TXTimes,\
     args.num_splits,T0T1TXTimes.shape[0],fold_split,subchunk_id,SAIF_duration, subchunkStart ) )
  else:
    calculateSAIF( (math.ceil(T0T1TXTimes.shape[0]/16),1), (512,1), \
     (new_waveforms_total,cp.asarray(bg.ndata['waveform_start']),cp.asarray(bg.ndata['waveform_end']),T0T1TXTimes,\
     args.num_splits,T0T1TXTimes.shape[0],fold_split,subchunk_duration,subchunk_id ) ) 
  subchunkStart = int(fold_split) * args.num_splits * subchunk_id ; subchunkEnd = subchunkStart + subchunk_duration ;
  T0s_temp = cp.sum(T0T1TXTimes[:,:,0], axis=1) ; T1s_temp = cp.sum(T0T1TXTimes[:,:,1], axis=1); 
  TCs_temp =cp.sum(T0T1TXTimes[:,:,3], axis=1) ;
  if clkTreeNodeNums.shape[0] >0:
    pushoutRiseTemp = cp.asarray(bg2.ndata['seq_pushoutSDF'][th.cuda.LongTensor(clkTreeNodeNums),0]).astype(cp.uint32)
    pushoutFallTemp = cp.asarray(bg2.ndata['seq_pushoutSDF'][th.cuda.LongTensor(clkTreeNodeNums),1]).astype(cp.uint32)
    pushoutDelaysAdjust= cp.zeros_like(pushoutRiseTemp).astype(cp.uint32)
    setPushoutDelays( (math.ceil(pushoutDelaysAdjust.shape[0]/512),1), (512,1), \
     (pushoutRiseTemp,pushoutFallTemp,pushoutDelaysAdjust,pushoutDelaysAdjust.shape[0]) )
    del pushoutRiseTemp; del pushoutFallTemp;
    if subchunk_id == (args.num_of_sub_chunks - 1):
      adjustClkTreeSAIFLastSubchunk( (math.ceil(clkTreeNodeNums.shape[0]/512),1), (512,1),\
      (clkTreeNodeNums,pushoutDelaysAdjust,T0s_temp,T1s_temp,TCs_temp,clkTreeNodePolarities,new_waveforms_total,\
      cp.asarray(bg.ndata['waveform_start'][th.cuda.LongTensor(clkTreeNodeNums)]),\
      cp.asarray(bg.ndata['waveform_end'][th.cuda.LongTensor(clkTreeNodeNums)]),\
      clkTreeNodeNums.shape[0],args.period,args.first_edge,args.num_splits,subchunk_duration))
    else:
      adjustClkTreeSAIF( (math.ceil(clkTreeNodeNums.shape[0]/512),1), (512,1),\
      (clkTreeNodeNums,pushoutDelaysAdjust,T0s_temp,T1s_temp,TCs_temp,clkTreeNodePolarities,new_waveforms_total,\
      cp.asarray(bg.ndata['waveform_start'][th.cuda.LongTensor(clkTreeNodeNums)]),\
      cp.asarray(bg.ndata['waveform_end'][th.cuda.LongTensor(clkTreeNodeNums)]),\
      clkTreeNodeNums.shape[0],args.period,args.first_edge,args.num_splits,subchunk_duration))
  T0s_master = T0s_master + T0s_temp ; T1s_master = T1s_master + T1s_temp;
  TC_master = TC_master + TCs_temp.astype(cp.uint32)
  IG_master = IG_master + cp.sum(T0T1TXTimes[:,:,4], axis=1).astype(cp.uint32)
  dt=timer() - start_saif_t0
  logging.info("Total time T0/T1 lookup for one subchunk took %f s", dt)
  dt=timer() - start_saif_dump
  logging.info("SAIF data update for one subchunk complete in %f s",dt)


def dump_saif():
  global TC_master, IG_master, T0s_master, T1s_master;
  global print_string
  start_saif_dump = timer()
  SAIF_duration = int(args.duration)
  #dump the SAIF files resulting here
  #here we dump the sim results into something more tractable than a 2D array in Python, like a .saif file
  #create saif file.
  #this file is created as a byproduct of creating the traces file
  #it houses all the output pin / net name pairs in the netlist, so the resulting file will be based on net names only
  #and we won't have to list out all the cell instances

  saif_file = "outGL0AM." + args.testname + ".saif.gz"
  printout = gzip.open(saif_file, "wb")
  #SAIF file header
  print_string = """(SAIFILE
  (SAIFVERSION "2.0")
  (DIRECTION "backward")
  (DESIGN )
  (DATE "not important")
  (VENDOR "GL0AM OPEN SOURCE")
  (PROGRAM_NAME "GL0AM")
  (VERSION "1.0")
  (DIVIDER / )
  (TIMESCALE 1 ps)
  (DURATION {})
  (INSTANCE tb_top
     (INSTANCE {}
        (NET
  """.format(SAIF_duration, args.block + "__" + args.instance_name)
  printout.write(bytes(print_string, 'utf-8'))

  instance_print_string='' ; print_string = ''
  TC_master = TC_master.get()
  IG_master = IG_master.get()
  T0s_master = T0s_master.get()
  T1s_master = T1s_master.get()
  #there's a connection from output net to output port. the output ports aren't simulated, just the output nets.
  for cellID in range( len(bg.nodes()) -2 ):
    if cellID % 100000 ==0:
      print(str(cellID) + ' cells processed for SAIF...')
    if ( cellID in port_names ): #check if it's an output port, which we skip. we print the driver, not the o-port itself
      if cell_dict[cell_names[cellID]] != cell_names[cellID]:
        continue
    if (cell_names[cellID] in cell_dict2):
      NET = (cell_dict2[cell_names[cellID]]).replace('[', '\[').replace(']', '\]').replace('/', '\/')
      TCi = int(TC_master[cellID]) ; IGi = int(IG_master[cellID]) ; T0i = int(T0s_master[cellID]) ; T1i = int(T1s_master[cellID]);
      print_string +=("""         ({}
            (T0 {}) (T1 {}) (TX {}) (TZ 0)
            (TC {}) (IG {})
         )
""".format(NET, T0i, T1i, SAIF_duration - T0i- T1i, TCi, IGi))
    elif 'vvd_' not in cell_names[cellID]:
      TCi = int(TC_master[cellID]) ; IGi = int(IG_master[cellID]) ; T0i = int(T0s_master[cellID]) ; T1i = int(T1s_master[cellID]);
      INSTANCE, PINNAME = cell_names[cellID].rsplit('/',1)
      instance_print_string +="""      (INSTANCE {}
         (NET
            ({}
               (T0 {}) (T1 {}) (TX {}) (TZ 0)
               (TC {}) (IG {})
            )
         )
      )
""".format(INSTANCE, PINNAME,T0i,T1i,SAIF_duration - T0i- T1i, TCi, IGi)
  print_string += """      )
{}
   )
)
)""".format(instance_print_string)
  printout.write(bytes(print_string, 'utf-8'))
  printout.close()
  dt=timer() - start_saif_dump
  logging.info("SAIF dump complete in: " + str(dt))
    

def reportDelays(outputPinName):
  debug_node_num = cell_names[outputPinName]
  in_pins, in_edges2 = bg.in_edges(debug_node_num);
  inEdgeIds = bg.edge_ids(in_pins,in_edges2)
  pinPositions = bg.edata['x'][inEdgeIds]
  pinPositions, pinPositionIndexes = th.sort(pinPositions, descending=True)
  in_pins = in_pins[pinPositionIndexes] ; inEdgeIds = bg.edge_ids(in_pins,in_edges2)
  debug_celltype = int(bg.ndata['cell_type'][debug_node_num])
  cellPinNames = cells_list[debug_celltype][1]
  print_string = ''
  print_string += 'Pin Weight Order is: ' 
  print_string += ' '.join(cellPinNames) + ' for ' + outputPinName + '\n'
  for i in range(len(in_pins)):
    thisInputPinName = cellPinNames[i]
    print_string += 'Delays for arc ' + thisInputPinName + '--> ' + outputPinName + ' :\n'
    thisArcFullDelays=full_delays[bg.edata['start_pointers'][inEdgeIds[i]]:bg.edata['end_pointers'][inEdgeIds[i]]]
    subArcLength = int(thisArcFullDelays.shape[0]/4)
    print_string += 'Posedge ' + thisInputPinName + ' --> Posedge ' + outputPinName + ': ' + str(thisArcFullDelays[0:subArcLength]) + '\n'
    print_string += 'Negedge ' + thisInputPinName + ' --> Posedge ' + outputPinName + ': ' + str(thisArcFullDelays[subArcLength:2*subArcLength]) + '\n'
    print_string += 'Posedge ' + thisInputPinName + ' --> Negedge ' + outputPinName + ': ' + str(thisArcFullDelays[2*subArcLength:3*subArcLength]) + '\n'
    print_string += 'Negedge ' + thisInputPinName + ' --> Negedge ' + outputPinName + ': ' + str(thisArcFullDelays[3*subArcLength:4*subArcLength]) + '\n'
    print_string += '\n'
  print(print_string)

def reconstruct_whole_waveform(signal_name):
  global debug_node_num,waveformPointersStart,waveformPointersEnd,thisSignalGATSPISubchunkWaves ;
  debug_node_num = cell_names[cell_dict[signal_name]] ; 
  for subchunk_id in range(args.num_of_sub_chunks):
    command = 'waveformPointersStart = saveSubchunkPointersStart_' + str(subchunk_id) + '[debug_node_num]' ; exec(command, globals()) ;
    command = 'waveformPointersEnd = saveSubchunkPointersEnd_' + str(subchunk_id) + '[debug_node_num]' ; exec(command, globals()) 
    thisNodeWaveformLength= int(waveformPointersEnd[args.num_splits-1] - waveformPointersStart[0] - 2*args.num_splits + 1)
    perNodeWholeWaveformPointers = th.LongTensor([0,thisNodeWaveformLength])
    wholeWaveformTempStorage = cp.zeros( (int(thisNodeWaveformLength),), dtype=cp.uint64)
    perNodePerSplitWaveformPointerStart = cp.asarray(waveformPointersStart - waveformPointersStart[0])
    perNodePerSplitWaveformPointerEnd = cp.asarray(waveformPointersEnd - waveformPointersStart[0])
    perNodeWholeWaveformPointers= cp.asarray(perNodeWholeWaveformPointers).astype(cp.uint32)
    T0T1TXTimes = cp.zeros( (1,args.num_splits,5), dtype=cp.uint64)
    command = 'thisSignalGATSPISubchunkWaves = cp.asarray(saveSubchunkWaveforms_' + str(subchunk_id) + '[waveformPointersStart[0]:waveformPointersEnd[-1]])' ; exec(command, globals())
    thisSubchunkDuration = subchunkEndTimes[subchunk_id] - subchunkStartTimes[subchunk_id]
    reconstructWholeWaveforms( (1,1) , (32,1), \
    (thisSignalGATSPISubchunkWaves,wholeWaveformTempStorage,perNodeWholeWaveformPointers,perNodePerSplitWaveformPointerStart,\
    perNodePerSplitWaveformPointerEnd,T0T1TXTimes,args.num_splits,1,fold_split,thisSubchunkDuration,subchunk_id) )
    T0T1TXTimes = cp.sum(T0T1TXTimes, axis=1)
    if subchunk_id == 0:
      masterWaveforms = wholeWaveformTempStorage ; masterT0T1TXTimes = T0T1TXTimes.repeat(1).reshape(5) ; 
    else:
      wholeWaveformTempStorage += subchunkStartTimes[subchunk_id]
      masterWaveforms = cp.concatenate( (masterWaveforms,wholeWaveformTempStorage[1:]), axis=0)
      masterT0T1TXTimes = masterT0T1TXTimes + T0T1TXTimes.reshape(5) ;
  print_stuff = signal_name + ' SAIF info:' + '\n'
  print_stuff += '(T0 '
  print_stuff += str(int(masterT0T1TXTimes[0])) + ') (T1 ' + str(int(masterT0T1TXTimes[1])) + ') (TX ' + str(int(masterT0T1TXTimes[2])) + ') '
  print_stuff += '(TC ' + str(int(masterT0T1TXTimes[3])) + ') (IG ' + str(int(masterT0T1TXTimes[4])) + ')'    
  print(print_stuff)
  return masterWaveforms


def reportSAIFInTimerange(signal_name, timeStart, timeEnd):
  global debug_node_num,waveformPointersStart,waveformPointersEnd,thisSignalGATSPISubchunkWaves  ;
  debug_node_num = cell_names[cell_dict[signal_name]] ; neededSubchunks = []
  for i in range(len(subchunkStartTimes)):
    if ((timeStart >= subchunkStartTimes[i]) and (timeEnd <= subchunkEndTimes[i])) or ((timeStart < subchunkEndTimes[i]) and (timeEnd >= subchunkEndTimes[i])) or ((timeStart <= subchunkStartTimes[i]) and (timeEnd > subchunkStartTimes[i])) :
       neededSubchunks.append(i)
  for subchunk_id in neededSubchunks:
    command = 'waveformPointersStart = saveSubchunkPointersStart_' + str(subchunk_id) + '[debug_node_num]' ; exec(command, globals()) ;
    command = 'waveformPointersEnd = saveSubchunkPointersEnd_' + str(subchunk_id) + '[debug_node_num]' ; exec(command, globals()) 
    perNodePerSplitWaveformPointerStart = cp.asarray(waveformPointersStart - waveformPointersStart[0])
    perNodePerSplitWaveformPointerEnd = cp.asarray(waveformPointersEnd - waveformPointersStart[0])
    T0T1TXTimes = cp.zeros( (1,args.num_splits,5), dtype=cp.uint64)
    command = 'thisSignalGATSPISubchunkWaves = cp.asarray(saveSubchunkWaveforms_' + str(subchunk_id) + '[waveformPointersStart[0]:waveformPointersEnd[-1]])' ; exec(command, globals())
    thisSubchunkDuration = subchunkEndTimes[subchunk_id] - subchunkStartTimes[subchunk_id]
    thisSubchunkTimeStart = max(timeStart, subchunkStartTimes[subchunk_id]) ; thisSubchunkTimeEnd = min(timeEnd, subchunkEndTimes[subchunk_id])
    thisSubchunkTimeStart -= subchunkStartTimes[subchunk_id] ; thisSubchunkTimeEnd -= subchunkStartTimes[subchunk_id]
    getSAIFForTimestampRegion( (1,1) , (32,1), \
     (thisSignalGATSPISubchunkWaves,perNodePerSplitWaveformPointerStart,\
     perNodePerSplitWaveformPointerEnd, T0T1TXTimes, args.num_splits,1,fold_split,thisSubchunkDuration, \
     thisSubchunkTimeStart, thisSubchunkTimeEnd, subchunk_id) )
    T0T1TXTimes = cp.sum(T0T1TXTimes, axis=1)
    if subchunk_id == neededSubchunks[0]:
      masterT0T1TXTimes = T0T1TXTimes.repeat(1).reshape(5) ;
    else:
      masterT0T1TXTimes = masterT0T1TXTimes + T0T1TXTimes.reshape(5) ;
  print_stuff = signal_name + ' SAIF info for timerange (' + str(timeStart) + ' -- ' + str(timeEnd) + ' ):' + '\n'
  print_stuff += '(T0 '
  print_stuff += str(int(masterT0T1TXTimes[0])) + ') (T1 ' + str(int(masterT0T1TXTimes[1])) + ') (TX ' + str(int(masterT0T1TXTimes[2])) + ') '
  print_stuff += '(TC ' + str(int(masterT0T1TXTimes[3])) + ') (IG ' + str(int(masterT0T1TXTimes[4])) + ')'    
  print(print_stuff)


def reportWaveformInTimerange(signal_name, timeStart,timeEnd):
  global debug_node_num,waveformPointersStart,waveformPointersEnd,thisSignalGATSPISubchunkWaves ;
  debug_node_num = cell_names[cell_dict[signal_name]]
  for subchunk_id in range(args.num_of_sub_chunks):
    command = 'waveformPointersStart = saveSubchunkPointersStart_' + str(subchunk_id) + '[debug_node_num]' ; exec(command, globals()) ;
    command = 'waveformPointersEnd = saveSubchunkPointersEnd_' + str(subchunk_id) + '[debug_node_num]' ; exec(command, globals()) 
    thisNodeWaveformLength= int(waveformPointersEnd[args.num_splits-1] - waveformPointersStart[0] - 2*args.num_splits + 1)
    perNodeWholeWaveformPointers = th.LongTensor([0,thisNodeWaveformLength])
    wholeWaveformTempStorage = cp.zeros( (int(thisNodeWaveformLength),), dtype=cp.uint64)
    perNodePerSplitWaveformPointerStart = cp.asarray(waveformPointersStart - waveformPointersStart[0])
    perNodePerSplitWaveformPointerEnd = cp.asarray(waveformPointersEnd - waveformPointersStart[0])
    perNodeWholeWaveformPointers= cp.asarray(perNodeWholeWaveformPointers).astype(cp.uint32)
    T0T1TXTimes = cp.zeros( (1,args.num_splits,5), dtype=cp.uint64)
    command = 'thisSignalGATSPISubchunkWaves = cp.asarray(saveSubchunkWaveforms_' + str(subchunk_id) + '[waveformPointersStart[0]:waveformPointersEnd[-1]])' ; exec(command, globals())
    thisSubchunkDuration = subchunkEndTimes[subchunk_id] - subchunkStartTimes[subchunk_id]
    reconstructWholeWaveforms( (1,1) , (32,1), \
    (thisSignalGATSPISubchunkWaves,wholeWaveformTempStorage,perNodeWholeWaveformPointers,perNodePerSplitWaveformPointerStart,\
    perNodePerSplitWaveformPointerEnd,T0T1TXTimes,args.num_splits,1,fold_split,thisSubchunkDuration,subchunk_id) )
    if subchunk_id == 0:
      masterWaveforms = wholeWaveformTempStorage ;
    else:
      wholeWaveformTempStorage += subchunkStartTimes[subchunk_id]
      masterWaveforms = cp.concatenate( (masterWaveforms,wholeWaveformTempStorage[1:]), axis=0)
  indexValues = cp.zeros( (2), dtype=cp.uint32)
  getVCDIndexesOfSignalInTimerange( (math.ceil(masterWaveforms.shape[0]/512),1), (512,1), \
   ( masterWaveforms,indexValues,timeStart,timeEnd,masterWaveforms.shape[0] ) )
  subWaveformLength = int(indexValues[1] - indexValues[0] + 1)
  Values = cp.zeros( (subWaveformLength), dtype=cp.uint8) ; Timestamps = cp.zeros( (subWaveformLength), dtype=cp.uint64)
  extractTimeAndValue( (math.ceil(subWaveformLength/512),1), (512,1), \
   (masterWaveforms,indexValues,Values,Timestamps,subWaveformLength ) )
  Values = Values.get() ; Timestamps = Timestamps.get()
  print_string =''
  print_string +=signal_name + ' for duration ' + str(timeStart) + ' -- ' + str(timeEnd) + ' : \n'
  thisValue = 'x' if Values[0] == 2 else str(int(Values[0]))
  print_string += thisValue + '@ ' + str(timeStart) + '\n'
  for i in range(1,subWaveformLength):
    thisValue = 'x' if Values[i] == 2 else str(int(Values[i]))
    print_string += thisValue + '@ ' + str(int(Timestamps[i])) + '\n'
  thisValue = 'x' if Values[-1] == 2 else str(int(Values[-1]))
  print_string += thisValue + '@ ' + str(timeEnd) + '\n' 
  print(print_string)

def reportGateStateAtTimestamp(outputPinName, targetTimestamp):
  global debug_node_num,waveformPointersStart,waveformPointersEnd,thisSignalGATSPISubchunkWaves ;
  debug_node_num2 = cell_names[outputPinName]
  in_pins, in_edges2 = bg.in_edges(debug_node_num2);
  inEdgeIds = bg.edge_ids(in_pins,in_edges2)
  pinPositions = bg.edata['x'][inEdgeIds]
  pinPositions, pinPositionIndexes = th.sort(pinPositions, descending=True)
  in_pins = in_pins[pinPositionIndexes] ; inEdgeIds = bg.edge_ids(in_pins,in_edges2)
  debug_celltype = int(bg.ndata['cell_type'][debug_node_num2])
  cellPinNames = cells_list[debug_celltype][1]
  allPins = th.cat( (in_pins, th.cuda.LongTensor([debug_node_num2])) )
  for i in range(args.num_of_sub_chunks):
    if (targetTimestamp >= subchunkStartTimes[i]) and (targetTimestamp < subchunkEndTimes[i]) :
      subchunk_id = i ; pseudoTargetTimestamp = targetTimestamp - subchunkStartTimes[subchunk_id] ;break;
  print_string = ''
  print_string += 'Gate Pin Values for ' + outputPinName + ' @' + str(targetTimestamp) + ' :\n' 
  #for each signal pin: reconstruct the waveform. report the pin state
  for signal in allPins:
    debug_node_num = int(signal)
    command = 'waveformPointersStart = saveSubchunkPointersStart_' + str(subchunk_id) + '[debug_node_num]' ; exec(command, globals()) ;
    command = 'waveformPointersEnd = saveSubchunkPointersEnd_' + str(subchunk_id) + '[debug_node_num]' ; exec(command, globals()) 
    thisNodeWaveformLength= int(waveformPointersEnd[args.num_splits-1] - waveformPointersStart[0] - 2*args.num_splits + 1)
    perNodeWholeWaveformPointers = th.LongTensor([0,thisNodeWaveformLength])
    wholeWaveformTempStorage = cp.zeros( (int(thisNodeWaveformLength),), dtype=cp.uint64)
    perNodePerSplitWaveformPointerStart = cp.asarray(waveformPointersStart - waveformPointersStart[0])
    perNodePerSplitWaveformPointerEnd = cp.asarray(waveformPointersEnd - waveformPointersStart[0])
    perNodeWholeWaveformPointers= cp.asarray(perNodeWholeWaveformPointers).astype(cp.uint32)
    T0T1TXTimes = cp.zeros( (1,args.num_splits,5), dtype=cp.uint64)
    command = 'thisSignalGATSPISubchunkWaves = cp.asarray(saveSubchunkWaveforms_' + str(subchunk_id) + '[waveformPointersStart[0]:waveformPointersEnd[-1]])' ; exec(command, globals())
    thisSubchunkDuration = subchunkEndTimes[subchunk_id] - subchunkStartTimes[subchunk_id]
    reconstructWholeWaveforms( (1,1) , (32,1), \
    (thisSignalGATSPISubchunkWaves,wholeWaveformTempStorage,perNodeWholeWaveformPointers,perNodePerSplitWaveformPointerStart,\
    perNodePerSplitWaveformPointerEnd,T0T1TXTimes,args.num_splits,1,fold_split,thisSubchunkDuration,subchunk_id) )
    vcValue = cp.zeros( (1), dtype=cp.uint8)
    getSignalStateAtTimestamp( (math.ceil(thisNodeWaveformLength/512),1), (512,1), \
     (wholeWaveformTempStorage,0,vcValue,pseudoTargetTimestamp, thisNodeWaveformLength ) )
    if vcValue[0] ==2:
      vcValue = 'x'
    else:
      vcValue = str(int(vcValue[0]))
    if int(signal) != debug_node_num2 :
      print_string += cellPinNames[i] + ': ' + vcValue + '\t'
    else :
      print_string += outputPinName + ': ' + vcValue
  print(print_string)


exec(open('GL0AM.cupy').read())

parser = argparse.ArgumentParser()
parser.add_argument('--block', type=str)
parser.add_argument('--instance_name', type=str)
parser.add_argument('--testname', type=str)
parser.add_argument('--cycles', type=int, help = "number of cycles in simulation")
parser.add_argument('--period', type=int, help = 'period, in 1ps units')
parser.add_argument('--input_trace_file', type=str, help = 'input .waveforms file, will be a pkled file')
parser.add_argument('--graphPrecompile', type=str, help = 'for debug purposes')
parser.add_argument('--graph', type=str, help = 'the DGL translated graph')
parser.add_argument('--graph2', type=str, help = 'the DGL translated graph for registers')
parser.add_argument('--graph3', type=str, help='the DGL translated graph for srams')
parser.add_argument('--graph4', type=str, help='the split FO graph')
parser.add_argument('--timescale', type=float, \
help = 'timescale relative to 1ps, used to massage waveform and delay numbers to match the simulation timescale')
parser.add_argument('--pin_net_file', type=str, help = 'path to the pin to net translation file that helps create SAIF')
parser.add_argument('--num_splits', type=int,  default=32, help ='number of waveform parallelism desired in re-simulation phase, must be power of 2, up to 512', \
choices=[1,2,4,8,16,32,64,128,256,512])
parser.add_argument('--duration', type=int, help='the total simulation duration, in 1ps units')
parser.add_argument('--clk', action = 'append', nargs='+', type=str, help = 'name of the clock input port(s)')
parser.add_argument('--first_edge', type=int, help = 'time of the first latching edge of the clk')
parser.add_argument('--rst',  action='append', nargs='+', default = [['']], type=str, help = 'name of the rst signal(s)')
parser.add_argument('--num_of_sub_chunks', type=int, help='number of sub chunks to stream if  GATSPI resimulation waveform is too big to fit on 1 GPU')
parser.add_argument('--zeroDelayReportSignals', action='append', nargs='+', type=str, help = 'name of net signals to report after 0-delay simulation is done')
parser.add_argument('--debugMode', type=bool, default=False, help='turn on the debug environment after all the simulation has finished')
parser.add_argument('--clkTreeFile', type=str,  help='File that lists output pins part of clock tree clock gates')
parser.add_argument('--partitionsFile', type=str,  help='File that stores the result of hypergraph partitioning of the combo graph')
args = parser.parse_args()
# Future work might be to do away with some parameters that are set manually and instead are set by analyzing the inputs files.
# Such as: cycles, period, duration, clk, first_edge, rst
if len(args.rst) == 1 :
  args.rst.append([''])

threads_per_block=512

# +
#load the important input files
fileObject = open(args.graph, 'rb') ; bg = pickle.load(fileObject)
fileObject = open(args.graph2, 'rb') ; bg2 = pickle.load(fileObject)
fileObject = open(args.graph3, 'rb') ; bg3 = pickle.load(fileObject)
f = gzip.GzipFile(args.graphPrecompile, "r"); data = np.load(f, allow_pickle=1) ; f.close()
print("pkl loaded")
cell_names=data['cell_index']
num_of_std_cells=data['num_of_std_cells']
cell_names=dict((k, cell_names[k]) for k in range(num_of_std_cells))
port_names=data['port_index'] 
port_names=dict((k, port_names[k]) for k in range(num_of_std_cells, len(bg.nodes())))
cell_names.update(port_names)
cell_names.update( { v:k for k,v in cell_names.items() } )
ram_rows = data['ram_rows']
ram_cols = data['ram_cols']
del data
#scale everything to units of 1ps
bg.edata['net_delay_rise'] = (th.round( (bg.edata['net_delay_rise'] *1000/args.timescale) + 0.001 ) * args.timescale).type(th.IntTensor)
bg.edata['net_delay_fall'] = (th.round( (bg.edata['net_delay_fall'] *1000/args.timescale) + 0.001 ) * args.timescale).type(th.IntTensor)

#this dictionary translates from net name to pin name
cell_dict_file=open(args.pin_net_file, 'r')
cell_dict= {}
for line in cell_dict_file:
  part0, part1, =line.strip().split(' ')
  cell_dict[part1] = part0
cell_dict["1'b0"] = "1'b0"
cell_dict["1'b1"] = "1'b1"
f = gzip.GzipFile(args.input_trace_file, "r"); waveforms = np.load(f, allow_pickle=1) ; f.close()
waveforms_value = waveforms['waveforms_value']
waveforms = waveforms['waveforms']
#add patch for 1'b1 and 1'b0
waveforms["1'b1"] = np.array([0])
waveforms["1'b0"] = np.array([0])
waveforms_value["1'b1"] = np.array([1])
waveforms_value["1'b0"] = np.array([0])

cell_dict_file=open(args.pin_net_file, 'r')
cell_dict2= {}
for line in cell_dict_file:
  part0, part1, =line.strip().split(' ')
  cell_dict2[part0] = part1
gnd_node=cell_names["1'b0"]
vdd_node =cell_names["1'b1"]


# +
#hook to create waveform format of 2 tensors
end_token = 9223372036854775800 #some very large in all likelihood unobtainable 64bit number
end_token_value = 9
waveforms_pointers = [] ; current_pointer = 0
node_nums=[]
for key in list(waveforms.keys()):
  waveforms_value[key] = waveforms_value[key].astype(np.int8)
  waveforms[key] = np.concatenate( (waveforms[key], np.array([end_token])) ).astype(np.int64)
  waveforms_value[key] = np.concatenate( (waveforms_value[key], np.array([end_token_value])) ).astype(np.int8)
  waveforms_pointers.append(current_pointer); 
  current_pointer += waveforms[key].shape[0]
  if key[1].isalpha():
    adjusted_netname=re.sub(r'^\\','', key)
  else:
    adjusted_netname = key
  if adjusted_netname in cell_names.keys():
    node_nums.append(cell_names[adjusted_netname])
waveforms_pointers.append(current_pointer);  


ending = 2 ; num_splits=1
device = "cuda:0"
waveforms_pointers = (th.LongTensor(waveforms_pointers) + ending).type(th.int64)
waveforms_total= th.LongTensor( np.concatenate( [np.array([0,end_token] * num_splits)] + list(waveforms.values()) ) ).to("cpu")
waveforms_values_total= th.CharTensor( np.concatenate( [np.array([0,end_token_value] * num_splits)] + list(waveforms_value.values()) ) ).to("cpu")
node_nums = th.LongTensor( node_nums )

bg.ndata['node_level'] = th.IntTensor( [0] ).repeat(len(bg.nodes()))
bg.ndata['waveform_start'] = th.LongTensor( [2*x for x in range(num_splits)] ).unsqueeze(0).repeat(len(bg.nodes()), 1).type(th.int64)
bg.ndata['waveform_end'] = th.LongTensor( [2*x+2 for x in range(num_splits)] ).unsqueeze(0).repeat(len(bg.nodes()), 1).type(th.int64)
bg.ndata['waveform_start'][node_nums] = waveforms_pointers[0:-1].view(len(node_nums), num_splits)
bg.ndata['waveform_end'][node_nums] = waveforms_pointers[1:].view(len(node_nums), num_splits)

#this gets a tensor of all the 'active' timestamps that will be simulated in 0 delay mode
#basically the rising edge of the clk + any rst events
#probably can be modified to include falling edges too, if we need that
#right now, I'm assuming there is one clock that is single phase, which may have 1 or more clock input ports
#should make some event-based timestamp streaming for future work.
clk_node_num = cell_names[args.clk[0][0]]
temp_active_timestamps = waveforms_total[bg.ndata['waveform_start'][clk_node_num]:bg.ndata['waveform_end'][clk_node_num]][:-1][1::2]
final_end = temp_active_timestamps[-1:]+args.period
first_edge_init = th.LongTensor([0]) 
temp_active_timestamps = th.cat( ( first_edge_init, temp_active_timestamps, final_end ) )
sync_or_async_event = th.zeros(temp_active_timestamps.size()[0], dtype = th.bool)
sync_or_async_event[0]=1
#add one more phantom cycle because the first value is actually the init value, and if I don't add another cycle
#the actual real last cycle doesn't get simulated
if args.rst[1][0] != '':
  for this_async_signal in args.rst[1]:
    rst_node_num = cell_names[this_async_signal]
    temp_rst_timestamps = waveforms_total[bg.ndata['waveform_start'][rst_node_num]:bg.ndata['waveform_end'][rst_node_num]][:-1]
    temp_active_timestamps = th.cat( ( temp_active_timestamps, temp_rst_timestamps ) )
    sync_or_async_event = th.cat( (sync_or_async_event,th.ones(temp_rst_timestamps.size()[0], dtype = th.bool) ))
  temp_active_timestamps, sorted_indexes = temp_active_timestamps.sort()
  sync_or_async_event = sync_or_async_event[sorted_indexes]
temp_active_timestamps = temp_active_timestamps.to(device)
sync_or_async_event = sync_or_async_event.to(device)
#temp_active_timestamps stores the timestamps the 0 delay sim will take place at


#if sometimes both async signals like rst and sync signals like clk happen at the same time, the event is considered asynchronous
temp_active_timestamps, timestamp_counts = th.unique(temp_active_timestamps, return_counts=True)
sync_or_async_event_processed = th.ones(temp_active_timestamps.size()[0], dtype = th.bool)
timestamp_counts = cp.asarray(timestamp_counts).astype(cp.int64)
sync_or_async_event = cp.asarray(sync_or_async_event.to("cpu")).astype(cp.bool_)
sync_or_async_event_processed = cp.asarray(sync_or_async_event_processed).astype(cp.bool_)
figure_out_sync_or_async( (math.ceil(temp_active_timestamps.size()[0]/1024),), (1024,), \
  (sync_or_async_event, timestamp_counts,sync_or_async_event_processed, \
  cp.int32(temp_active_timestamps.size()[0]) ) )


temp_active_timestamps = cp.asarray(temp_active_timestamps).astype(cp.int64)
sync_or_async_event_processed = cp.asarray(sync_or_async_event_processed).astype(cp.bool_)

#clk and rst get 'special treatment', their signal type is 1
#this step compiles all the input signals in relation to their value at each timestamp in temp_active_timestamps
#probably should iterate and improve upon this for future work too.
waveforms_total = waveforms_total.to(device)
waveforms_values_total =waveforms_values_total.to(device)
pointers = (bg.ndata['waveform_start'][node_nums]).squeeze(1).to(device)
change_list_size = temp_active_timestamps.shape[0] -1
input_values_array = th.zeros( ( node_nums.size()[0], change_list_size), dtype=th.int8, device=device)
signal_type = th.zeros( (node_nums.size()[0]), dtype=th.int8, device=device)
for this_clk_input_port in args.clk[0]:
  signal_type[(node_nums == cell_names[this_clk_input_port]).nonzero().squeeze(0)] = 2
if args.rst[1][0] != '':
  for this_asynch_signal in args.rst[1]:
    signal_type[(node_nums == cell_names[this_asynch_signal]).nonzero().squeeze(0)] = 1
waveforms_total = cp.asarray(waveforms_total).astype(cp.int64)
waveforms_values_total=cp.asarray(waveforms_values_total).astype(cp.int8)
pointers = cp.asarray(pointers).astype(cp.int64)
input_values_array = cp.asarray(input_values_array).astype(cp.int8)
signal_type = cp.asarray(signal_type).astype(cp.int8)
temp_active_timestamps = cp.asarray(temp_active_timestamps).astype(cp.int64)
set_input_array( (math.ceil(input_values_array.shape[0]/1024),), (1024,), (waveforms_total, waveforms_values_total, \
  pointers, input_values_array, sync_or_async_event_processed,\
  signal_type, temp_active_timestamps, cp.int32(change_list_size),\
  cp.int32(input_values_array.shape[0]) ) ) 
# -

#also prep the input waveforms for the GATSPI (re-simulation) phase.
saveInputNodeNums = cp.asarray(node_nums).astype(cp.int32)
saveInputPointers = pointers - ending
del pointers
saveInputWaveformsTotal = cp.zeros( (waveforms_total.shape[0]), dtype=cp.uint64)
formatWholeInputWaveformsForGATSPISubchunking( (math.ceil(waveforms_total.shape[0]/512),1), (512,1), \
 ( waveforms_total,waveforms_values_total,saveInputWaveformsTotal,saveInputWaveformsTotal.shape[0],end_token ) )
del waveforms_total; del waveforms_values_total;  del signal_type
saveInputWaveformsTotal = saveInputWaveformsTotal[ending:]
mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()


# +
#load the truth tables for the simulation
#the boolean functionality of each gate is hardcoded to the row number within the 'cells_list' list.
#need to think of a more generic way to compile std cell libs in the future.
def dec_to_bin(x,bit_width):
  return bin(x)[2:].zfill(bit_width)

#had some standard cells that didn't end up in the final GENERIC.vlib, hence 'dummy' cell types.
logic_truth_tables = {}
cells_list=[ ("AND2", ['A1', 'A2'], "int(bits[0] and bits[1])"), \
("AND3", ['A1', 'A2', 'A3'], "int(bits[0] and bits[1] and bits[2])"), \
("AND4", ['A1', 'A2', 'A3', 'A4'], "int(bits[0] and bits[1] and bits[2] and bits[3])"), \
("AO211", ['A1', 'A2', 'B', 'C'], "int((bits[0] and bits[1]) or bits[2] or bits[3])"), \
("AO21", ['A1', 'A2', 'B'], "int((bits[0] and bits[1]) or bits[2])"), \
("AO221", ['A1', 'A2', 'B1', 'B2', 'C'], "int((bits[0] and bits[1]) or (bits[2] and bits[3]) or bits[4])"), \
("AO2222", ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2'], "int((bits[0] and bits[1]) or (bits[2] and bits[3]) or (bits[4] and bits[5]) or (bits[6] and bits[7]))"), \
("AO222", ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], "int((bits[0] and bits[1]) or (bits[2] and bits[3]) or (bits[4] and bits[5]))"), \
("AO22", ['A1', 'A2', 'B1', 'B2'], "int((bits[0] and bits[1]) or (bits[2] and bits[3]))"), \
("AO31", ['A1', 'A2', 'A3', 'B'], "int((bits[0] and bits[1] and bits[2]) or bits[3])"), \
("AO32", ['A1', 'A2', 'A3', 'B1', 'B2'], "int((bits[0] and bits[1] and bits[2]) or (bits[3] and bits[4]))"), \
("AO33", ['A1', 'A2', 'A3', 'B1', 'B2', 'B3'], "int((bits[0] and bits[1] and bits[2]) or (bits[3] and bits[4] and bits[5]))"), \
("AOAI211", ['A1', 'A2', 'B', 'C'], "int(not((((bits[0] and bits[1]) or bits[2]) and bits[3])))"), \
("AOI211", ['A1', 'A2', 'B', 'C'], "int(not((bits[0] and bits[1]) or bits[2] or bits[3]))"), \
("AOI21", ['A1', 'A2', 'B'], "int(not((bits[0] and bits[1]) or bits[2]))"), \
("AOI221", ['A1', 'A2', 'B1', 'B2', 'C'], "int(not((bits[0] and bits[1]) or (bits[2] and bits[3]) or bits[4]))"), \
("AOI222", ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], "int(not((bits[0] and bits[1]) or (bits[2] and bits[3]) or (bits[4] and bits[5])))"), \
("AOI22", ['A1', 'A2', 'B1', 'B2'], "int(not((bits[0] and bits[1]) or (bits[2] and bits[3])))"), \
("AOI31", ['A1', 'A2', 'A3', 'B'], "int(not((bits[0] and bits[1] and bits[2]) or bits[3]))"), \
("AOI32", ['A1', 'A2', 'A3', 'B1', 'B2'], "int(not((bits[0] and bits[1] and bits[2]) or (bits[3] and bits[4])))"), \
("AOI33", ['A1', 'A2', 'A3', 'B1', 'B2', 'B3'], "int(not((bits[0] and bits[1] and bits[2]) or (bits[3] and bits[4] and bits[5])))"), \
("BUFF", ['I'], "int(bits[0])"), \
("MUX2", ['I0', 'I1', 'S'], "int((not(bits[2]) and bits[0]) or (bits[2] and bits[1]))"), \
("MUX2N", ['I0', 'I1', 'S'], "int(not((not(bits[2]) and bits[0]) or (bits[2] and bits[1])))"), \
("INV", ['I'], "int(not(bits[0]))"), \
("NAND2", ['A1', 'A2'], "int(not(bits[0] and bits[1]))"), \
("NAND3", ['A1', 'A2', 'A3'], "int(not(bits[0] and bits[1] and bits[2]))"), \
("NOR2", ['A1', 'A2'], "int(not(bits[0] or bits[1]))"), \
("NOR3", ['A1', 'A2', 'A3'], "int(not(bits[0] or bits[1] or bits[2]))"), \
("OR2", ['A1', 'A2'], "int(bits[0] or bits[1])"), \
("XOR2", ['A1', 'A2'], "int(bits[0] ^ bits[1])"), \
("FA_SUM", ['A', 'B', 'CI'], "int(bits[0] ^ bits[1] ^ bits[2])"), \
("FA_CO", ['A', 'B', 'CI'], "int((bits[0] and bits[1]) or (bits[2] and (bits[0] ^ bits[1])))"), \
("dummy0", ['A', 'B', 'CI'], "int(bits[0] ^ bits[1] ^ bits[2])"), \
("dummy1", ['A', 'B', 'CI'], "int((bits[0] and bits[1]) or (bits[2] and (bits[0] ^ bits[1])))"), \
("dummy2", ['A', 'B', 'CI'], "int(bits[0] ^ bits[1] ^ bits[2])"), \
("dummy3", ['A', 'B', 'CI'], "int((bits[0] and bits[1]) or (bits[2] and (bits[0] ^ bits[1])))"), \
("dummy4", ['A', 'B', 'CI'], "int(bits[0] ^ bits[1] ^ bits[2])"), \
("dummy5", ['A', 'B', 'CI'], "int((bits[0] and bits[1]) or (bits[2] and (bits[0] ^ bits[1])))"), \
("dummy6", ['A', 'B', 'CI'], "int(bits[0] ^ bits[1] ^ bits[2])"), \
("dummy7", ['A', 'B', 'CI'], "int((bits[0] and bits[1]) or (bits[2] and (bits[0] ^ bits[1])))"), \
("dummy8", ['A', 'B', 'CI'], "int(bits[0] ^ bits[1] ^ bits[2])"), \
("dummy9", ['A', 'B', 'CI'], "int((bits[0] and bits[1]) or (bits[2] and (bits[0] ^ bits[1])))"), \
("dummy10", ['A', 'B', 'CI'], "int(bits[0] ^ bits[1] ^ bits[2])"), \
("dummy11", ['A', 'B', 'CI'], "int((bits[0] and bits[1]) or (bits[2] and (bits[0] ^ bits[1])))"), \
("HA_SUM", ['A', 'B'], "int(bits[0] ^ bits[1])"), \
("HA_CO", ['A', 'B'], "int(bits[0] and bits[1])"), \
("IAN2", ['A1', 'B1'], "int(not(bits[0]) and bits[1])"), \
("IAO21", ['A1', 'A2', 'B'], "int(not((not(bits[0]) and not(bits[1])) or bits[2]))"), \
("IAO22", ['A1', 'A2', 'B1', 'B2'], "int(not((not(bits[0]) and not(bits[1])) or (bits[2] and bits[3])))"), \
("IAOI211", ['A1', 'A2', 'B', 'C'], "int(not((not(bits[0]) and bits[1]) or bits[2] or bits[3]))"), \
("IAOI21", ['A1', 'A2', 'B'], "int(not((bits[0] and bits[1]) or (not(bits[2]))))"), \
("IBAO21", ['A1', 'A2', 'B'], "int((bits[0] and bits[1]) or (not(bits[2])))"), \
("IBOA21", ['A1', 'A2', 'B'], "int((bits[0] or bits[1]) and (not(bits[2])))"), \
("IIAOI21", ['A1', 'A2', 'B'], "int(not((not(bits[0]) and bits[1]) or bits[2]))"), \
("IIAOI22", ['A1', 'A2', 'B1', 'B2'], "int(not((not(bits[0]) and not(bits[1])) or (bits[2] and bits[3])))"), \
("IIND3", ['A1', 'A2', 'B1'], "int(not(not(bits[0]) and not(bits[1]) and bits[2]))"), \
("IIND4", ['A1', 'A2', 'B1', 'B2'], "int(not(not(bits[0]) and not(bits[1]) and bits[2] and bits[3]))"), \
("IINR3", ['A1', 'A2', 'B1'], "int(not(not(bits[0]) or not(bits[1]) or bits[2]))"), \
("IINR4", ['A1', 'A2', 'B1', 'B2'], "int(not(not(bits[0]) or not(bits[1]) or bits[2] or bits[3]))"), \
("IIOAI21", ['A1', 'A2', 'B'], "int(not((not(bits[0]) or bits[1]) and bits[2]))"), \
("IIOAI22", ['A1', 'A2', 'B1', 'B2'], "int(not((not(bits[0]) or not(bits[1])) and (bits[2] or bits[3])))"), \
("IND2", ['A1', 'B1'], "int(not(not(bits[0]) and bits[1]))"), \
("IND3", ['A1', 'B1', 'B2'], "int(not(not(bits[0]) and bits[1] and bits[2]))"), \
("IND4", ['A1', 'B1', 'B2', 'B3'], "int(not(not(bits[0]) and bits[1] and bits[2] and bits[3]))"), \
("INR2", ['A1', 'B1'], "int(not(not(bits[0]) or bits[1]))"), \
("INR3", ['A1', 'B1', 'B2'], "int(not(not(bits[0]) or bits[1] or bits[2]))"), \
("INR4", ['A1', 'B1', 'B2', 'B3'], "int(not(not(bits[0]) or bits[1] or bits[2] or bits[3]))"), \
("IOA21", ['A1', 'A2', 'B'], "int(not((not(bits[0]) or not(bits[1])) and bits[2]))"), \
("IOA22", ['A1', 'A2', 'B1', 'B2'], "int(not((not(bits[0]) or not(bits[1])) and (bits[2] or bits[3])))"), \
("IOAI211", ['A1', 'A2', 'B', 'C'], "int(not((not(bits[0]) or bits[1]) and bits[2] and bits[3]))"), \
("IOAI21", ['A1', 'A2', 'B'], "int(not((bits[0] or bits[1]) and not(bits[2])))"), \
("IOR2", ['A1', 'B1'], "int(not(bits[0]) or bits[1])"), \
("MAOI222", ['A', 'B', 'C'], "int(not((bits[0] and bits[1]) or (bits[0] and bits[2]) or (bits[1] and bits[2])))"), \
("MAOI22", ['A1', 'A2', 'B1', 'B2'], "int(not((bits[0] and bits[1]) or (not(bits[2] or bits[3]))))"), \
("MOAI22", ['A1', 'A2', 'B1', 'B2'], "int(not((bits[0] or bits[1]) and (not(bits[2] and bits[3]))))"), \
("MUX3", ['I0', 'I1', 'I2', 'S0', 'S1'], "int((not(bits[3]) and not(bits[4]) and bits[0]) or (bits[3] and (not(bits[4])) and bits[1]) or (bits[4] and bits[2]))"), \
("MUX3N", ['I0', 'I1', 'I2', 'S0', 'S1'], "int(not((not(bits[3]) and not(bits[4]) and bits[0]) or (bits[3] and (not(bits[4])) and bits[1]) or (bits[4] and bits[2])))"), \
("MUX4", ['I0', 'I1', 'I2', 'I3', 'S0', 'S1'], "int((not(bits[4]) and not(bits[5]) and bits[0]) or (bits[4] and (not(bits[5])) and bits[1]) or (not(bits[4]) and bits[5] and bits[2]) or (bits[4] and bits[5] and bits[3]))"), \
("MUX4N", ['I0', 'I1', 'I2', 'I3', 'S0', 'S1'], "int(not((not(bits[4]) and not(bits[5]) and bits[0]) or (bits[4] and (not(bits[5])) and bits[1]) or (not(bits[4]) and bits[5] and bits[2]) or (bits[4] and bits[5] and bits[3])))"), \
("NAND4", ['A1', 'A2', 'A3', 'A4'], "int(not(bits[0] and bits[1] and bits[2] and bits[3]))"), \
("NOR4", ['A1', 'A2', 'A3', 'A4'], "int(not(bits[0] or bits[1] or bits[2] or bits[3]))"), \
("OA211", ['A1', 'A2', 'B', 'C'], "int((bits[0] or bits[1]) and bits[2] and bits[3])"), \
("OA21", ['A1', 'A2', 'B'], "int((bits[0] or bits[1]) and bits[2])"), \
("OA221", ['A1', 'A2', 'B1', 'B2', 'C'], "int((bits[0] or bits[1]) and (bits[2] or bits[3]) and bits[4])"), \
("OA222", ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], "int((bits[0] or bits[1]) and (bits[2] or bits[3]) and (bits[4] or bits[5]))"), \
("OA22", ['A1', 'A2', 'B1', 'B2'], "int((bits[0] or bits[1]) and (bits[2] or bits[3]))"), \
("OA31", ['A1', 'A2', 'A3', 'B'], "int((bits[0] or bits[1] or bits[2]) and bits[3])"), \
("OA32", ['A1', 'A2', 'A3', 'B1', 'B2'], "int((bits[0] or bits[1] or bits[2]) and (bits[3] or bits[4]))"), \
("OA33", ['A1', 'A2', 'A3', 'B1', 'B2', 'B3'], "int((bits[0] or bits[1] or bits[2]) and (bits[3] or bits[4] or bits[5]))"), \
("OAI211", ['A1', 'A2', 'B', 'C'], "int(not((bits[0] or bits[1]) and bits[2] and bits[3]))"), \
("OAI21", ['A1', 'A2', 'B'], "int(not((bits[0] or bits[1]) and bits[2]))"), \
("OAI221", ['A1', 'A2', 'B1', 'B2', 'C'], "int(not((bits[0] or bits[1]) and (bits[2] or bits[3]) and bits[4]))"), \
("OAI222", ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], "int(not((bits[0] or bits[1]) and (bits[2] or bits[3]) and (bits[4] or bits[5])))"), \
("OAI22", ['A1', 'A2', 'B1', 'B2'], "int(not((bits[0] or bits[1]) and (bits[2] or bits[3])))"), \
("OAI31", ['A1', 'A2', 'A3', 'B'], "int(not((bits[0] or bits[1] or bits[2]) and bits[3]))"), \
("OAI32", ['A1', 'A2', 'A3', 'B1', 'B2'], "int(not((bits[0] or bits[1] or bits[2]) and (bits[3] or bits[4])))"), \
("OAI33", ['A1', 'A2', 'A3', 'B1', 'B2', 'B3'], "int(not((bits[0] or bits[1] or bits[2]) and (bits[3] or bits[4] or bits[5])))"), \
("OAOI211", ['A1', 'A2', 'B', 'C'], "int(not(((bits[0] or bits[1]) and bits[2]) or bits[3]))"), \
("OR3", ['A1', 'A2', 'A3'], "int(bits[0] or bits[1] or bits[2])"), \
("OR4", ['A1', 'A2', 'A3', 'A4'], "int(bits[0] or bits[1] or bits[2] or bits[3])"), \
("XNR2", ['A1', 'A2'], "int(not(bits[0] ^ bits[1]))"), \
("XNR3", ['A1', 'A2', 'A3'], "int(not(bits[0] ^ bits[1] ^ bits[2]))"), \
("XOR3", ['A1', 'A2', 'A3'], "int(bits[0] ^ bits[1] ^ bits[2])") ]

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


#includes 'x' processing
cell_counter = 0
for cell_info in cells_list:
  cell_name=cell_info[0]
  cell_pins=cell_info[1]
  cell_func=cell_info[2]
  logic_truth_tables[cell_name]={}
  logic_truth_tables[cell_name]['pins']=cell_pins
  logic_truth_tables[cell_name]['cell_id']=cell_counter ; cell_counter+=1 ;
  truth_table = np.zeros(shape=(3**len(logic_truth_tables[cell_name]['pins']),len(logic_truth_tables[cell_name]['pins'])+1))
  for i in range(3**len(logic_truth_tables[cell_name]['pins'])):
    num_of_pins = len(logic_truth_tables[cell_name]['pins'])
    bits_np = np.array(numberToBase(i,3,num_of_pins))
    if 2 not in bits_np:
      bits = list(bits_np)
      output = eval(cell_func)
      bits.append(output)
    else:
      eval_list = []
      num_of_2s = list(bits_np).count(2)
      indexes2 = (bits_np == 2).nonzero()[0]
      temp_bits_np = np.zeros(bits_np.shape[0]).astype(np.int64)
      for j in range(2 ** num_of_2s):
        np.copyto(temp_bits_np, bits_np) ;
        replacements = np.array(numberToBase(j,2,num_of_2s)) ; 
        temp_bits_np[indexes2] = replacements ; 
        eval_list.append(temp_bits_np.repeat(1))  ;   
      bits = list(eval_list[0])
      temp_output = eval(cell_func)
      j=1 ; temp_output2 = temp_output
      while (j < 2 ** num_of_2s) and (temp_output2 == temp_output):
        bits = list(eval_list[j])
        temp_output2 = eval(cell_func)
        j+=1
      if temp_output == temp_output2:
        output = temp_output
      else:
        output = 2
      bits = list(bits_np)
      bits.append(output)
    truth_table[i] = bits
  logic_truth_tables[cell_name]['truth_table']=truth_table

out_tables=th.zeros([len(logic_truth_tables.keys()), 6561], dtype=th.int8)
for cell_type in logic_truth_tables.keys():
  out_tables[logic_truth_tables[cell_type]['cell_id'],0:len(logic_truth_tables[cell_type]['truth_table'][:,-1])]=th.ByteTensor(logic_truth_tables[cell_type]['truth_table'][:,-1])

#compress the logic truth table LUT
comboCelltypeOffsets  = [] ; comboCelltypeOffset = 0 ; byteAddressTensor = [] ; bitAddressTensor = [] ; outs = [];
for cell_type in logic_truth_tables.keys():
  lengthOfOuts = len(logic_truth_tables[cell_type]['truth_table'][:,-1])
  outs.append(cp.asarray(logic_truth_tables[cell_type]['truth_table'][:,-1]).astype(cp.int8))
  howManyBytesToStoreOuts = math.ceil(lengthOfOuts/16)
  comboCelltypeOffsets.append(comboCelltypeOffset) ; 
  enumerateLengthOfOuts = np.asarray([x for x in range(lengthOfOuts)] , dtype = np.int32)
  bitAddresses = enumerateLengthOfOuts % 16 ; bitAddressTensor.append(cp.asarray(bitAddresses).astype(cp.int32))
  byteAddresses = (enumerateLengthOfOuts>>4).astype(np.int32) + comboCelltypeOffset ; 
  byteAddressTensor.append(cp.asarray(byteAddresses).astype(cp.int32))
  comboCelltypeOffset += howManyBytesToStoreOuts
outs = cp.concatenate(outs)
byteAddressTensor = cp.concatenate(byteAddressTensor)
bitAddressTensor = cp.concatenate(bitAddressTensor)
compactOutTables = cp.zeros(comboCelltypeOffset, dtype=cp.uint32)
comboCelltypeOffsets = cp.asarray(comboCelltypeOffsets).astype(cp.int32)
createCompactOutTables( (math.ceil(outs.shape[0]/512),1), (512,1),\
  (outs,byteAddressTensor,bitAddressTensor,compactOutTables,outs.shape[0]) )
compactOutTables = compactOutTables.astype(cp.uint32)
mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()
del outs,byteAddressTensor,bitAddressTensor

seq_truth_tables = {}
seq_list=[("GEN_FLOP", ['CDN', 'SDN', 'D', 'ENABLE', 'SE', 'SI', 'Q0', 'CP'], "int( ( ( ( ( (bits[5] and bits[4]) or ( not(bits[4]) and bits[3] and bits[2] ) or (not(bits[4]) and not(bits[3]) and bits[6]) ) and bits[7] ) or ( not(bits[7]) and bits[6] ) ) or not(bits[1]) ) and bits[0] )"), \
("GEN_CLKGATE", ['SDN', 'D', 'SE', 'SI', 'TD', 'TE', 'Q0', 'CP'], "int( ( ( ( ( ( ( (bits[3] and bits[2]) or ( not(bits[2]) and bits[1] ) ) and not(bits[4]) ) or bits[5] ) ) ) or not(bits[0]) ) and bits[7])"),\
("GEN_LATCH", ['SDN', 'D', 'SE', 'SI', 'TD', 'TE', 'Q0', 'CP'], "int( ( ( ( ( ( (bits[3] and bits[2]) or ( not(bits[2]) and bits[1] ) ) and not(bits[4]) ) or bits[5] ) ) ) or not(bits[0]) )") ]

cell_counter = 0
for cell_info in seq_list:
  cell_name=cell_info[0]
  cell_pins=cell_info[1]
  cell_func=cell_info[2]
  seq_truth_tables[cell_name]={}
  seq_truth_tables[cell_name]['pins']=cell_pins
  seq_truth_tables[cell_name]['cell_id']=cell_counter ; cell_counter+=1 ;
  truth_table = np.zeros(shape=(3**len(seq_truth_tables[cell_name]['pins']),len(seq_truth_tables[cell_name]['pins'])+1))
  for i in range(3**len(seq_truth_tables[cell_name]['pins'])):
    num_of_pins = len(seq_truth_tables[cell_name]['pins'])
    bits_np = np.array(numberToBase(i,3,num_of_pins))
    if 2 not in bits_np:
      bits = list(bits_np)
      output = eval(cell_func)
      bits.append(output)
    else:
      eval_list = []
      num_of_2s = list(bits_np).count(2)
      indexes2 = (bits_np == 2).nonzero()[0]
      temp_bits_np = np.zeros(bits_np.shape[0]).astype(np.int64)
      for j in range(2 ** num_of_2s):
        np.copyto(temp_bits_np, bits_np) ;
        replacements = np.array(numberToBase(j,2,num_of_2s)) ; 
        temp_bits_np[indexes2] = replacements ; 
        eval_list.append(temp_bits_np.repeat(1))  ;   
      bits = list(eval_list[0])
      temp_output = eval(cell_func)
      j=1 ; temp_output2 = temp_output
      while (j < 2 ** num_of_2s) and (temp_output2 == temp_output):
        bits = list(eval_list[j])
        temp_output2 = eval(cell_func)
        j+=1
      if temp_output == temp_output2:
        output = temp_output
      else:
        output = 2
      bits = list(bits_np)
      bits.append(output)
    truth_table[i] = bits
  seq_truth_tables[cell_name]['truth_table']=truth_table

out_tables_seq=th.zeros([len(seq_truth_tables.keys()), 6561], dtype=th.int8)
for cell_type in seq_truth_tables.keys():
  out_tables_seq[seq_truth_tables[cell_type]['cell_id'],0:len(seq_truth_tables[cell_type]['truth_table'][:,-1])]=th.ByteTensor(seq_truth_tables[cell_type]['truth_table'][:,-1])
# -

start_compile=timer()
#graph stuff starts here...setup the simulation graph
#most of this roundabout compile should be phased out in future versions by switching to C++ and doing away with python and DGL...
device = "cuda:0"
fileObject = open(args.graph4, 'rb') ; bg = pickle.load(fileObject)
bg=bg.to(device) ; bg2=bg2.to(device)
out_tables = out_tables.to(device)
temp=bg2.ndata['seq_pushoutSDF']
temp=(th.round( (temp *1000/args.timescale) + 0.001 ) * args.timescale).type(th.cuda.IntTensor)
bg2.ndata['seq_pushoutSDF']= temp
gnd_node = int(cell_names["1'b0"])
vdd_node = int(cell_names["1'b1"])
num_splits = args.num_splits


# +
#0 delay mode simulation
#set variables that define the 'static simulation graph'
topo_nodes_cpu =  dgl.traversal.topological_nodes_generator(bg)
topo_nodes2_cpu = dgl.traversal.topological_nodes_generator(bg2)
num_of_seqs = len(bg2.nodes()) - topo_nodes2_cpu[0].size()[0]

#seq setup
nodes_list_gpu_seqs = [] ; in_degs_list_seqs = [] ; pin_positions_list_seqs = [] ; cell_types_seq_list= [] ; first_level_inputs_list_seqs=[] ;
pushout_rise=[] ; pushout_fall = []; sr_rise=[] ; sr_fall = [];
for stage in range(1,len(topo_nodes2_cpu[1:])+1):
  nodes_list = topo_nodes2_cpu[stage].to(device) ;
  nodes_list_gpu_seqs.append(nodes_list)
  in_degs_temp = bg2.in_degrees(nodes_list).type(th.cuda.IntTensor) ; 
  in_degs_list_seqs.append( in_degs_temp )
  first_level_inputs, receiver_nodes =  bg2.in_edges( nodes_list ) ; edges_list = bg2.edge_ids(first_level_inputs, receiver_nodes)
  stage_cell_types =   (bg.ndata['cell_type'][nodes_list]).type(th.cuda.IntTensor) 
  cell_types_seq_list.append( stage_cell_types )
  pushout_rise_temp = (bg2.ndata['seq_pushoutSDF'][nodes_list][:,0]).type(th.cuda.IntTensor) ; sr_rise_temp = (bg2.ndata['seq_pushoutSDF'][nodes_list][:,2]).type(th.cuda.IntTensor)
  pushout_rise.append( pushout_rise_temp ) ; sr_rise.append( sr_rise_temp )
  pushout_fall_temp = (bg2.ndata['seq_pushoutSDF'][nodes_list][:,1]).type(th.cuda.IntTensor)  ; sr_fall_temp = (bg2.ndata['seq_pushoutSDF'][nodes_list][:,3]).type(th.cuda.IntTensor) 
  pushout_fall.append( pushout_fall_temp ) ; sr_fall.append( sr_fall_temp )
  stage_pin_positions = (bg2.edata['x'][edges_list]).type(th.cuda.CharTensor)
  pin_positions_list_seqs.append( stage_pin_positions )
  first_level_inputs_list_seqs.append(first_level_inputs) ;
first_level_inputs_seq_list_lengths = th.IntTensor([len(first_level_inputs_list_seqs[x]) for x in range(len(first_level_inputs_list_seqs))])
in_degs_seq_list_lengths = th.IntTensor([len(in_degs_list_seqs[x]) for x in range(len(in_degs_list_seqs))])
nodes_list_gpu_seqs = th.cat(nodes_list_gpu_seqs)
first_level_inputs_list_seqs = th.cat(first_level_inputs_list_seqs)
cell_types_seq_list = th.cat(cell_types_seq_list)
pushout_rise = th.cat(pushout_rise) ; sr_rise = th.cat(sr_rise) 
pushout_fall = th.cat(pushout_fall) ; sr_fall = th.cat(sr_fall)
in_degs_list_seqs = th.cat(in_degs_list_seqs)
pin_positions_list_seqs = th.cat(pin_positions_list_seqs)
edge_breakpoints_seq_start = th.roll(th.cumsum(first_level_inputs_seq_list_lengths,  dim=0), 1, 0)
edge_breakpoints_seq_size = edge_breakpoints_seq_start[0].repeat(1)
edge_breakpoints_seq_start[0] = 0
edge_breakpoints_seq_end = th.cat( (edge_breakpoints_seq_start[1:], th.IntTensor([edge_breakpoints_seq_size])), dim=0)
edge_breakpoints_seq_start = edge_breakpoints_seq_start.type(th.int32).to(device)
edge_breakpoints_seq_end = edge_breakpoints_seq_end.type(th.int32).to(device)
node_breakpoints_seq_start = th.roll(th.cumsum(in_degs_seq_list_lengths,  dim=0), 1, 0)
node_breakpoints_seq_size = node_breakpoints_seq_start[0].repeat(1)
node_breakpoints_seq_start[0] = 0
node_breakpoints_seq_end = th.cat( (node_breakpoints_seq_start[1:], th.IntTensor([node_breakpoints_seq_size])), dim=0)
node_breakpoints_seq_start = node_breakpoints_seq_start.type(th.int32).to(device)
node_breakpoints_seq_end = node_breakpoints_seq_end.type(th.int32).to(device)
per_seq_stage_num_of_nodes = (node_breakpoints_seq_end - node_breakpoints_seq_start)
num_of_seqs = in_degs_list_seqs.size()[0]
previous_states = th.zeros( (in_degs_list_seqs.size()[0],), dtype=th.int32, device=device)

first_level_inputs_seq_list_lengths = cp.asarray(first_level_inputs_seq_list_lengths)
in_degs_seq_list_lengths = cp.asarray(in_degs_seq_list_lengths)
nodes_list_gpu_seqs = cp.asarray(nodes_list_gpu_seqs).astype(cp.int32)
first_level_inputs_list_seqs  =cp.asarray(first_level_inputs_list_seqs).astype(cp.int32)
cell_types_seq_list = cp.asarray(cell_types_seq_list)
pushout_rise = cp.asarray(pushout_rise).astype(cp.uint32) ; sr_rise = cp.asarray(sr_rise).astype(cp.uint32)
pushout_fall = cp.asarray(pushout_fall).astype(cp.uint32) ; sr_fall = cp.asarray(sr_fall).astype(cp.uint32)
in_degs_list_seqs = cp.asarray(in_degs_list_seqs).astype(cp.int8)
pin_positions_list_seqs = cp.asarray(pin_positions_list_seqs).astype(cp.int8)
edge_breakpoints_seq_start = cp.asarray(edge_breakpoints_seq_start)
edge_breakpoints_seq_end = cp.asarray(edge_breakpoints_seq_end)
node_breakpoints_seq_start = cp.asarray(node_breakpoints_seq_start)
node_breakpoints_seq_end = cp.asarray(node_breakpoints_seq_end)
per_seq_stage_num_of_nodes = cp.asarray(per_seq_stage_num_of_nodes)
out_tables_cp = cp.asarray(out_tables).astype(cp.int8)
seq_out_tables_cp = cp.asarray(out_tables_seq).astype(cp.int8)


pushout_delays= cp.zeros_like(pushout_rise).astype(cp.uint32)
setPushoutDelays( (math.ceil(pushout_delays.shape[0]/512),1), (512,1), \
 (pushout_rise,pushout_fall,pushout_delays,pushout_delays.shape[0]) )
del temp; del pushout_rise; del pushout_fall;

sr_delays= cp.zeros_like(sr_rise).astype(cp.uint32)
setPushoutDelays( (math.ceil(sr_delays.shape[0]/512),1), (512,1), \
 (sr_rise,sr_fall,sr_delays,sr_delays.shape[0]) )
del sr_rise; del sr_fall;


clkTreeNodeNums = cp.asarray(th.load(args.block + "_adjustedSAIFClkTreeNodes")).astype(cp.int32)
clkTreeNodePolarities = cp.asarray(th.load(args.block + "_adjustedSAIFClkTreeNodePolarities")).astype(cp.int8)

print("setting up simulation graph...")
new_data = np.load(args.partitionsFile,allow_pickle=True)
total_num_of_nodes_all_parts=0
for part in new_data:
  total_num_of_nodes_all_parts += len(part.nodes())
no_write_node_num = total_num_of_nodes_all_parts

#first, figure out the static simulation graph
#needs a map to and from sequentials
#combo kernel will write to sequential addresses
combo_celltypes_list = [] ;  num_of_FOs_list = []; next_stage_write_addresses_list=[]; pin_positions_list=[]
next_stage_write_stage_list = [] ; next_stage_write_stage_list_within_block = [] ; endpoint_translation_list = []
per_part_node_offset=0 ; per_part_edge_offset=0 ; per_part_stage_offset=0
per_part_node_offsets=[] ; per_part_edge_offsets=[] ; per_part_stage_offsets=[]
src_node_read_addresses=[] ; bg_node_write_addresses=[] ; level0Offsets =[0]
node_position_running_tally=0; widestSizeAtStage = np.zeros(len(topo_nodes_cpu)) ; maxPartitionSize=0 ; maxPartitionStages=0;
for j in range(len(new_data)):
  part = new_data[j].to(device) ; maxPartitionSize = max(maxPartitionSize, len(part.nodes()))
  topo_nodes_part =  dgl.traversal.topological_nodes_generator(part);maxPartitionStages=max(maxPartitionStages,len(topo_nodes_part));
  part.ndata['node_level'] = th.zeros(len(part.nodes()), dtype=th.int32, device=device)
  part.ndata['node_level_within_block'] = th.zeros(len(part.nodes()), dtype=th.int32, device=device)
  part.ndata['node_position']= th.zeros_like(part.ndata['node_level'])
  part.ndata['endpoint_translation'] = th.full(size=(len(part.nodes()),), fill_value=no_write_node_num, dtype=th.int32, device=device)
  for i in range(len(topo_nodes_part)):
    nodes_list = topo_nodes_part[i].to(device)
    part.ndata['node_level'][nodes_list] = i + per_part_stage_offset
    part.ndata['node_level_within_block'][nodes_list] = i
    part.ndata['node_position'][nodes_list] = th.cuda.IntTensor(range(len(nodes_list))) + node_position_running_tally;
    node_position_running_tally += len(nodes_list)
    part.ndata['endpoint_translation'][th.where(part.ndata['mark_new_endpoints'])[0]] = \
      (part.ndata['_ID'][th.where(part.ndata['mark_new_endpoints'])[0]]).type(th.int32)
    if i==0:
      src_nodes = part.ndata['_ID'][nodes_list] ; src_node_read_addresses.append(src_nodes)
      bg_node_write_address = part.ndata['node_position'][nodes_list];
      bg_node_write_addresses.append(bg_node_write_address)
      level0Offsets.append( len(bg_node_write_address) )
  per_part_node_offsets.append(per_part_node_offset) ; per_part_edge_offsets.append(per_part_edge_offset) ; per_part_stage_offsets.append(per_part_stage_offset)
  per_part_node_offset += len(part.nodes()) ; per_part_edge_offset += len(part.edges()[0]); per_part_stage_offset += len(topo_nodes_part)
  for stage in range(len(topo_nodes_part)):
    nodes_list = topo_nodes_part[stage].to(device) ; 
    widestSizeAtStage[stage] = max(widestSizeAtStage[stage], nodes_list.size()[0])
    stage_cell_types =   (part.ndata['cell_type'][nodes_list]).type(th.cuda.IntTensor)
    combo_celltypes_list.append( stage_cell_types ) 
    num_of_FOs = part.out_degrees(nodes_list).type(th.int32)
    num_of_FOs_list.append(num_of_FOs)
    endpoint_translations = part.ndata['endpoint_translation'][nodes_list]
    endpoint_translation_list.append(endpoint_translations)
    pin_positions = (part.edata['x'][part.edge_ids(part.out_edges(nodes_list)[0],part.out_edges(nodes_list)[1])]-1).type(th.int8)
    pin_positions_list.append(pin_positions)
    thread_address = part.ndata['node_position'][(part.out_edges(nodes_list)[1])]
    next_stage_write_addresses_list.append(thread_address)
    thread_address2 = part.ndata['node_level'][(part.out_edges(nodes_list)[1])]
    thread_address3 = part.ndata['node_level_within_block'][(part.out_edges(nodes_list)[1])]
    next_stage_write_stage_list.append(thread_address2)
    next_stage_write_stage_list_within_block.append(thread_address3)
  new_data[j] = part
per_part_stage_offsets.append(per_part_stage_offset)
level0Offsets = cp.asarray(level0Offsets).astype(cp.int32)
level0Offsets = cp.cumsum(level0Offsets,  axis=0)
maxPartitionStages +=1
#simulateLogicCode is RESERVED in GL0AM.cupy
#need to set constant memory items for per_part_stage_offsets 
simulateLogicCode = simulateLogicCode.replace("PARTS", f"{len(per_part_stage_offsets)}")
simulateLogicCode = simulateLogicCode.replace("LOGIC_LENGTH", f"{compactOutTables.shape[0]}")
simulateLogicModule = cp.RawModule(code=simulateLogicCode, options=('-lineinfo',),backend='nvcc')
simulate_stage3dBlockSyncConstantMemory = simulateLogicModule.get_function("simulate_stage3dBlockSyncConstantMemory")
simulate_stage3dBlockSyncConstantMemory.max_dynamic_shared_size_bytes = 49152*2
#simulate_stage3dBlockSyncConstantMemory.preferred_shared_memory_carveout =75
#simulate_reg2combo_interfaceBlockSyncConstantMemory = simulateLogicModule.get_function("simulate_reg2combo_interfaceBlockSyncConstantMemory")
per_part_stage_offsetsPtr = simulateLogicModule.get_global("per_part_stage_offsets")
per_part_stage_offsetsGpu = cp.ndarray(len(per_part_stage_offsets), cp.int32, per_part_stage_offsetsPtr)
per_part_stage_offsetsGpu[...] = cp.asarray(per_part_stage_offsets).astype(cp.int32)
compactOutTablesGpuPtr = simulateLogicModule.get_global("logics")
compactOutTablesGpu = cp.ndarray(compactOutTables.shape[0], cp.uint32, compactOutTablesGpuPtr)
compactOutTablesGpu[...] = compactOutTables ; del compactOutTables
three_powPtr = simulateLogicModule.get_global("three_pow")
three_powGpu = cp.ndarray(8, cp.int32, three_powPtr)
three_powGpu[...] = cp.array([1, 3, 9, 27, 81, 243, 729, 2187], dtype=cp.int32) ;
level0OffsetsPtr = simulateLogicModule.get_global("level0Offsets")
level0OffsetsGpu = cp.ndarray(len(level0Offsets), cp.int32, level0OffsetsPtr)
level0OffsetsGpu[...] = level0Offsets

#flatten the simulation graph
node_nums = node_nums.to(device)
node_nums = cp.asarray(node_nums).astype(cp.int32)

next_stage_write_addresses_list=th.cat(next_stage_write_addresses_list).type(th.int32)
next_stage_write_stage_list=th.cat(next_stage_write_stage_list).type(th.int32)
next_stage_write_stage_list_within_block=th.cat(next_stage_write_stage_list_within_block).type(th.int32)
pin_positions_list=th.cat(pin_positions_list).type(th.int8)
combo_celltypes_list_lengths = th.IntTensor([len(combo_celltypes_list[x]) for x in range(len(combo_celltypes_list))])
combo_celltypes_list = th.cat(combo_celltypes_list)
endpoint_translation_list = th.cat(endpoint_translation_list)
num_of_FOs_list=th.cat(num_of_FOs_list)
edge_breakpoints = th.cat( (th.cuda.IntTensor([0]),th.cumsum(num_of_FOs_list,  dim=0, dtype=th.int32)) )
node_breakpoints_start = th.roll(th.cumsum(combo_celltypes_list_lengths,  dim=0), 1, 0)
node_breakpoints_size = node_breakpoints_start[0].repeat(1)
node_breakpoints_start[0] = 0
node_breakpoints_end = th.cat( (node_breakpoints_start[1:], th.IntTensor([node_breakpoints_size])), dim=0)
node_breakpoints_start = th.cat( (node_breakpoints_start, th.IntTensor([node_breakpoints_size])), dim=0)
node_breakpoints_start = node_breakpoints_start.type(th.int32).to(device)
node_breakpoints_end = node_breakpoints_end.type(th.int32).to(device)
per_stage_num_of_nodes = (node_breakpoints_end - node_breakpoints_start[0:-1])

src_node_read_addresses = th.cat(src_node_read_addresses)
bg_node_write_addresses= th.cat(bg_node_write_addresses)
 

next_stage_write_addresses_list = cp.asarray(next_stage_write_addresses_list).astype(cp.int32)
next_stage_write_stage_list = cp.asarray(next_stage_write_stage_list).astype(cp.int32)
next_stage_write_stage_list_within_block = cp.asarray(next_stage_write_stage_list_within_block).astype(cp.int32)
pin_positions_list = cp.asarray(pin_positions_list).astype(cp.int8)
combo_celltypes_list = cp.asarray(combo_celltypes_list).astype(cp.int32)
combo_celltypes_list = (comboCelltypeOffsets[combo_celltypes_list]) ; del comboCelltypeOffsets;
combo_celltypes_list = cp.concatenate( (combo_celltypes_list, cp.asarray([0]).astype(cp.int32)) ) if (combo_celltypes_list.shape[0] % 2) else combo_celltypes_list
combo_celltypes_list_evens = combo_celltypes_list[0::2]
combo_celltypes_list_odds = combo_celltypes_list[1::2] << 16
combo_celltypes_list = cp.bitwise_or(combo_celltypes_list_evens,combo_celltypes_list_odds, dtype=cp.int32)
del combo_celltypes_list_evens,combo_celltypes_list_odds;
endpoint_translation_list = cp.asarray(endpoint_translation_list).astype(cp.int32) #long
num_of_FOs_list = cp.asarray(num_of_FOs_list).astype(cp.int32)
edge_breakpoints = cp.asarray(edge_breakpoints).astype(cp.int32) #int32
node_breakpoints_start = cp.asarray(node_breakpoints_start).astype(cp.int32) #int32
node_breakpoints_end = cp.asarray(node_breakpoints_end).astype(cp.int32) #int32
node_breakpoints_start = cp.concatenate( (node_breakpoints_start,cp.asarray([node_breakpoints_end[-1]])), dtype=cp.int32 )
per_stage_num_of_nodes = cp.asarray(per_stage_num_of_nodes).astype(cp.int32) #int32
per_stage_num_of_nodes_cp_list = per_stage_num_of_nodes

src_node_read_addresses = cp.asarray(src_node_read_addresses).astype(cp.int32)
bg_node_write_addresses = cp.asarray(bg_node_write_addresses).astype(cp.int32)

                                      
previous_states = cp.asarray(previous_states).astype(cp.int8)
                                                                              

in_edges9=cp.asarray(th.full( size = (int(node_breakpoints_seq_end[-1]),8), dtype=th.int32, device=device, fill_value = gnd_node)).astype(cp.int32)
in_edges9[:,3:7]  = vdd_node
organize_input_addresses_for_seqs( (math.ceil(in_degs_list_seqs.shape[0]/threads_per_block),), (threads_per_block,), \
      (first_level_inputs_list_seqs, pin_positions_list_seqs,in_degs_list_seqs,in_edges9,cell_types_seq_list,\
      gnd_node, in_degs_list_seqs.shape[0] ) )

#may need to set some ties on sequential cells to tie hi
tie_info = np.load(args.block + '_tie_info', allow_pickle=True) if os.path.getsize(args.block + '_tie_info') else { 'node_nums' : cp.array([]), 'pin_positions' : cp.array([])}
tie_node_nums = cp.asarray(tie_info['node_nums']).astype(cp.int32)
tie_pin_positions = cp.asarray(tie_info['pin_positions']).astype(cp.int8)
in_edges9_indexes = cp.zeros_like(tie_node_nums)

map_to_new_nodes( (math.ceil(tie_node_nums.shape[0]/1024),), (1024,), (tie_node_nums, nodes_list_gpu_seqs, \
  in_edges9_indexes, \
  tie_node_nums.shape[0], nodes_list_gpu_seqs.shape[0] ) )
tie_some_highs( (math.ceil(tie_node_nums.shape[0]/1024),), (1024,), (tie_node_nums, in_edges9_indexes, \
  tie_pin_positions,in_edges9,cell_types_seq_list, \
  vdd_node, tie_node_nums.shape[0]) ) 

per_stage_num_of_nodes_cp_list = per_stage_num_of_nodes
simulate_stage_blocks = min(320,math.ceil(cp.max(per_stage_num_of_nodes_cp_list)/threads_per_block))

# +
#setup rams
total_ram_bits = int(th.sum(bg.ndata['cell_type'] >=500))
bla=th.unique(bg.ndata['cell_type'])
num_rams=len(bla[bla>=500])
bg3=bg3.to(device)
ram_id = [] ; di_address = [];  dbyp_address = []  ; dout_address = []
clk_ra_re_ore_wa_we_bypsel_address = th.zeros( (num_rams,29) , dtype=th.int32)
#defaults to main clk
clk_ra_re_ore_wa_we_bypsel_address[:,0] = clk_node_num
clk_ra_re_ore_wa_we_bypsel_address[:,1:14] = gnd_node
clk_ra_re_ore_wa_we_bypsel_address[:,14] = len(bg.nodes())+1
clk_ra_re_ore_wa_we_bypsel_address[:,15:] = gnd_node
for i in range(num_rams):
  this_ram_id = th.full( (ram_cols[i],), i ,dtype=th.int32 ) ; ram_id.append(this_ram_id)
  this_ram_cell_type = i+500
  ram_nodes = (bg.ndata['cell_type'] == this_ram_cell_type).nonzero().squeeze(1)
  pick_a_ram_node = ram_nodes[0] ; prefix = cell_names[int(pick_a_ram_node)].split('[')[0]
  in_nodes = bg3.in_edges(pick_a_ram_node)[0] ; in_edge_ids = bg3.edge_ids(bg3.in_edges(pick_a_ram_node)[0],bg3.in_edges(pick_a_ram_node)[1])
  for j in range(29):
    test = (bg3.edata['x'][in_edge_ids] == (1000+j)).nonzero().squeeze(1).size()[0]
    if test > 0:
      found_node = in_nodes[(bg3.edata['x'][in_edge_ids] == (1000+j)).nonzero().squeeze(1)[0]]
      found_node = int(found_node)
      clk_ra_re_ore_wa_we_bypsel_address[i,j] = found_node
  for j in range(ram_cols[i]):
    test = (bg3.edata['x'][in_edge_ids] == (5000+j)).nonzero().squeeze(1).size()[0]
    found_node = in_nodes[(bg3.edata['x'][in_edge_ids] == (5000+j)).nonzero().squeeze(1)[0]] if (test>0) else gnd_node
    found_node = int(found_node) if (test>0) else gnd_node
    di_address.append(int(found_node))
    test = (bg3.edata['x'][in_edge_ids] == (6000+j)).nonzero().squeeze(1).size()[0]
    found_node = in_nodes[(bg3.edata['x'][in_edge_ids] == (6000+j)).nonzero().squeeze(1)[0]] if (test>0) else len(bg.nodes())
    found_node = int(found_node) if (test>0) else len(bg.nodes())
    dbyp_address.append(int(found_node))
    out_name = prefix + '[' + str(j) + ']' ; found_node = cell_names[out_name]
    found_node = int(found_node)
    dout_address.append(found_node)
if len(ram_id):
  ram_id = th.cat(ram_id).type(th.int32)

total_ram_memory_bits=0 ; M_breakpoints =[0]; ram_col_widths = [] ; ram_thread_breakpoints = [0]
ram_thread_breakpoint=0
for i in range(num_rams):
  total_ram_memory_bits += ram_cols[i] * ram_rows[i]
  ram_thread_breakpoint += ram_cols[i]
  M_breakpoints.append(total_ram_memory_bits)
  ram_col_widths.append(ram_cols[i])
  ram_thread_breakpoints.append(ram_thread_breakpoint)
M_01 = cp.zeros( (total_ram_memory_bits,), dtype=cp.bool_)
M_XZ = cp.ones( (total_ram_memory_bits,), dtype=cp.bool_)
ra_x_address = 9999999
ra_d = cp.full( (num_rams,), fill_value=ra_x_address, dtype=cp.int32)
ram_id = cp.asarray(ram_id)
clk_ra_re_ore_wa_we_bypsel_address = cp.asarray(clk_ra_re_ore_wa_we_bypsel_address)
di_address = cp.asarray(di_address).astype(cp.int32)
dout_address = cp.asarray(dout_address).astype(cp.int32)
dbyp_address = cp.asarray(dbyp_address).astype(cp.int32)
M_breakpoints = cp.asarray(M_breakpoints[0:-1]).astype(cp.int32)
ram_thread_breakpoints = cp.asarray(ram_thread_breakpoints[0:-1]).astype(cp.int32)
ram_col_widths = cp.asarray(ram_col_widths).astype(cp.int32)
address_space_width = 29

#make tensors for all clock signals
bnode_nums = cp.asarray(node_nums).astype(cp.int32)
clk_node_nums = [] ; clk_input_values_array_indexes = []
for this_clk in args.clk[0]:
  this_clk_node_num = cell_names[this_clk]
  clk_node_nums.append(this_clk_node_num)
  this_clk_input_values_array_index = int((bnode_nums==this_clk_node_num).nonzero()[0])
  clk_input_values_array_indexes.append(this_clk_input_values_array_index)
clk_node_nums = cp.asarray(clk_node_nums).astype(cp.int32)
clk_input_values_array_indexes = cp.asarray(clk_input_values_array_indexes).astype(cp.int32)


# -
threads_per_block2=1024 ; dynamicSharedMemSize = maxPartitionStages*8 ; 
assert dynamicSharedMemSize <= 98304, 'check dynamic shared mem size, may need reconfiguration, dynamicSharedMemSize=' + str(dynamicSharedMemSize)
def sync_event():
  setClocks( (math.ceil(clk_node_nums.shape[0]/128),), (128,),\
    (input_values_array,current_logic_value,clk_node_nums,clk_input_values_array_indexes,\
    clk_node_nums.shape[0],input_values_array.shape[1],cc) )
  for stage in range(node_breakpoints_seq_start.shape[0]):
    simulateSequentialStage( (math.ceil(per_seq_stage_num_of_nodes_cp_list[stage]/threads_per_block),), (threads_per_block,), \
    (current_logic_value,output_waveforms_storage,in_edges9,cell_types_seq_list,\
    nodes_list_gpu_seqs,node_breakpoints_seq_start,seq_out_tables_cp,temp_active_timestamps,\
    global_vc_write_pointer,pushout_delays,numOfSwitchesPerDriverPerSplit,stage,cc,\
    per_seq_stage_num_of_nodes_cp_list[stage],zeroDelaySplits,zeroDelayFoldSplit) )
  simulate_rams( (math.ceil(total_ram_bits/512),1), (512,1),\
    (current_logic_value, M_01,M_XZ, ra_d, ram_id,clk_ra_re_ore_wa_we_bypsel_address,\
     di_address,dout_address,dbyp_address,M_breakpoints,ram_thread_breakpoints,ram_col_widths,temp_active_timestamps,\
     global_vc_write_pointer,numOfSwitchesPerDriverPerSplitRAMs,output_waveforms_storage,\
     cc,address_space_width,ra_x_address,total_ram_bits,zeroDelaySplits,zeroDelayFoldSplit ) )
  set_inputs( (math.ceil(input_values_array.shape[0]/threads_per_block),), (threads_per_block,), \
    (cc,input_values_array,current_logic_value,bnode_nums,input_values_array.shape[0],input_values_array.shape[1]) )
  simulate_stage3dBlockSyncConstantMemory( (len(new_data),), (threads_per_block2,), \
    (edgeWriteInfo,current_logic_value,endpoint_translation_list,node_breakpoints_start,nodeWriteInfo,\
    combo_celltypes_list,per_thread_msg,src_node_read_addresses,register_combo_interface_previous_states,\
    edge_breakpoints,threads_per_block2 ), shared_mem=dynamicSharedMemSize)


def async_event():
  set_inputs( (math.ceil(input_values_array.shape[0]/threads_per_block),), (threads_per_block,), \
    (cc,input_values_array,current_logic_value,bnode_nums,input_values_array.shape[0],input_values_array.shape[1]) )
  simulate_stage3dBlockSyncConstantMemory( (len(new_data),), (threads_per_block2,), \
    (edgeWriteInfo,current_logic_value,endpoint_translation_list,node_breakpoints_start,nodeWriteInfo,\
    combo_celltypes_list,per_thread_msg,src_node_read_addresses,register_combo_interface_previous_states,\
    edge_breakpoints,threads_per_block2 ), shared_mem=dynamicSharedMemSize)
  continueLoop=1
  while continueLoop:
    global_vc_write_pointerTemp = global_vc_write_pointer.repeat(1)
    for stage in range(node_breakpoints_seq_start.shape[0]):
      simulateSequentialStage( (math.ceil(per_seq_stage_num_of_nodes_cp_list[stage]/threads_per_block),), (threads_per_block,), \
       (current_logic_value,output_waveforms_storage,in_edges9,cell_types_seq_list,\
       nodes_list_gpu_seqs,node_breakpoints_seq_start,seq_out_tables_cp,temp_active_timestamps,\
       global_vc_write_pointer,sr_delays,numOfSwitchesPerDriverPerSplit,stage,cc,\
       per_seq_stage_num_of_nodes_cp_list[stage],zeroDelaySplits,zeroDelayFoldSplit) )
    simulate_rams( (math.ceil(total_ram_bits/512),1), (512,1),\
     (current_logic_value, M_01,M_XZ, ra_d, ram_id,clk_ra_re_ore_wa_we_bypsel_address,\
     di_address,dout_address,dbyp_address,M_breakpoints,ram_thread_breakpoints,ram_col_widths,temp_active_timestamps,\
     global_vc_write_pointer,numOfSwitchesPerDriverPerSplitRAMs,output_waveforms_storage,\
     cc,address_space_width,ra_x_address,total_ram_bits,zeroDelaySplits,zeroDelayFoldSplit ) )
    simulate_stage3dBlockSyncConstantMemory( (len(new_data),), (threads_per_block2,), \
     (edgeWriteInfo,current_logic_value,endpoint_translation_list,node_breakpoints_start,nodeWriteInfo,\
     combo_celltypes_list,per_thread_msg,src_node_read_addresses,register_combo_interface_previous_states,\
     edge_breakpoints,threads_per_block2 ), shared_mem=dynamicSharedMemSize)
    continueLoop = global_vc_write_pointerTemp != global_vc_write_pointer


# +
#the simulation loop
#simulation variable setup
sync_or_async_event_processed = sync_or_async_event_processed.get()
per_seq_stage_num_of_nodes_cp_list=[int(x.get()) for x in per_seq_stage_num_of_nodes]
cycles=change_list_size
sim_duration = args.duration
other_end_token=4294964295
zeroDelaySplits = 80;
zeroDelayFoldSplit = math.ceil(args.duration / zeroDelaySplits )
output_waveforms_storage = cp.full( 2500000000, dtype=cp.uint64,  fill_value = 0 )
current_cycle = cp.asarray([0]).astype(cp.int32)
previous_states = cp.asarray(th.zeros( (in_degs_list_seqs.shape[0],), dtype=th.int8, device=device)).astype(cp.int8)
theres_an_event = cp.zeros( node_breakpoints_start.shape[0], dtype=cp.int32 )
current_logic_value = cp.full( (len(bg.nodes())+2,), fill_value=2, dtype=cp.int8)  
size_of_widest_logic_stage=int(cp.max(per_stage_num_of_nodes_cp_list)) 

newg = dgl.batch(new_data)
values,indices=th.sort(newg.ndata['node_position'])
per_thread_msg= cp.asarray(3**(newg.in_degrees(indices))-1).astype(cp.int32) + 16384 #0100_0000_0000_0000
register_combo_interface_previous_states = cp.full( (src_node_read_addresses.shape[0],), fill_value=2, dtype=cp.int8)
del newg

current_logic_value[-1]=9
M_01 = cp.zeros( (total_ram_memory_bits,), dtype=cp.bool_)
M_XZ = cp.ones( (total_ram_memory_bits,), dtype=cp.bool_)
ra_x_address = 9999999
ra_d = cp.full( (num_rams,), fill_value=ra_x_address, dtype=cp.int32)
per_node_activity=cp.zeros( (int(register_combo_interface_previous_states.shape[0]),), dtype=cp.int8)

global_vc_write_pointer = cp.array([0]).astype(cp.uint32)
num_of_drivers = nodes_list_gpu_seqs.shape[0]
numOfSwitchesPerDriverPerSplit= cp.zeros( (num_of_drivers,zeroDelaySplits), dtype=cp.uint32)
numOfSwitchesPerDriverPerSplitRAMs = cp.zeros( (dout_address.shape[0],zeroDelaySplits), dtype=cp.uint32)

nodeWriteInfo = cp.bitwise_or( ((endpoint_translation_list != no_write_node_num).astype(cp.int32) << 31),\
   ((edge_breakpoints[1:] - edge_breakpoints[0:-1]) << 26), dtype=cp.int32)
nodeWriteInfo = cp.bitwise_or(nodeWriteInfo,edge_breakpoints[0:-1], dtype=cp.int32)

edgeWriteInfo = cp.bitwise_or((next_stage_write_stage_list_within_block.astype(cp.uint32) << 25),\
  (next_stage_write_addresses_list.astype(cp.uint32) << 3), dtype=cp.uint32)
del next_stage_write_stage_list_within_block
edgeWriteInfo = cp.bitwise_or(edgeWriteInfo,pin_positions_list.astype(cp.uint32), dtype=cp.int32)

delta_compile = timer() -start_compile
print('waveform and partitioned graphs creation compile time is: ' + str(delta_compile))


# +
temp_start = timer() ;
##### per event loop 
for cc in range(change_list_size):
  if sync_or_async_event_processed[cc]:
    async_event()
  else:
    sync_event()
temp_add = timer() -temp_start
print("0-delay sim done in " + str(temp_add))

#For sanity 'display' value to check if 0-delay simulation is correct or not
def report_results(signal_name, bus_big_endian, bus_little_endian, bus_or_no):
  list_of_node_values_of_a_reg=[] ; print_string='' ;
  if bus_or_no:
    for i in range(bus_big_endian,bus_little_endian-1,-1):
      signal_bit_name = signal_name + '[' + str(i) + ']'
      pin_name = cell_dict[signal_bit_name]
      if signal_bit_name in cell_names.keys():
        orig_pin_num = cell_names[signal_bit_name]
      else:
        orig_pin_num = cell_names[pin_name]
      new_pin_num = int(orig_pin_num)
      list_of_node_values_of_a_reg.append(str(int(current_logic_value[new_pin_num])))
    list_of_node_values_of_a_reg = "".join(list_of_node_values_of_a_reg)
    print_string += signal_name + '[' + str(bus_big_endian) + ':' + str(bus_little_endian) + ']' + " : " + str(hex(int(list_of_node_values_of_a_reg, base=2)))
    print(print_string)
  else:
    pin_name = cell_dict[signal_name]
    orig_pin_num = cell_names[pin_name]
    new_pin_num = int(orig_pin_num)
    list_of_node_values_of_a_reg.append(str(int(current_logic_value[new_pin_num])))
    print_string += signal_name + " : " + str(int(list_of_node_values_of_a_reg[0]))
    print(print_string)

for signal in args.zeroDelayReportSignals[0]:
  if '[' in signal:
    busOrNo = 1
    signalName = signal.split('[')[0].strip()
    bigEndian = int(signal.split('[')[1].split(':')[0]) ; smallEndian = int(signal.split(']')[0].split(':')[1])
    report_results(signalName,bigEndian,smallEndian,busOrNo)
  else:
    busOrNo = 0
    report_results(signal,0,0,0)

# +
#change signal storage format from vcd-like 'timestamp based' to 'signal based' for re-simulation
start=timer()
end_token = 3221222471
end_token_timestamp = 1073738823
end_token_fullWaveform = 9223372036854775800
num_of_timestamps = global_vc_write_pointer.get()[0]
output_waveforms_storage = output_waveforms_storage[0:num_of_timestamps]
mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()

numOfRegisters = numOfSwitchesPerDriverPerSplit.shape[0]
numOfSwitchesPerDriverPerSplit = cp.concatenate( (numOfSwitchesPerDriverPerSplit,numOfSwitchesPerDriverPerSplitRAMs), axis=0)
nodes_list_gpu_seqs = cp.concatenate( (nodes_list_gpu_seqs,dout_address)) ; num_of_drivers = nodes_list_gpu_seqs.shape[0]
del numOfSwitchesPerDriverPerSplitRAMs

foldSplitBeginIndexes= cp.sum(numOfSwitchesPerDriverPerSplit, axis=0)
foldSplitBeginIndexes=cp.roll(cp.cumsum(foldSplitBeginIndexes), 1)
foldSplitBeginIndexes[0] = 0
foldSplitBeginIndexes = cp.concatenate((foldSplitBeginIndexes, cp.asarray([num_of_timestamps], dtype=cp.uint64))).astype(cp.uint32)

numOfSwitchesPerDriverPerSplit[:,0] +=1
numOfSwitchesPerDriverPerSplit[:,-1] +=1
pointers_temp = cp.cumsum(numOfSwitchesPerDriverPerSplit).astype(cp.uint32).reshape(-1)
pointers_temp = cp.concatenate( (cp.array([0]).astype(cp.uint32),pointers_temp) )
perDriverPerSplitPointersBegin = pointers_temp[0:-1].reshape(-1,zeroDelaySplits)
perDriverPerSplitPointersEnd = pointers_temp[1:].reshape(-1,zeroDelaySplits)
del pointers_temp

resimAllWaveformLength= num_of_timestamps + nodes_list_gpu_seqs.shape[0]*2
resimAllWaveformLength = resimAllWaveformLength + saveInputWaveformsTotal.shape[0]
theConcatenatePoint = num_of_timestamps + nodes_list_gpu_seqs.shape[0]*2

saveRegisterWaveformsTotal = cp.zeros(resimAllWaveformLength, dtype=cp.uint64)
perDriverPerSplitPointersBegin_tracker = perDriverPerSplitPointersBegin.repeat(1,1)
perDriverPerSplitPointersBegin_tracker[:,0] += 1
index_T = cp.zeros(len(bg.nodes()), dtype=cp.int32)
setIndex_T(  (math.ceil(num_of_drivers/512),1), (512,1), \
 (nodes_list_gpu_seqs,index_T,num_of_drivers) )
setFullWaveformRegisterPseudoInputsBlockProcess(  (zeroDelaySplits,1), (1,1), \
 (output_waveforms_storage,foldSplitBeginIndexes,perDriverPerSplitPointersBegin_tracker,\
 saveRegisterWaveformsTotal,index_T,zeroDelaySplits) )
saveRegisterWaveformsTotal[perDriverPerSplitPointersBegin[:,0]] = 9223372036854775808
saveRegisterWaveformsTotal[perDriverPerSplitPointersEnd[:,-1]-1] = end_token_fullWaveform
del perDriverPerSplitPointersBegin_tracker

nodes_list_gpu_seqs = cp.concatenate( (nodes_list_gpu_seqs,saveInputNodeNums) ).astype(cp.int32)
saveInputPointers += theConcatenatePoint
saveRegisterWaveformsTotal[theConcatenatePoint:] = saveInputWaveformsTotal
saveRegisterPointers = perDriverPerSplitPointersBegin[:,0].astype(cp.int64)
saveRegisterPointers = cp.concatenate( (saveRegisterPointers,saveInputPointers,cp.array([saveRegisterWaveformsTotal.shape[0]], dtype=cp.int64)) )
#print('total inputs saved: ' + str(saveRegisterWaveformsTotal.shape[0]))
del saveInputNodeNums, saveInputPointers, saveInputWaveformsTotal

mempool = cp.get_default_memory_pool()
del first_level_inputs_seq_list_lengths;
del in_degs_seq_list_lengths;
del first_level_inputs_list_seqs;
del cell_types_seq_list;
del in_degs_list_seqs;
del pin_positions_list_seqs;
del edge_breakpoints_seq_start;
del edge_breakpoints_seq_end;
del per_seq_stage_num_of_nodes;
del seq_out_tables_cp;
del node_nums;
del input_values_array;
del next_stage_write_addresses_list;
del pin_positions_list;
del combo_celltypes_list;
del num_of_FOs_list;
del edge_breakpoints;
del node_breakpoints_start;
del node_breakpoints_end;
del per_stage_num_of_nodes;
del per_stage_num_of_nodes_cp_list;
del in_edges9;
del output_waveforms_storage;
del current_cycle;
del previous_states;
del theres_an_event;
del current_logic_value;
del per_thread_msg;
del register_combo_interface_previous_states;
del global_vc_write_pointer;
del foldSplitBeginIndexes;
del per_part_stage_offsetsGpu;
del compactOutTablesGpu;

del perDriverPerSplitPointersBegin;
del perDriverPerSplitPointersEnd;
del bnode_nums;

mempool.free_all_blocks()

delta = timer() - start
print("signal T time: " + str(delta))


# +
fileObject = open(args.graph, 'rb') ; bg = pickle.load(fileObject) ; bg = bg.to(device)
bg.edata['net_delay_rise'] = (th.round( (bg.edata['net_delay_rise'] *1000/args.timescale) + 0.001 ) * args.timescale).type(th.cuda.IntTensor)
bg.edata['net_delay_fall'] = (th.round( (bg.edata['net_delay_fall'] *1000/args.timescale) + 0.001 ) * args.timescale).type(th.cuda.IntTensor)
bg.ndata['waveform_start'] = th.cuda.LongTensor( [2*x for x in range(args.num_splits)] ).unsqueeze(0).repeat(len(bg.nodes()), 1).type(th.int64)
bg.ndata['waveform_end'] = th.cuda.LongTensor( [2*x+2 for x in range(args.num_splits)] ).unsqueeze(0).repeat(len(bg.nodes()), 1).type(th.int64)
mempool.free_all_blocks()
#This is the GATSPI part, load the delays
#need to revert back to original, non maxFO processed combo graph
full_delays = th.load(args.block + "_fullSDF_delays")
full_delays = (th.round( (full_delays *1000/args.timescale) + 0.001 ) * args.timescale ).type(th.cuda.IntTensor)
full_delays = cp.asarray(full_delays).astype(cp.uint32)
full_delays[0:1024]=0
X_delays = th.load(args.block + "_XCondSDF_delays")
X_delays = (th.round( (X_delays *1000/args.timescale) + 0.001 ) * args.timescale ).type(th.cuda.IntTensor)
X_delays = cp.asarray(X_delays).astype(cp.uint32)
topo_nodes = dgl.traversal.topological_nodes_generator(bg)
topo_nodes = list(topo_nodes)
for topo in range(len(topo_nodes)):
  topo_nodes[topo] = topo_nodes[topo].to(device)
bg.ndata['node_level'] = th.zeros(len(bg.nodes()), dtype=th.int32, device=device)

chunk_id=0
# -

start=timer()
saveRegisterPointers_tracker = cp.asarray(saveRegisterPointers.repeat(1))
fold_split = math.ceil( math.ceil( args.duration / (args.num_splits * args.num_of_sub_chunks) ) / args.period ) * args.period
for subchunk_id in range(args.num_of_sub_chunks):
  print("Begin simulating subchunk: " + str(subchunk_id + 1))
  subchunkStart = subchunk_id*args.num_splits*fold_split ; simTime0Flag = 1 if subchunkStart == 0 else 0;
  subchunkEnd = args.duration if (subchunk_id == (args.num_of_sub_chunks-1) ) else subchunkStart + args.num_splits*fold_split
  start_waveform_creation = timer()
  if(subchunk_id>0):
    del new_waveforms_total
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
  saveRegisterWaveformsTotal=cp.asarray(saveRegisterWaveformsTotal)
  saveRegisterPointers= cp.asarray(saveRegisterPointers)
  initValuesOfSubchunk = cp.zeros( (nodes_list_gpu_seqs.shape[0]), dtype=cp.uint8)
  thisSubchunkStartPointer = cp.zeros( (nodes_list_gpu_seqs.shape[0]), dtype=cp.int64)
  numOfPerDriverPerSplitSwitches = cp.zeros( (nodes_list_gpu_seqs.shape[0],args.num_splits), dtype=cp.uint32)
  calculateNumberOfPerSplitSwitches( (math.ceil(nodes_list_gpu_seqs.shape[0]/512),1), (512,1), \
   ( saveRegisterWaveformsTotal,saveRegisterPointers,saveRegisterPointers_tracker,initValuesOfSubchunk,\
   numOfPerDriverPerSplitSwitches,thisSubchunkStartPointer,fold_split,args.num_splits,nodes_list_gpu_seqs.shape[0],subchunkStart,subchunkEnd) ) 
  numOfPerDriverPerSplitSwitches= numOfPerDriverPerSplitSwitches+2
  totalDriverWaveformSize = int(cp.sum(numOfPerDriverPerSplitSwitches))
  perDriverPerSplitPointers = cp.cumsum(numOfPerDriverPerSplitSwitches.reshape(-1))
  perDriverPerSplitPointersBegin = (cp.concatenate( (cp.asarray([0], dtype=cp.uint64), perDriverPerSplitPointers[:-1]) )).reshape(-1,args.num_splits)
  perDriverPerSplitPointersEnd = (perDriverPerSplitPointers).reshape(-1,args.num_splits)
  del perDriverPerSplitPointers ; del numOfPerDriverPerSplitSwitches
  driverWaveformsStorage = cp.full(totalDriverWaveformSize, fill_value = end_token, dtype=cp.uint32)
  perDriverPerSplitPointersBegin_tracker = perDriverPerSplitPointersBegin.repeat(1,1)
  writeDriverWaveforms( (math.ceil(nodes_list_gpu_seqs.shape[0]/512),1), (512,1),\
   (saveRegisterWaveformsTotal,thisSubchunkStartPointer,saveRegisterPointers_tracker,perDriverPerSplitPointersBegin,\
   perDriverPerSplitPointersBegin_tracker,initValuesOfSubchunk,driverWaveformsStorage,fold_split,args.num_splits,\
   nodes_list_gpu_seqs.shape[0],subchunkStart,subchunkEnd) )
  bg.ndata['waveform_start'] = th.cuda.LongTensor( [2*x for x in range(args.num_splits)] ).unsqueeze(0).repeat(len(bg.nodes()), 1).type(th.int64)
  bg.ndata['waveform_end'] = th.cuda.LongTensor( [2*x+2 for x in range(args.num_splits)] ).unsqueeze(0).repeat(len(bg.nodes()), 1).type(th.int64)
  bg.ndata['waveform_start'][th.cuda.LongTensor(nodes_list_gpu_seqs)] = th.cuda.LongTensor(perDriverPerSplitPointersBegin.astype(cp.int64)) + 2*args.num_splits
  bg.ndata['waveform_end'][th.cuda.LongTensor(nodes_list_gpu_seqs)] = th.cuda.LongTensor(perDriverPerSplitPointersEnd.astype(cp.int64)) + 2*args.num_splits
  saveRegisterWaveformsTotal = saveRegisterWaveformsTotal.get()
  saveRegisterPointers = saveRegisterPointers.get()
  del initValuesOfSubchunk, thisSubchunkStartPointer, perDriverPerSplitPointersBegin ;
  del perDriverPerSplitPointersEnd, perDriverPerSplitPointersBegin_tracker ;
  mempool = cp.get_default_memory_pool()
  mempool.free_all_blocks()
  new_waveforms_total = cp.full( (6000000000,), end_token, dtype=cp.uint32 )
  new_waveforms_total[64:totalDriverWaveformSize+64] = driverWaveformsStorage
  new_waveforms_total[0:64:2] = 2147483648
  append_length = 64+totalDriverWaveformSize
  #clean up this chunk/subchunk's waveform creation process.
  #we need to move some saved tensors back to the CPU to not clog GPU memory
  #intermediate GPU results are deleted
  del driverWaveformsStorage ;  mempool.free_all_blocks()
  TC=cp.zeros( ( len(bg.nodes()), ) , dtype=cp.uint32)
  dt = timer()- start_waveform_creation
  print("Waveform creation time: " + str(dt))
  for stage in range(1,len(topo_nodes[1:])+1):
    nodes_list = topo_nodes[stage] ; nodes_list_cp = cp.asarray(nodes_list)
    in_degs = cp.asarray(bg.in_degrees(nodes_list)).astype(cp.uint8)
    edge_breakpoints = cp.roll(cp.cumsum(in_degs).astype(cp.uint32),1)
    edge_breakpoints[0]=0
    first_level_inputs, receiver_nodes =  bg.in_edges( nodes_list ) ; edges_list = bg.edge_ids(first_level_inputs, receiver_nodes) 
    celltypes = cp.asarray(bg.ndata['cell_type'][nodes_list]).astype(cp.uint32)
    waveform_pointers = cp.asarray(bg.ndata['waveform_start'][first_level_inputs])
    net_delays_rise= cp.asarray(bg.edata['net_delay_rise'][edges_list]).astype(cp.uint32)
    net_delays_fall= cp.asarray(bg.edata['net_delay_fall'][edges_list]).astype(cp.uint32)
    XCond_pointers= cp.asarray(bg.edata['XCond_pointers'][edges_list]).astype(cp.uint32)
    pin_positions= cp.asarray(bg.edata['x'][edges_list] - 1).astype(cp.uint8)
    delay_pointer_start= cp.asarray(bg.edata['start_pointers'][edges_list] ).astype(cp.uint32) ; 
    delay_pointer_end = cp.asarray(bg.edata['end_pointers'][edges_list] ).astype(cp.uint32)
    output_TC=cp.zeros( (in_degs.shape[0],args.num_splits) , dtype=cp.uint32)
    simulate_gate_TC(  (math.ceil(nodes_list_cp.shape[0]/16),1), (512,1),\
     (new_waveforms_total,full_delays,X_delays,out_tables_cp,celltypes,in_degs,edge_breakpoints,\
     net_delays_rise,net_delays_fall,XCond_pointers,pin_positions,delay_pointer_start,delay_pointer_end,waveform_pointers,output_TC,\
     nodes_list_cp.shape[0],args.num_splits,end_token,end_token_timestamp,simTime0Flag) )
    TC[nodes_list_cp] = cp.sum(output_TC, axis=1).astype(cp.uint32)
    waveform_pointers = cp.asarray(bg.ndata['waveform_start'][first_level_inputs])
    output_TC = output_TC + 2
    thisStageGatePointers=( cp.roll( cp.cumsum(output_TC.reshape(-1)), 1 ) ).reshape(-1, args.num_splits).astype(cp.int64)
    stage_length = int(thisStageGatePointers[0,0]) ; thisStageGatePointers[0,0] = 0
    bg.ndata['node_level'][nodes_list] = stage;
    thisStageGatePointers = thisStageGatePointers + append_length
    bg.ndata['waveform_start'][nodes_list] = th.cuda.LongTensor(thisStageGatePointers)
    append_length+=stage_length
    if ( append_length >= 6000000000) :
      print("max pointer value reached! exiting..." + str(append_length) )
      sys.exit()
    simulate_gate_GATSPI(  (math.ceil(nodes_list_cp.shape[0]/16),1), (512,1),\
     (new_waveforms_total,full_delays,X_delays,out_tables_cp,celltypes,in_degs,edge_breakpoints,\
     net_delays_rise,net_delays_fall,XCond_pointers,pin_positions,delay_pointer_start,delay_pointer_end,waveform_pointers,thisStageGatePointers,\
     nodes_list_cp.shape[0],args.num_splits,end_token,end_token_timestamp,simTime0Flag) )
    bg.ndata['waveform_end'][nodes_list] = th.cuda.LongTensor(thisStageGatePointers+1)
  update_saif()
  print('total events simulated: ' + str(append_length))
  if args.debugMode :
    debugStoreRuntime = timer()
    if subchunk_id == 0:
      subchunkStartTimes = [] ; subchunkEndTimes = []
    subchunkStartTimes.append(subchunkStart) ;  subchunkEndTimes.append(subchunkEnd)
    command = "saveSubchunkWaveforms_" + str(subchunk_id) + '=new_waveforms_total[0:append_length].get()' ; exec(command)
    command = 'saveSubchunkLength_' + str(subchunk_id) + '=append_length' ; exec(command)
    command = 'saveSubchunkPointersStart_' + str(subchunk_id) + "=bg.ndata['waveform_start'].to('cpu')" ; exec(command)
    command = 'saveSubchunkPointersEnd_' + str(subchunk_id) + "=bg.ndata['waveform_end'].to('cpu')" ; exec(command)
    deltaDebugStoreRuntime = timer() - debugStoreRuntime
    print('Save debug state took ' + str(deltaDebugStoreRuntime) + 'seconds.')
delta = timer() -start
print('re-sim time: ' + str(delta))
dump_saif()

if args.debugMode:
  print_string ='''Welcome to GL0AM's (really suss and poorly done :P ) debug environment!
For questions, email yanqingz@nvidia.com
Type a command and hit enter.
Valid commands and syntax are:

1) exit #exits the debug environment, and likely ends the GL0AM simulation program
2) reportDelays( string outputPinName ) #reports SDF delay values in local simulation timestep units for all arcs from a gate's input pins to a specified output pin.
#Ex: reportDelays('UI_0__add0_GL0AMs_U908/S')
3) reportWaveformInTimerange( string netName, int timeStart , int timeEnd ) # reports VCD style waveform text for a specified signal net within specified time window, time unit in  local simulation timestep.
# The time window specified must be within a simulated window. The waveform is for the net connection connected to the driving output pin, and doesn't include delays across the net to a load input pin.
#Ex: reportWaveformInTimerange("UI_219__add0_GL0AMs_n855", 0,10000)
4) reportSAIFInTimerange(string netName,  int timeStart, int timeEnd) # reports SAIF data for a specified net within specified time window, time unit in local simulation timestep.
#Ex: reportSAIFInTimerange("b[19272]",0,10000)
5) reportGateStateAtTimestamp(str outputPinName, int targetTimestamp) # reports all pin values for the cell specified by (one of) its output pin at a specified timestamp, time unit in  local simulation timestep.
#Ex: reportGateStateAtTimestamp("UI_0__add0_GL0AMs_U908/S",794625)


'''
  print(print_string)
  validCommands = ['reportDelays\(','reportWaveformInTimerange\(','reportSAIFInTimerange\(','reportGateStateAtTimestamp\(']
  command = 'dummy string'
  while command != 'exit':
    validInstruction=0
    command = input('GL0AM:')
    for thisCommand in validCommands:
      if bool(re.search(r'\A' + thisCommand, command)) :
        validInstruction=1; break;
    if validInstruction == 1:
      exec(command)
    elif command == 'exit':
      print('Thanks for using GL0AM debug environment, bye for now!')
    else:
      print('Not a valid command, try again!')

