##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/GL0AM/blob/main/LICENSE
#
##############################################################################

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
parser.add_argument('--graph4', type=str, help='the max FO constraint processed combo DGL graph')
args = parser.parse_args()

threads_per_block=512

start_compile=timer()
device = "cuda:0"
fileObject = open(args.graph4, 'rb') ; bg = pickle.load(fileObject)

topo_nodes_cpu =  dgl.traversal.topological_nodes_generator(bg)

fromEndpointsToSrc = dgl.topological_nodes_generator(bg, reverse=True)

howManyNodes = len(bg.nodes())
bg.ndata['howManyEndpoints'] = th.zeros(howManyNodes,device=device, dtype = th.int32 )
bg.ndata['endpointsStoragePointer'] = th.zeros(howManyNodes,device=device, dtype = th.int64)
dummyNodeNum = len(bg.nodes()) + 1000
listOfEndpointsStorage = cp.zeros( howManyNodes*1000, dtype = cp.int32)
endpointsStoragePointerOffsetTracker = 0

bg.ndata['howManyEndpoints'][fromEndpointsToSrc[0]] = 1
tempTensor = th.roll(th.cumsum(bg.ndata['howManyEndpoints'][fromEndpointsToSrc[0]], dtype=th.int64, dim = 0), 1, 0)
thisSize = tempTensor[0].repeat(1)
tempTensor[0] = 0
tempTensor += endpointsStoragePointerOffsetTracker
endpointsStoragePointerOffsetTracker += thisSize
bg.ndata['endpointsStoragePointer'][fromEndpointsToSrc[0]] = tempTensor.repeat(1)
listOfEndpointsStorage[0:fromEndpointsToSrc[0].size()[0]] = cp.asarray(fromEndpointsToSrc[0]).astype(cp.int32)

startTraverseCones = timer()
for level in range(1,len(fromEndpointsToSrc)):
  thisLevelOfNodes = fromEndpointsToSrc[level].to(device) # th.int64
  howManyFanoutsPerNode = cp.asarray(bg.out_degrees(thisLevelOfNodes)).astype(cp.int32)
  loadBreakpointsPerNode = cp.roll(cp.cumsum(howManyFanoutsPerNode, dtype=cp.int32, axis = 0), 1, 0)
  loadBreakpointsPerNode[0] = 0
  outputEdges = bg.out_edges(thisLevelOfNodes)[1]
  endpointsDrivenByEachLoad = cp.asarray(bg.ndata['howManyEndpoints'][outputEdges]).astype(cp.int32)
  thisLevelHowManyEndpoints = cp.zeros(thisLevelOfNodes.size()[0], dtype=cp.int32)
  calculateHowManyEndpointsDrivenPerNodeNonUniqueNonSorted( (math.ceil(thisLevelOfNodes.size()[0]/512),1), (512,1),\
   (howManyFanoutsPerNode,loadBreakpointsPerNode,endpointsDrivenByEachLoad,thisLevelHowManyEndpoints,thisLevelOfNodes.size()[0]) )
  tempDrivenEndpointsPointerPerNode = cp.roll(cp.cumsum(thisLevelHowManyEndpoints, dtype = cp.int64, axis=0), 1, 0)
  tempSize = tempDrivenEndpointsPointerPerNode[0].repeat(1)
  tempDrivenEndpointsPointerPerNode[0] = 0 ; offsets = cp.concatenate((tempDrivenEndpointsPointerPerNode,tempSize), axis=0).astype(cp.int32)
  tempEndpointsStorage = cp.full(tempSize[0].get(), dummyNodeNum, dtype=cp.int32)
  numDuplicatesPerNode = cp.zeros(thisLevelOfNodes.size()[0], dtype = cp.int32)
  endpointsDrivenByEachLoadPointer = cp.asarray(bg.ndata['endpointsStoragePointer'][outputEdges]).astype(cp.int64)
  storeTempEndpointsDrivenPerNode( (math.ceil(thisLevelOfNodes.size()[0]/512),1), (512,1),\
   (howManyFanoutsPerNode,loadBreakpointsPerNode,endpointsDrivenByEachLoad,thisLevelHowManyEndpoints,listOfEndpointsStorage,\
   tempDrivenEndpointsPointerPerNode,endpointsDrivenByEachLoadPointer,tempEndpointsStorage,thisLevelOfNodes.size()[0]) )
  tempEndpointsStorage = cupyx.segmented_sort(tempEndpointsStorage,offsets)
  calculateDuplicates( (math.ceil(thisLevelOfNodes.size()[0]/512),1), (512,1),\
   (thisLevelHowManyEndpoints,tempDrivenEndpointsPointerPerNode,tempEndpointsStorage,numDuplicatesPerNode,thisLevelOfNodes.size()[0]) )
  thisLevelHowManyEndpointsFinal = thisLevelHowManyEndpoints - numDuplicatesPerNode ;
  drivenEndpointsPointerPerNodeFinal = cp.roll(cp.cumsum(thisLevelHowManyEndpointsFinal, dtype = cp.int64, axis=0), 1, 0)
  sizeFinal = drivenEndpointsPointerPerNodeFinal[0].repeat(1)
  drivenEndpointsPointerPerNodeFinal[0] = 0
  drivenEndpointsPointerPerNodeFinal += endpointsStoragePointerOffsetTracker
  endpointsStoragePointerOffsetTracker += sizeFinal
  bg.ndata['howManyEndpoints'][thisLevelOfNodes] = th.cuda.IntTensor(thisLevelHowManyEndpointsFinal)
  bg.ndata['endpointsStoragePointer'][thisLevelOfNodes] = th.cuda.LongTensor(drivenEndpointsPointerPerNodeFinal)
  storeEndpointsDrivenPerNodeFinal( (math.ceil(thisLevelOfNodes.size()[0]/512),1), (512,1),\
   (thisLevelHowManyEndpoints,tempDrivenEndpointsPointerPerNode,tempEndpointsStorage,thisLevelHowManyEndpointsFinal,\
    drivenEndpointsPointerPerNodeFinal,listOfEndpointsStorage,thisLevelOfNodes.size()[0],dummyNodeNum) )
  print('Total storage addresses used: ' + str(endpointsStoragePointerOffsetTracker))
deltaTraverseCones = timer() - startTraverseCones
print("whole Traverse Cones process took " + str(deltaTraverseCones) + " seconds")

listOfEndpointsStorage = listOfEndpointsStorage[0:endpointsStoragePointerOffsetTracker]
del tempTensor,thisLevelOfNodes,outputEdges;
del howManyFanoutsPerNode,loadBreakpointsPerNode,endpointsDrivenByEachLoad,thisLevelHowManyEndpoints;
del tempDrivenEndpointsPointerPerNode,tempEndpointsStorage,numDuplicatesPerNode,endpointsDrivenByEachLoadPointer;
del thisLevelHowManyEndpointsFinal,drivenEndpointsPointerPerNodeFinal,offsets;
th.cuda.empty_cache()
mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()

print("traverse cones step checker:")
for targetFanouts in range(1,bg.ndata['howManyEndpoints'].max()+1):
  if  th.where(bg.ndata['howManyEndpoints'] == targetFanouts)[0].size()[0]:
    debugNode = th.where(bg.ndata['howManyEndpoints'] == targetFanouts)[0][0]  
    experimentFanouts = listOfEndpointsStorage[bg.ndata['endpointsStoragePointer'][debugNode]:bg.ndata['endpointsStoragePointer'][debugNode]+ targetFanouts]
    sub0, dummy =dgl.khop_out_subgraph(bg,debugNode,len(topo_nodes_cpu),relabel_nodes=True,store_ids=True,output_device=device)
    goldenFanouts = cp.asarray(th.sort(sub0.ndata['_ID'][th.where(sub0.out_degrees() == 0)[0]])[0]).astype(cp.int32)
    result = cp.all(goldenFanouts == experimentFanouts)
    print("targetFanout " + str(targetFanouts) + ' result is: ' + str(result))
    if result == False:
      print("traverse cones not correct for " + str(debugNode) + ' !') ; break
  else:
    continue

bg.ndata['clusterID'] = th.full((howManyNodes,),dummyNodeNum, device=device, dtype = th.int32 )
bg.ndata['clusterID'][fromEndpointsToSrc[0]] = th.cuda.IntTensor([x for x in range(fromEndpointsToSrc[0].size()[0])])
newClusterIDOffset = fromEndpointsToSrc[0].size()[0]

startBuildClusters = timer()
for level in range(1,len(fromEndpointsToSrc)):
  startThisLevel = timer()
  thisLevelOfNodes = fromEndpointsToSrc[level].to(device) # th.int64
  thisLevelOfNodes = thisLevelOfNodes[th.where(bg.ndata['clusterID'][thisLevelOfNodes] == dummyNodeNum)[0]] #don't need to figure out nodes that are already assigned a cluster
  howManyFanoutsPerNode = bg.out_degrees(thisLevelOfNodes).type(th.int32)
  loadBreakpointsPerNode = th.roll(th.cumsum(howManyFanoutsPerNode, dtype=th.int32, dim = 0), 1, 0)
  lastBreakpoint = loadBreakpointsPerNode[0].repeat(1)
  loadBreakpointsPerNode[0] = 0 ;  loadBreakpointsPerNode = th.cat((loadBreakpointsPerNode,lastBreakpoint), dim = 0)
  outputEdges = bg.out_edges(thisLevelOfNodes)[1]
  thisLevelOfNodesInterleaveRepeated = th.repeat_interleave(thisLevelOfNodes, howManyFanoutsPerNode)
  isThereMatchingCluster = (bg.ndata['howManyEndpoints'][thisLevelOfNodesInterleaveRepeated] == bg.ndata['howManyEndpoints'][outputEdges]).type(th.int8)
  indexesThatNeedFullCompare = th.where(isThereMatchingCluster)[0]
  if indexesThatNeedFullCompare.size()[0] > 0:
    sourcePointers = bg.ndata['endpointsStoragePointer'][thisLevelOfNodesInterleaveRepeated[indexesThatNeedFullCompare]]
    dstPointers = bg.ndata['endpointsStoragePointer'][outputEdges[indexesThatNeedFullCompare]]
    howManyEndpointsToCompare = bg.ndata['howManyEndpoints'][thisLevelOfNodesInterleaveRepeated[indexesThatNeedFullCompare]].type(th.int32)
    comparisonOffsets = th.roll(th.cumsum(howManyEndpointsToCompare, dtype=th.int32, dim = 0), 1, 0)
    lastOffset = comparisonOffsets[0].repeat(1)
    comparisonOffsets[0] = 0 ;  
    perEndpointComparisonResults = cp.asarray(th.zeros(lastOffset[0], dtype=th.int8, device=device))
    whichOffsetThisThreadBelongsTo = cp.asarray(th.repeat_interleave(comparisonOffsets, howManyEndpointsToCompare))
    sourcePointersRepeated = cp.asarray(th.repeat_interleave(sourcePointers, howManyEndpointsToCompare))
    dstPointersRepeated = cp.asarray(th.repeat_interleave(dstPointers, howManyEndpointsToCompare))
    segmentedCompare( (math.ceil(lastOffset[0]/512),1),(512,1),\
     (whichOffsetThisThreadBelongsTo,sourcePointersRepeated,dstPointersRepeated,listOfEndpointsStorage,\
     perEndpointComparisonResults,int(lastOffset)) )
    howManyEndpointsToCompareTemp = howManyEndpointsToCompare.repeat(1)
    howManyEndpointsToCompareTemp = (howManyEndpointsToCompareTemp/2).type(th.int32)
    howManyEndpointsToCompareTempMod = (howManyEndpointsToCompareTemp%2).type(th.int8)
    whichOffsetThisThreadBelongsTo = cp.asarray(th.repeat_interleave(comparisonOffsets, howManyEndpointsToCompareTemp))
    strideOffsets = th.roll(th.cumsum(howManyEndpointsToCompareTemp, dtype=th.int32, dim = 0), 1, 0)
    totalThreads = int(strideOffsets[0].repeat(1))
    strideOffsets[0] = 0 ;
    perThreadStrideOffsets = cp.asarray(th.repeat_interleave(strideOffsets, howManyEndpointsToCompareTemp))
    perThreadStride = cp.asarray(th.repeat_interleave(howManyEndpointsToCompareTemp, howManyEndpointsToCompareTemp))
    while totalThreads > 0:
      indexesThatAreOdd = th.where(howManyEndpointsToCompareTempMod)[0]
      perEndpointComparisonResults[comparisonOffsets[indexesThatAreOdd]] = cp.logical_and(perEndpointComparisonResults[comparisonOffsets[indexesThatAreOdd]],\
       perEndpointComparisonResults[comparisonOffsets[indexesThatAreOdd] + 2*howManyEndpointsToCompareTemp[indexesThatAreOdd]]).astype(cp.int8)
      segmentedCompareReduction( (math.ceil(totalThreads/512),1),(512,1),\
       (perThreadStride,perThreadStrideOffsets,whichOffsetThisThreadBelongsTo,perEndpointComparisonResults,totalThreads) )
      howManyEndpointsToCompareTemp = (howManyEndpointsToCompareTemp/2).type(th.int32)
      howManyEndpointsToCompareTempMod = howManyEndpointsToCompareTemp%2
      whichOffsetThisThreadBelongsTo = cp.asarray(th.repeat_interleave(comparisonOffsets, howManyEndpointsToCompareTemp))
      strideOffsets = th.roll(th.cumsum(howManyEndpointsToCompareTemp, dtype=th.int32, dim = 0), 1, 0)
      totalThreads = int(strideOffsets[0].repeat(1))
      strideOffsets[0] = 0 ;
      perThreadStrideOffsets = cp.asarray(th.repeat_interleave(strideOffsets, howManyEndpointsToCompareTemp))
      perThreadStride = cp.asarray(th.repeat_interleave(howManyEndpointsToCompareTemp, howManyEndpointsToCompareTemp))
  isThereMatchingCluster = cp.asarray(isThereMatchingCluster)
  if indexesThatNeedFullCompare.size()[0] > 0:
    isThereMatchingCluster[indexesThatNeedFullCompare] = perEndpointComparisonResults[comparisonOffsets]
  inheritClusterIDs = cp.asarray(bg.ndata['clusterID'][outputEdges])
  thisLevelClusterIDs = cp.full(thisLevelOfNodes.size()[0], dummyNodeNum, dtype=cp.int32)
  howManyFanoutsPerNode =cp.asarray(howManyFanoutsPerNode) ; loadBreakpointsPerNode =  cp.asarray(loadBreakpointsPerNode)
  assignMatchingClusterIDs( (math.ceil(thisLevelOfNodes.size()[0]/512),1),(512,1),\
   (isThereMatchingCluster,inheritClusterIDs,howManyFanoutsPerNode,loadBreakpointsPerNode,thisLevelClusterIDs,thisLevelOfNodes.size()[0]) )
  assignedClusterIDIndexes = cp.where(thisLevelClusterIDs!=dummyNodeNum)[0]
  if assignedClusterIDIndexes.shape[0] > 0 :
    assignedClusterIDIndexes = th.cuda.LongTensor(assignedClusterIDIndexes)
    bg.ndata['clusterID'][thisLevelOfNodes[assignedClusterIDIndexes]] = th.cuda.IntTensor(thisLevelClusterIDs[assignedClusterIDIndexes])
  #end matching
  unassignedClusterIDIndexes = th.cuda.LongTensor(th.where(th.cuda.IntTensor(thisLevelClusterIDs)==dummyNodeNum)[0])
  thisLevelOfNodes = thisLevelOfNodes[unassignedClusterIDIndexes]
  level0SubNodes = th.unique(bg.in_edges(bg.out_edges(thisLevelOfNodes)[1])[0])
  unassignedClusterIDIndexes = th.where(bg.ndata['clusterID'][level0SubNodes] == dummyNodeNum)[0]
  level0SubNodes=level0SubNodes[unassignedClusterIDIndexes]
  #search for same clusters across this list. create clusters for these
  #create clusters for the independent new clusters from the original nodes in level that are still cluster-less.
  #level0: calculate (howManyEndpoints,sumOfEndpoints) for level0SubNodes
  # unassignedClusterIDIndexes everything regarding cluster formation goes back to this indexing
  unassignedClusterIDIndexes = cp.asarray([x for x in range(level0SubNodes.size()[0])]).astype(cp.int64)
  endpointIDSumsAndHowManyEndpoints = cp.zeros(level0SubNodes.size()[0], dtype=cp.int64)
  level0EndpointListPointers = cp.asarray(bg.ndata['endpointsStoragePointer'][level0SubNodes])
  level0HowManyEndpoints = cp.asarray(bg.ndata['howManyEndpoints'][level0SubNodes])
  calculateEndpointsAndSumAttribute( (math.ceil(level0SubNodes.size()[0]/512),1),(512,1),\
   (level0EndpointListPointers,level0HowManyEndpoints,listOfEndpointsStorage,endpointIDSumsAndHowManyEndpoints,level0SubNodes.size()[0]) )
  uniqueLevel0Vals, uniqueLevel0Counts = cp.unique(endpointIDSumsAndHowManyEndpoints, return_counts =True)
  relevantLevel0CountIndexes = cp.where(uniqueLevel0Counts > 1)[0]
  relevantLevel0Values = uniqueLevel0Vals[relevantLevel0CountIndexes]
  horizontalClusters = []
  for level0Value in relevantLevel0Values:
    level1SubClusterIndexes = cp.where(endpointIDSumsAndHowManyEndpoints == level0Value)[0]
    # calculateAttributeLevel1  = range
    level1SubNodes = level0SubNodes[th.cuda.LongTensor(level1SubClusterIndexes)]
    endpointRanges = cp.zeros(level1SubNodes.size()[0], dtype=cp.int64)
    level0EndpointListPointers = cp.asarray(bg.ndata['endpointsStoragePointer'][level1SubNodes])
    level0HowManyEndpoints = cp.asarray(bg.ndata['howManyEndpoints'][level1SubNodes])
    calculateRanges( (math.ceil(level1SubNodes.size()[0]/512),1),(512,1),\
     (level0EndpointListPointers,level0HowManyEndpoints,listOfEndpointsStorage,endpointRanges,level1SubNodes.size()[0]) )
    uniqueLevel1Vals, uniqueLevel1Counts = cp.unique(endpointRanges, return_counts =True)
    relevantLevel1CountIndexes = cp.where(uniqueLevel1Counts > 1)[0]
    relevantLevel1Values = uniqueLevel1Vals[relevantLevel1CountIndexes]
    for level1Value in relevantLevel1Values:
      level2SubClusterIndexes = cp.where(endpointRanges == level1Value)[0]
      # calculateAttributeLevel2  = variance
      level2SubNodes = level1SubNodes[th.cuda.LongTensor(level2SubClusterIndexes)]
      endpointVariances = cp.zeros(level2SubNodes.size()[0], dtype=cp.float64)
      level0EndpointListPointers = cp.asarray(bg.ndata['endpointsStoragePointer'][level2SubNodes])
      level0HowManyEndpoints = cp.asarray(bg.ndata['howManyEndpoints'][level2SubNodes])
      endpointIDSumsAndHowManyEndpointsForVariance = endpointIDSumsAndHowManyEndpoints[th.cuda.LongTensor(level1SubClusterIndexes)][th.cuda.LongTensor(level2SubClusterIndexes)] #level0Results
      calculateVariances( (math.ceil(level2SubNodes.size()[0]/512),1),(512,1),\
       (level0EndpointListPointers,level0HowManyEndpoints,listOfEndpointsStorage,endpointIDSumsAndHowManyEndpointsForVariance,endpointVariances,level2SubNodes.size()[0]) )
      uniqueLevel2Vals, uniqueLevel2Counts = cp.unique(endpointVariances, return_counts =True)
      relevantLevel2CountIndexes = cp.where(uniqueLevel2Counts > 1)[0]
      relevantLevel2Values = uniqueLevel2Vals[relevantLevel2CountIndexes]
      for level2Value in relevantLevel2Values:
        level3SubClusterIndexes = cp.where(endpointVariances == level2Value)[0]
        # calculateAttributeLevel3  = median
        level3SubNodes = level2SubNodes[th.cuda.LongTensor(level3SubClusterIndexes)]
        endpointMedians = cp.zeros(level3SubNodes.size()[0], dtype=cp.int64)
        level0EndpointListPointers = cp.asarray(bg.ndata['endpointsStoragePointer'][level3SubNodes])
        level0HowManyEndpoints = cp.asarray(bg.ndata['howManyEndpoints'][level3SubNodes])
        calculateMedians( (math.ceil(level3SubNodes.size()[0]/512),1),(512,1),\
         (level0EndpointListPointers,level0HowManyEndpoints,listOfEndpointsStorage,endpointMedians,level3SubNodes.size()[0]) )
        uniqueLevel3Vals, uniqueLevel3Counts = cp.unique(endpointMedians, return_counts =True)
        relevantLevel3CountIndexes = cp.where(uniqueLevel3Counts > 1)[0]
        relevantLevel3Values = uniqueLevel3Vals[relevantLevel3CountIndexes]
        for level3Value in relevantLevel3Values:
          level4SubClusterIndexes = cp.where(endpointMedians == level3Value)[0]
          # calculateAttributeLevel4  = skewness
          level4SubNodes = level3SubNodes[th.cuda.LongTensor(level4SubClusterIndexes)]
          endpointSkews = cp.zeros(level4SubNodes.size()[0], dtype=cp.float64)
          level0EndpointListPointers = cp.asarray(bg.ndata['endpointsStoragePointer'][level4SubNodes])
          level0HowManyEndpoints = cp.asarray(bg.ndata['howManyEndpoints'][level4SubNodes])
          calculateSkewness( (math.ceil(level4SubNodes.size()[0]/512),1),(512,1),\
           (level0EndpointListPointers,level0HowManyEndpoints,listOfEndpointsStorage,endpointSkews,level4SubNodes.size()[0]) )
          uniqueLevel4Vals, uniqueLevel4Counts = cp.unique(endpointSkews, return_counts =True)
          relevantLevel4CountIndexes = cp.where(uniqueLevel4Counts > 1)[0]
          relevantLevel4Values = uniqueLevel4Vals[relevantLevel4CountIndexes]
          for level4Value in relevantLevel4Values:
            level5SubClusterIndexes = cp.where(endpointSkews == level4Value)[0]
            level5SubNodes = level4SubNodes[th.cuda.LongTensor(level5SubClusterIndexes)] ; masterNode = level5SubNodes[0] ;
            #full compare here, report assertion and exit if false!
            masterCompareNumEndpoints = int(bg.ndata['howManyEndpoints'][masterNode])
            masterCompareTensor = listOfEndpointsStorage[bg.ndata['endpointsStoragePointer'][masterNode]:bg.ndata['endpointsStoragePointer'][masterNode] + 	masterCompareNumEndpoints].repeat(1)
            comparisonPointers = cp.asarray(bg.ndata['endpointsStoragePointer'][level5SubNodes[1:]])
            numCompareElements = (level5SubNodes.size()[0]-1) * masterCompareNumEndpoints
            fullComparisonResults= cp.zeros(numCompareElements, dtype=cp.int8)
            fullComparisonAgainstMasterTensor( (math.ceil(numCompareElements/512),1),(512,1),\
             (masterCompareTensor,comparisonPointers,listOfEndpointsStorage,fullComparisonResults,masterCompareNumEndpoints,numCompareElements) )
            assert cp.all(fullComparisonResults), "The necessary but not sufficient conditions to find same level gate clusters is wrong!"
            horizontalClusterIndexes =unassignedClusterIDIndexes[level1SubClusterIndexes[level2SubClusterIndexes[level3SubClusterIndexes[level4SubClusterIndexes[level5SubClusterIndexes]]]]]
            horizontalClusters.append(horizontalClusterIndexes)
  horizontalClusterIDs = [(x+newClusterIDOffset) for x in range(len(horizontalClusters))]
  howManyHorizontalClusters = len(horizontalClusterIDs)
  horizontalClusterLengths = [len(x) for x in horizontalClusters]
  horizontalClusterIDsRepeated = cp.asarray(th.repeat_interleave(th.cuda.IntTensor(horizontalClusterIDs),th.cuda.IntTensor(horizontalClusterLengths) ))
  if len(horizontalClusters) > 0:
    horizontalClusters = cp.concatenate(horizontalClusters)
    horizontalLevelClusterIDs = th.full((level0SubNodes.size()[0],), dummyNodeNum, dtype=th.int32, device=device)
    horizontalLevelClusterIDs = horizontalLevelClusterIDs.scatter_(0,th.cuda.LongTensor(horizontalClusters),th.cuda.IntTensor(horizontalClusterIDsRepeated)) # check scatter operation is persistent
    newClusterIDOffset += howManyHorizontalClusters
    relevantIndexes = th.where( horizontalLevelClusterIDs != dummyNodeNum)[0]
    bg.ndata['clusterID'][level0SubNodes[relevantIndexes]] = horizontalLevelClusterIDs[relevantIndexes] # check if persistent
  remainingIndexes = th.where(bg.ndata['clusterID'][thisLevelOfNodes] == dummyNodeNum)[0]
  remainingNodes = thisLevelOfNodes[remainingIndexes]
  newClusterIDsNeeded = remainingNodes.shape[0]
  bg.ndata['clusterID'][remainingNodes] = th.cuda.IntTensor([x for x in range(newClusterIDsNeeded)]) + newClusterIDOffset
  newClusterIDOffset +=newClusterIDsNeeded
  deltaThisLevel = timer() - startThisLevel;
  print("Time for level " + str(level) + ' is ' + str(deltaThisLevel) + ' seconds. ' + str(newClusterIDOffset) + ' clusterIDs needed so far for ' + str(len(bg.nodes())) + ' gates.')
deltaBuildClusters = timer() - startBuildClusters
print("whole Build Clusters process took " + str(deltaBuildClusters) + " seconds")


values, counts = th.unique(bg.ndata['clusterID'], return_counts=True) ; outerLoopBreaker=0
unique_counts = th.unique(counts)
for thisCount in unique_counts:
  debugCluster = th.where(bg.ndata['clusterID'] ==  values[th.where(counts==thisCount)[0]][0])[0]
  goldenNode = debugCluster[0];
  goldenFanouts = listOfEndpointsStorage[bg.ndata['endpointsStoragePointer'][goldenNode]:bg.ndata['endpointsStoragePointer'][goldenNode]+\
   bg.ndata['howManyEndpoints'][goldenNode].type(th.int64)]
  for thisNode in debugCluster[1:]:
    compareFanouts = listOfEndpointsStorage[bg.ndata['endpointsStoragePointer'][thisNode]:bg.ndata['endpointsStoragePointer'][thisNode]+\
   bg.ndata['howManyEndpoints'][thisNode].type(th.int64)]
    if cp.all(goldenFanouts == compareFanouts) != True:
      print("error on " + str(goldenNode) + 'and ' + str(thisNode) + '! Exiting...') ; outerLoopBreaker=1;  break
  if outerLoopBreaker:
    break
  print("Cluster size of " + str(int(thisCount)) + ' seems correct.')

numNodesInHypergraph = fromEndpointsToSrc[0].size()[0]
numEdgesInHypergraph = int(bg.ndata['clusterID'].max() - numNodesInHypergraph + 1)

endpointNodes = cp.asarray(fromEndpointsToSrc[0]).astype(cp.int32)
numHypergraphNodes = endpointNodes.shape[0]
hypergraphVertexID = cp.full( int(endpointNodes.max()+1), dummyNodeNum, dtype = cp.int32)
setIndex_T( (math.ceil(endpointNodes.shape[0]/512),1), (512,1),\
 (endpointNodes,hypergraphVertexID,endpointNodes.shape[0]) )

uniqueClusterIDPointers = cp.zeros( int(bg.ndata['clusterID'].max()+1), dtype=cp.int64)
uniqueClusterIDHowManyEndpoints = cp.zeros(  int(bg.ndata['clusterID'].max()+1), dtype=cp.int32)
writeUniqueClusterIDAttributes( (math.ceil(len(bg.nodes())/512),1),(512,1),\
 (cp.asarray(bg.ndata['clusterID']),cp.asarray(bg.ndata['howManyEndpoints']),cp.asarray(bg.ndata['endpointsStoragePointer']),\
  uniqueClusterIDPointers,uniqueClusterIDHowManyEndpoints,len(bg.nodes())) )

numClusters = int(bg.ndata['clusterID'].max()+1)
uniqueClusterIDHowManyEndpointsOffset = cp.roll(cp.cumsum(uniqueClusterIDHowManyEndpoints, dtype=cp.int32, axis = 0), 1, 0)
uniqueClusters1DSize = int(uniqueClusterIDHowManyEndpointsOffset[0].repeat(1)) ; uniqueClusterIDHowManyEndpointsOffset[0] = 0
uniqueClusterIDHowManyEndpointsOffsetRepeated = cp.asarray(th.repeat_interleave(th.cuda.IntTensor(uniqueClusterIDHowManyEndpointsOffset),th.cuda.IntTensor(uniqueClusterIDHowManyEndpoints)))
uniqueClusterIDPointersRepeated = cp.asarray(th.repeat_interleave(th.cuda.LongTensor(uniqueClusterIDPointers),th.cuda.IntTensor(uniqueClusterIDHowManyEndpoints)))
uniqueClusters1D = cp.zeros(uniqueClusters1DSize,dtype=cp.int32)
storeUniqueClusters1D( (math.ceil(uniqueClusters1DSize/512),1),(512,1),\
 (listOfEndpointsStorage,uniqueClusterIDPointersRepeated,uniqueClusterIDHowManyEndpointsOffsetRepeated,uniqueClusters1D,uniqueClusters1DSize) )
th.cuda.empty_cache()
mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()

clusterIDs, clusterWeights = th.unique(bg.ndata['clusterID'],return_counts=True)
clusterWeightsSplit = clusterWeights.type(th.float32) / th.cuda.IntTensor(uniqueClusterIDHowManyEndpoints).type(th.float32)

scatterAddValues = cp.asarray(th.repeat_interleave(clusterWeightsSplit, th.cuda.IntTensor(uniqueClusterIDHowManyEndpoints)))
hypergraphNodeWeights = cp.zeros(numHypergraphNodes, dtype=cp.float32)
scatterAddSlices = cp.zeros(scatterAddValues.shape[0], dtype=cp.int32)
translateNodeIDToHypergraphNodeID( (math.ceil(scatterAddValues.shape[0]/512),1),(512,1),\
 (hypergraphVertexID,uniqueClusters1D,scatterAddSlices,scatterAddValues.shape[0]) )
cupyx.scatter_add(hypergraphNodeWeights,scatterAddSlices,scatterAddValues)
hypergraphNodeWeights = hypergraphNodeWeights.astype(cp.int32)

numHypergraphNodes = numNodesInHypergraph
numEdges = numClusters - numHypergraphNodes

endpointIDSumsAndHowManyEndpoints = cp.zeros(numClusters,dtype=cp.int64)
uniqueClusterIDHowManyEndpointsOffset = uniqueClusterIDHowManyEndpointsOffset.astype(cp.int64)
calculateEndpointsAndSumAttribute( (math.ceil(numClusters/512),1),(512,1),\
 (uniqueClusterIDHowManyEndpointsOffset,uniqueClusterIDHowManyEndpoints,uniqueClusters1D,endpointIDSumsAndHowManyEndpoints,numClusters) )

start=timer()
endpointIDSumsAndHowManyEndpoints = cp.zeros(numClusters,dtype=cp.int64)
uniqueClusterIDHowManyEndpointsOffset = uniqueClusterIDHowManyEndpointsOffset.astype(cp.int64)
calculateEndpointsAndSumAttribute( (math.ceil(numClusters/512),1),(512,1),\
 (uniqueClusterIDHowManyEndpointsOffset,uniqueClusterIDHowManyEndpoints,uniqueClusters1D,endpointIDSumsAndHowManyEndpoints,numClusters) )
uniqueLevel0Vals, uniqueLevel0Counts = cp.unique(endpointIDSumsAndHowManyEndpoints, return_counts =True)
relevantLevel0CountIndexes = cp.where(uniqueLevel0Counts > 1)[0]
relevantLevel0Values = uniqueLevel0Vals[relevantLevel0CountIndexes]
clusterOfClusters = []
for level0Value in relevantLevel0Values:
  level1SubClusterIndexes = cp.where(endpointIDSumsAndHowManyEndpoints == level0Value)[0]
  # calculateAttributeLevel1  = range
  endpointRanges = cp.zeros(level1SubClusterIndexes.shape[0], dtype=cp.int64)
  level0EndpointListPointers = uniqueClusterIDHowManyEndpointsOffset[level1SubClusterIndexes]
  level0HowManyEndpoints = uniqueClusterIDHowManyEndpoints[level1SubClusterIndexes]
  calculateRanges( (math.ceil(level1SubClusterIndexes.shape[0]/512),1),(512,1),\
   (level0EndpointListPointers,level0HowManyEndpoints,uniqueClusters1D,endpointRanges,level1SubClusterIndexes.shape[0]) )
  uniqueLevel1Vals, uniqueLevel1Counts = cp.unique(endpointRanges, return_counts =True)
  relevantLevel1CountIndexes = cp.where(uniqueLevel1Counts > 1)[0]
  relevantLevel1Values = uniqueLevel1Vals[relevantLevel1CountIndexes]
  #delta0 = timer();
  #print('level1 : ' + str(delta0-start0))
  for level1Value in relevantLevel1Values:
    #start2 = timer()
    level2SubClusterIndexes = cp.where(endpointRanges == level1Value)[0]
    # calculateAttributeLevel2  = variance
    level2SubClusters = level1SubClusterIndexes[level2SubClusterIndexes]
    endpointVariances = cp.zeros(level2SubClusters.shape[0], dtype=cp.float64)
    level0EndpointListPointers = uniqueClusterIDHowManyEndpointsOffset[level2SubClusters]
    level0HowManyEndpoints = uniqueClusterIDHowManyEndpoints[level2SubClusters]
    endpointIDSumsAndHowManyEndpointsForVariance = endpointIDSumsAndHowManyEndpoints[level1SubClusterIndexes[level2SubClusterIndexes]]
    calculateVariances( (math.ceil(level2SubClusters.shape[0]/512),1),(512,1),\
     (level0EndpointListPointers,level0HowManyEndpoints,uniqueClusters1D,endpointIDSumsAndHowManyEndpointsForVariance,endpointVariances,level2SubClusters.shape[0]) )
    uniqueLevel2Vals, uniqueLevel2Counts = cp.unique(endpointVariances, return_counts =True)
    relevantLevel2CountIndexes = cp.where(uniqueLevel2Counts > 1)[0]
    relevantLevel2Values = uniqueLevel2Vals[relevantLevel2CountIndexes]
    #delta2 = timer();
    #print('level2 : ' + str(delta2-start2))
    for level2Value in relevantLevel2Values:
      #start3 = timer()
      level3SubClusterIndexes = cp.where(endpointVariances == level2Value)[0]
      # calculateAttributeLevel3  = median
      level3SubClusters = level2SubClusters[level3SubClusterIndexes]
      endpointMedians = cp.zeros(level3SubClusters.shape[0], dtype=cp.int64)
      level0EndpointListPointers = uniqueClusterIDHowManyEndpointsOffset[level3SubClusters]
      level0HowManyEndpoints = uniqueClusterIDHowManyEndpoints[level3SubClusters]
      calculateMedians( (math.ceil(level3SubClusters.shape[0]/512),1),(512,1),\
       (level0EndpointListPointers,level0HowManyEndpoints,uniqueClusters1D,endpointMedians,level3SubClusters.shape[0]) )
      uniqueLevel3Vals, uniqueLevel3Counts = cp.unique(endpointMedians, return_counts =True)
      relevantLevel3CountIndexes = cp.where(uniqueLevel3Counts > 1)[0]
      relevantLevel3Values = uniqueLevel3Vals[relevantLevel3CountIndexes]
      #delta3 = timer();
      #print('level3 : ' + str(delta3-start3))
      for level3Value in relevantLevel3Values:
        #start4 = timer()
        level4SubClusterIndexes = cp.where(endpointMedians == level3Value)[0]
        # calculateAttributeLevel4  = skewness
        level4SubClusters = level3SubClusters[level4SubClusterIndexes]
        endpointSkews = cp.zeros(level4SubClusters.shape[0], dtype=cp.float64)
        level0EndpointListPointers = uniqueClusterIDHowManyEndpointsOffset[level4SubClusters]
        level0HowManyEndpoints = uniqueClusterIDHowManyEndpoints[level4SubClusters]
        calculateSkewness( (math.ceil(level4SubClusters.shape[0]/512),1),(512,1),\
         (level0EndpointListPointers,level0HowManyEndpoints,uniqueClusters1D,endpointSkews,level4SubClusters.shape[0]) )
        uniqueLevel4Vals, uniqueLevel4Counts = cp.unique(endpointSkews, return_counts =True)
        relevantLevel4CountIndexes = cp.where(uniqueLevel4Counts > 1)[0]
        relevantLevel4Values = uniqueLevel4Vals[relevantLevel4CountIndexes]
        #delta4 = timer();
        #print('level4 : ' + str(delta4-start4))
        for level4Value in relevantLevel4Values:
          #start5 = timer()
          level5SubClusterIndexes = cp.where(endpointSkews == level4Value)[0]
          level5SubClusters = level4SubClusters[level5SubClusterIndexes] ; masterNode = level5SubClusters[0] ;
          #full compare here, report assertion and exit if false!
          masterCompareNumEndpoints = int(uniqueClusterIDHowManyEndpoints[masterNode])
          masterCompareTensor = uniqueClusters1D[uniqueClusterIDHowManyEndpointsOffset[masterNode]:uniqueClusterIDHowManyEndpointsOffset[masterNode] + 	masterCompareNumEndpoints].repeat(1)
          comparisonPointers = uniqueClusterIDHowManyEndpointsOffset[level5SubClusters[1:]]
          numCompareElements = (level5SubClusters.shape[0]-1) * masterCompareNumEndpoints
          fullComparisonResults= cp.zeros(numCompareElements, dtype=cp.int8)
          fullComparisonAgainstMasterTensor( (math.ceil(numCompareElements/512),1),(512,1),\
           (masterCompareTensor,comparisonPointers,uniqueClusters1D,fullComparisonResults,masterCompareNumEndpoints,numCompareElements) )
          assert cp.all(fullComparisonResults), "The necessary but not sufficient conditions to find same cluster of clusters is wrong!"
          sameClusterIndexes =level1SubClusterIndexes[level2SubClusterIndexes[level3SubClusterIndexes[level4SubClusterIndexes[level5SubClusterIndexes]]]]
          clusterOfClusters.append(sameClusterIndexes)
          #delta5 = timer();
          #print('level5 : ' + str(delta5-start5))
mergeClusters =[] ; scatterClusters = [] ; scatterLengths = []
for clusterOfCluster in clusterOfClusters:
  scatterLengths.append(len(clusterOfCluster) - 1)
  mergeClusters.append(clusterOfCluster[0])
  scatterClusters.append(clusterOfCluster[1:])
scatterLengths = cp.asarray(scatterLengths, dtype=cp.int32)
mergeClusters = cp.asarray(mergeClusters, dtype=cp.int32)
scatterClusters = cp.concatenate(scatterClusters, dtype=cp.int32)
clustersToDelete = len(scatterClusters)
skipTheseHyperEdges = cp.ones(numClusters, dtype=cp.int8)
skipTheseHyperEdges[scatterClusters] = 0
scatterAddSlices2 = cp.asarray(th.repeat_interleave(th.cuda.IntTensor(mergeClusters), th.cuda.IntTensor(scatterLengths)))
clusterWeights = cp.asarray(clusterWeights).astype(cp.int32)
scatterAddValues2 = clusterWeights[scatterClusters]
cupyx.scatter_add(clusterWeights,scatterAddSlices2,scatterAddValues2)
delta = timer() - start;
print('final cluster of clusters clustering took: ' + str(delta) + ' seconds')

np.set_printoptions(threshold=sys.maxsize)

#translate from DGL node IDS to hMETIS nodeIDs here
#use hypergraphVertexID
uniqueClusters1DHMetisVersion = cp.zeros(uniqueClusters1DSize,dtype=cp.int32)
translateDGLNodesToHMetisNodes( (math.ceil(uniqueClusters1DSize/512),1),(512,1),\
 (hypergraphVertexID,uniqueClusters1D,uniqueClusters1DHMetisVersion,uniqueClusters1DSize) )

fileDumpStart = timer()
print_string =''
print_string += str(numEdges-clustersToDelete) + ' ' + str(numHypergraphNodes) + ' 11\n'
for i in range(numHypergraphNodes,numClusters):
  if skipTheseHyperEdges[i]:
    edgeWeight = math.ceil(clusterWeightsSplit[i]) ; maxEdgeOffset = min(uniqueClusterIDHowManyEndpoints[i],1000)
    edgeDefinition = uniqueClusters1DHMetisVersion[uniqueClusterIDHowManyEndpointsOffset[i]:uniqueClusterIDHowManyEndpointsOffset[i]+maxEdgeOffset] + 1
    edgeDefinition = str(edgeDefinition).replace('\n','')
    edgeDefinition = re.sub(r'\[\s*','',edgeDefinition)
    edgeDefinition = re.sub(r'\s*\]','',edgeDefinition)
    edgeDefinition = re.sub(r'\s+',' ',edgeDefinition)
    print_string += str(int(edgeWeight)) + ' ' + edgeDefinition + '\n'
for i in range(numHypergraphNodes-1):
  nodeWeight = hypergraphNodeWeights[i]; 
  print_string += str(nodeWeight) + '\n'
nodeWeight = hypergraphNodeWeights[i+1]; 
print_string += str(nodeWeight)
fileDumpDelta = timer() - fileDumpStart
print("file dump processing took: " + str(fileDumpDelta)  + ' seconds')

f = open(args.block + ".hgr",'w')
f.write(print_string)
f.close()


 
