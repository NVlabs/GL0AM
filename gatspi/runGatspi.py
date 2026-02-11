import gatspiLib.gatspiLib as gatspiLib
import argparse

def main():
 parser = argparse.ArgumentParser()
 parser.add_argument('--topName', type=str, help = 'top module name')
 parser.add_argument('--instanceName', type=str, default = 'uut', help = 'instance name in SAIF header')
 parser.add_argument('--testname', type=str, help = 'test name, used in SAIF header')
 parser.add_argument('--graphFilePath', type=str, help='raw csr graph file path, or stored DGL graph file path')
 parser.add_argument('--inputTraceFile', type = str, default=None, help = 'path to the input trace file. None value will generate random source waveforms')
 parser.add_argument('--duration', type = int, help = 'Duration of the test, in ps')
 parser.add_argument('--period', type =int, help = 'clock period, in ps')
 parser.add_argument('--numOfSubchunks', type = int, help = 'number of sub chunk divisions needed to get through the whole test, increase for longer tests or GPU buffer memory will overflow')
 parser.add_argument('--waveformBufferSize', type = int, default=6000000000, help = 'Size, in units of 32bit words, of the waveform buffer. Can be larger for larger memory GPUs. Usually use 6000000000 for 24GB buffer')
 parser.add_argument('--dumpDGLGraph', type=bool, default=False, help='dump the created DGL graph or not. \
 Should be true and ran once when creating the DGL graph from raw CSR graph, from then on can be set to False to simply load the DGL graph')
 parser.add_argument('--createStdCellLibLUT', type=bool, default=False, help='compile the std cell library truth tables or not. should be run once for each new technology')

 args = parser.parse_args()
 #args = args = parser.parse_args(['--topName', 'qadd_pipe', '--testname', 'testFlow', '--graphFilePath', \
 #'../test.pkl', '--inputTraceFile', '../qadd_pipe.waveforms_part0', '--duration', '6000000', \
 #'--period', '1000', '--numOfSubChunks', '1', '--dumpDGLGraph', '1','--createStdCellLibLUT', '1'])
 
 gatspiSim0 = gatspiLib.GATSPI(topName = args.topName, instanceName = args.instanceName, \
  graphFilePath = args.graphFilePath, testname = args.testname, inputTraceFile = args.inputTraceFile, \
  testDuration =args.duration, period = args.period, numOfSubchunks = args.numOfSubchunks, \
  waveformBufferSize = args.waveformBufferSize, dumpDGLGraph = args.dumpDGLGraph, createStdCellLibLUT = args.createStdCellLibLUT )
 
 gatspiSim0.doEverything()

if __name__=="__main__":
 main()

