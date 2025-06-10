import gatspi4mlcad2025_contest_sim.gatspi4mlcad2025_contest_sim as gatspi4mlcad2025_contest_sim
import argparse

def main():
 parser = argparse.ArgumentParser()
 parser.add_argument('--top_name', type=str, help = 'top module name')
 parser.add_argument('--graph0FilePath', type=str, help='raw csr graph file path, or stored DGL graph file path for golden netlist')
 parser.add_argument('--graph1FilePath', type=str, help='raw csr graph file path, or stored DGL graph file path for resynth netlist')
 parser.add_argument('--dumpDGLGraph', type=bool, default=False, help='dump the created DGL graph or not. \
 Should be true and ran once when creating the DGL graph from raw CSR graph, from then on can be set to False to simply load the DGL graph')
 parser.add_argument('--createStdCellLibLUT', type=bool, default=False, help='compile the std cell library truth tables or not. should be run once for each new technology')
 parser.add_argument('--cycles', type=int, default=50000, help='target verification cycles to run')
 parser.add_argument('--parallel_sim_cycles', type=int, default=32, choices=[1,2,4,8,16,32,64,128,256], help='# of cycles to be simulated in parallel on GPU')
 parser.add_argument('--queryNetsListFile', type=str, default = '', help='path to file which houses a list of nets to compare. Expected file format is one net per line in file.')
 args = parser.parse_args()
 #args = parser.parse_args(['--top_name', 'adder', '--graph0FilePath', './adder.pkl', '--graph1FilePath', \
 #'./adder_altIncorrect.pkl', '--dumpDGLGraph', '1', '--queryNetsListFile', 'queryNets.lst'])
 
 gatspiSim0 = gatspi4mlcad2025_contest_sim.gatspiSimulateAndCompareTool(topName = args.top_name, \
  graph0FilePath = args.graph0FilePath, graph1FilePath = args.graph1FilePath, dumpDGLGraph = args.dumpDGLGraph, \
  createStdCellLibLUT = args.createStdCellLibLUT, cycles = args.cycles, PARALLEL_CYCLES = args.parallel_sim_cycles, \
  queryNetsListFile = args.queryNetsListFile )
 
 gatspiSim0.doEverything()
 '''for c in range(args.parallel_sim_cycles):
  A=[] ; B=[]; C=[] ; printA='' ; printB='' ;  printC='' ; 
  for i in range(31,-1,-1):
   aName = 'a' + '[' + str(i) + ']' ; bName = 'b' + '[' + str(i) + ']' ; cName = 'c' + '[' + str(i) + ']' ; 
   bitIDa = gatspiSim0.net2id1[aName] ;  bitIDb = gatspiSim0.net2id1[bName] ;bitIDc = gatspiSim0.net2id1[cName] ;
   A.append(str(int(gatspiSim0.currentLogicValue[bitIDa,c]))) ; B.append(str(int(gatspiSim0.currentLogicValue[bitIDb,c]))) ; C.append(str(int(gatspiSim0.currentLogicValue[bitIDc,c]))) ; 
  A = "".join(A) ; B = "".join(B) ; C = "".join(C) ; 
  printA += 'a' + '[' + str(31) + ':' + str(0) + ']' + " : " + str(hex(int(A, base=2)))
  printB += 'b' + '[' + str(31) + ':' + str(0) + ']' + " : " + str(hex(int(B, base=2)))
  printC += 'c' + '[' + str(31) + ':' + str(0) + ']' + " : " + str(hex(int(C, base=2)))
  print(printA + ' ' + printB + ' : ' + printC)'''

if __name__=="__main__":
 main()
