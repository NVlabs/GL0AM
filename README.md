# GL0AM

## Introduction


GL0AM is a delay annotated, GPU accelerated gate-level logic simulator. It simulates in 4 value, and covers most simulation scenarios and SDF delay annotation involving single frequency clock stimuli and designs. At the end of simulation, a simple debug environment is also included. It operates by first performing a GPU accelerated 0-delay simulation on the design-under-test (DUT) from primary port inputs in order to acquire register/SRAM/clock gate output waveforms, and then does a GPU accelerated parallel re-simulation to acquire the remaining combinational gate waveforms. To minimize the issues of memory locality and synchronization overheads that often plague GPU accelerated logic simulation, we use a partitioning strategy to break top level netlists into smaller independent groups of logic cones that map to a single GPU thread block--thus using only the fastest block synchronization.

![GL0AM Partitioning Strategy](https://github.com/NVlabs/GL0AM/blob/main/images/GL0AM_ToolFlow4.jpg?raw=true)

1. intro, paper link, flow chart
2. prerequisites, packages, dataset
3. tutorial/trial run
4. results, future work.
