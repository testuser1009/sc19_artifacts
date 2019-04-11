# sc19_artifacts

The data folder contains a csv file with all collected data across all 5 application, 3 power caps, 5 bandwidth levels, 3 task placement algorithms, and 3 number of threads.

The graph-analysis directory contains the python scripts we created
for graph spectral analysis. The main files to look into will be
GraphProcess.py and GraphConstruct.py. These two files in conjunction
constructs the graphs, uses the Gower.py file for gower distance while
building edges, and analyzes the data.
