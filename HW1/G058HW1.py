from pyspark import SparkContext, SparkConf
import sys
import numpy as np
import os, time
from collections import defaultdict
import random as rand


def raw_to_edges(raw_str):
    t = []
    for e in raw_str.split(','):
        t.append(int(e))
    return [tuple(t)]


def CountTriangles_color(edges):
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph.
    # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
    for u in neighbors:
        # Iterate over each pair of neighbors of u
        for v in neighbors[u]:
            if v > u:
                for w in neighbors[v]:
                    # If w is also a neighbor of u, then we have a triangle
                    if w > v and w in neighbors[u]:
                        triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count


def CountTriangles_sparkpartitions(pairs):
    count_dict = {}
    for pair in pairs:
        k, edge = pair[0], pair[1]
        count_dict[k] = CountTriangles_color(edge)
    return [(k,count_dict[k]) for k in count_dict.keys()]


def MR_ApproxTCwithNodeColors(edges, C):
    P = 8191
    a, b = rand.randint(1,P-1), rand.randint(0,P-1)
    def hash_func(u_v): 
        u, v = u_v[0], u_v[1]
        h_u, h_v =  ((a*u + b) % P) % C , ((a*v + b) % P) % C
        if h_u == h_v:
            return [(h_u, (u, v))]
        return []
        
    triangles_count = (edges.flatMap(hash_func) # R1 (Map Phase)
                        .groupByKey()             # R1 (Shuffling)
                        .mapValues(CountTriangles_color)# R1 (Reduce Phase)
                        .map(lambda x: (0, x[1]))# R2 (Map Phase)
                        .reduceByKey(lambda x,y: x + y) # R2 (Reduce Phase)
                        )
    return C**2 * triangles_count.collect()[0][1]


def MR_ApproxTCwithSparkPartitions(edges, C):
    triangles_count = (edges.flatMap(lambda x: [(rand.randint(0, C-1), x)]) # R1 (Map Phase)
                        .groupByKey() # R1 (Shuffling)
                        .mapPartitions(CountTriangles_sparkpartitions) # R1 (Reduce Phase)
                        .values() # R2 (Reduce Phase)
                        .sum() # R2 (Reduce Phase)
                       )
    return C**2 * triangles_count


def main():

    assert len(sys.argv) == 4, "Usage: python GxxxHW1.py <C> <R> <input_file_name>"
        
    C = sys.argv[1]
    R = sys.argv[2]
    assert C.isdigit(), "C must be an integer"
    assert R.isdigit(), "R must be an integer"
    C, R = int(C), int(R)

    data_path = sys.argv[3]
    assert os.path.isfile(data_path), "File or folder not found"
    
    conf = SparkConf().setAppName('TriangleCounting')
    sc = SparkContext(conf=conf)
    rawData = sc.textFile(data_path,minPartitions=C).cache()
    
    edges = rawData.flatMap(raw_to_edges)
    edges = edges.repartition(numPartitions= C)
    N = edges.count()

    # MR_ApproxTCwithNodeColors
    N_NodeColor_list = []
    Time_NodeColor_list = []
    for _ in range(R):
        start = time.time()
        N_Triangles_NodeColor = MR_ApproxTCwithNodeColors(edges, C)
        NodeColor_RunningTime = (time.time() - start) * 1000
        N_NodeColor_list.append(N_Triangles_NodeColor)
        Time_NodeColor_list.append(NodeColor_RunningTime)

    Median_N_Triangles = np.median(np.array(N_NodeColor_list)) 
    Average_Time_NodeColor = np.array(Time_NodeColor_list).mean() 
    
    MR_ApproxTCwithSparkPartitions
    start = time.time()
    N_Triangles_SparkPartitions = MR_ApproxTCwithSparkPartitions(edges, C)
    SparkPartitions_RunningTime = (time.time() - start) * 1000

    print(f"Dataset = {data_path}")
    print(f"Number of Edges = {N}")
    print(f"Number of Colors = {C}")
    print(f"Number of Repetitions = {R}")
    print("Approximation through node coloring")
    print(f"- Number of triangles (median over {R} runs) = {int(Median_N_Triangles)}")
    print(f"- Running time (average over {R} runs) = {int(Average_Time_NodeColor)} ms")
    print("Approximation through Spark partitions")
    print(f"- Number of triangles = {int(N_Triangles_SparkPartitions)}")
    print(f"- Running time = {int(SparkPartitions_RunningTime)} ms")

if __name__ == '__main__':
    main()
    
