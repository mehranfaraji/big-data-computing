from pyspark import SparkContext, SparkConf
import sys
import os 
import random
import pyspark
from collections import defaultdict
import time
import numpy as np

    
def raw_to_edges(rdd):
    edge = []
    for e in rdd.split(','):
        edge.append(int(e))
    return [tuple(edge)]


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


def countTriangles2(colors_tuple, edges, rand_a, rand_b, p, num_colors):

    #We assume colors_tuple to be already sorted by increasing colors. Just transform in a list for simplicity
    colors = list(colors_tuple)  
    #Create a dictionary for adjacency list
    neighbors = defaultdict(set)
    #Creare a dictionary for storing node colors
    node_colors = dict()
    for edge in edges:

        u, v = edge
        node_colors[u]= ((rand_a*u+rand_b)%p)%num_colors
        node_colors[v]= ((rand_a*v+rand_b)%p)%num_colors
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph
    for v in neighbors:
        # Iterate over each pair of neighbors of v
        for u in neighbors[v]:
            if u > v:
                for w in neighbors[u]:
                    # If w is also a neighbor of v, then we have a triangle
                    if w > u and w in neighbors[v]:
                        # Sort colors by increasing values
                        triangle_colors = sorted((node_colors[u], node_colors[v], node_colors[w]))
                        # If triangle has the right colors, count it.
                        if colors==triangle_colors:
                            triangle_count += 1
    # Return the total number of triangles in the graph
    return [triangle_count]


def MR_ApproxTCwithNodeColors(edges, C):
    P = 8191
    a, b = random.randint(1,P-1), random.randint(0,P-1)
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


def MR_ExactTC(edges_rdd: pyspark.RDD, C):
    p = 8191
    a, b= random.randint(1,p-1), random.randint(0,p-1)
    def hash_func(u_v): 
        u, v= u_v[0], u_v[1]   
        h_u, h_v = ((a * u + b) % p) % C, ((a * v + b) % p) % C
        hash_out = []
        for i in range(C):
            key = [h_u,h_v,i]
            key.sort()
            key = tuple(key)
            hash_out.append( (key, u_v) )
        return hash_out

    out = (edges_rdd.flatMap(hash_func)
            .groupByKey()
            .flatMap(lambda x: countTriangles2(colors_tuple=x[0], edges=x[1], rand_a=a, rand_b=b, p=p, num_colors=C))  ### That's the point!!
            .sum()
            )
    return out


def run_MR_ApproxTCwithNodeColors(R, edges, C):
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
    return Median_N_Triangles, Average_Time_NodeColor


def run_MR_Exact(R, edges, C):
    # MR_ExactTC
    Time_Exact_list = []
    for _ in range(R):
        start = time.time()
        N_Triangles_Exact= MR_ExactTC(edges, C)
        Exact_RunningTime = (time.time() - start) * 1000
        Time_Exact_list.append(Exact_RunningTime)

    Average_Time_Exact = np.array(Time_Exact_list).mean()
    return N_Triangles_Exact, Average_Time_Exact

def main():
    
    assert len(sys.argv) == 5, "Usage: python GxxxHW1.py <C> <R> <F> <input_file_name>"
        
    C = sys.argv[1]
    R = sys.argv[2]
    F = sys.argv[3]
    data_path = sys.argv[4]
    assert C.isdigit(), "C must be an integer"
    assert R.isdigit(), "R must be an integer"
    assert F.isdigit(), "F must be an integer"
    C, R, F = int(C), int(R), int(F)
    
    
    conf = SparkConf().setAppName('TriangleCounting')
    conf.set("spark.locality.wait", "0s")
    sc = SparkContext(conf=conf)
    rawData = sc.textFile(data_path,minPartitions=32).cache()
    edges = rawData.flatMap(raw_to_edges)

    N = edges.count()

    if F == 0:
        Median_N_Triangles, Average_Time_NodeColor = run_MR_ApproxTCwithNodeColors(R, edges, C)
    else:
        N_Triangles_Exact, Average_Time_Exact = run_MR_Exact(R, edges, C)

    
    print("Dataset =", data_path)
    print("Number of Edges =", N)
    print("Number of Colors =", C)
    print("Number of Repetitions =", R)
    if F ==0:
        print("Approximation algorithm with node coloring")
        print("- Number of triangles (median over", R,"runs) =", int(Median_N_Triangles))
        print("- Running time (average over",R,"runs) =", int(Average_Time_NodeColor),"ms")
    else:
        print("Exact algorithm with node coloring")                 
        print("- Number of triangles =", int(N_Triangles_Exact))
        print("- Running time (average over", R, "runs) =", int(Average_Time_Exact),"ms")   


if __name__ == "__main__":
    main()
