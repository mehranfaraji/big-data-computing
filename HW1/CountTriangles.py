from collections import defaultdict

def CountTriangles(edges): #(0,1)
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    # print(edges)
    # print("*"*40)
    # lofedges = [i for i in edges]
    # print(lofedges)
    # print("len is: ", len(lofedges))
    for edge in edges:
    # for edge in lofedges:
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
    # return triangle_count
    yield triangle_count