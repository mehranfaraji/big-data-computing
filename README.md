# big-data-computing

This repository contains the my projects for Big Data Computing.

THe WordCountExample.py contains an example code using pyspark to count the numer of distinct words on the document\s

Usage: ``` python WordCountExample.py <K> <file_name> ```

In wich ```K``` is the number of partitions and ```file_name``` is the name of a file or a folder.

## HW1

HW1 folder contains code and some dataset file (in the .txt format) to estimate the number of riangles in an undirected graph using two PySpark algorithms, Approximation through node coloring and Approximation through Spark partitions.

Usage: ``` python G058HW1.py <C> <R> <input_file_name> ``` 

In wich ```C``` is the number of partitons and ```R``` is the number of repetiotion of the triangle count with code colors algorithm to obtain better estimation.

## HW2

HW2 folder contains code and sample dataset (in the .txt format) to estimate the number of riangles in an undirected graph using two PySpark algorithms, Approximation through node coloring and Approximation through Spark partitions.

Usage: ```python G058HW2.py <C> <R> <F> <input_file_name>```

In which ```C``` and ```R``` are the same parameters and ```F=0``` then MR_ApproxTCwithNodeColors algorithm is used 
and if ```F=0``` then MR_Exact algorithm is used. 

## HW3 

HW3 folder contains code that uses the Spark Streaming API which processes a stream of items and assesses experimentally the space-accuracy tradeoffs 
featured by the count sketch to estimate the individual frequencies of the items and the second moment F2.

Usage: ```python G058HW2.py "D, W, left, right, K, portExp```
In which:

An integer D: the number of rows of the count sketch

An integer W: the number of columns of the count sketch

An integer left: the left endpoint of the interval of interest

An integer right: the right endpoint of the interval of interest

An integer K: the number of top frequent items of interest

An integer portExp: the port number
