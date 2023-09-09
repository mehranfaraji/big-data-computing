from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
from random import randint
import numpy as np

# After how many items should we stop?
THRESHOLD = 10000000

# Operations to perform after receiving an RDD 'batch' at time 'time'
def process_batch(time, batch):
    
    def hash_func(x):
        P = 8191
        a, b = randint(1, P-1), randint(0, P-1)
        return ((a*x + b) % P) % W 
    
    # We are working on the batch at time `time`.
    global streamLength, exact_freq_sigma, exact_freq_sigma_R, dist_items_sigma_R, CountSketch,  left, right, W, D, K
    batch_size = batch.count()
    streamLength[0] += batch_size
    # Extract the distinct items from the batch
    batch_R = batch.filter(lambda e: int(e) in range(left,right+1)).map(lambda e: (int(e), 1))
    streamLength[1] += batch_R.count()
    exact_freq_batch = batch_R.reduceByKey(lambda x,y: x + y).collectAsMap()

            
    # If we wanted, here we could run some additional code on the global
    if batch_size > 0:
        # Update the streaming state
        for key in exact_freq_batch:
            if key not in dist_items_sigma_R:
                dist_items_sigma_R[key] = {"hash": [hash_func(key) for _ in range(D)],
                                           "G": [-1 if randint(-1,0) == -1 else 1 for _ in range(D)]}
                
            if key not in exact_freq_sigma_R:
                exact_freq_sigma_R[key] = exact_freq_batch[key]
            else:
                exact_freq_sigma_R[key] += exact_freq_batch[key]

            for i in range(D):
                CountSketch[i][dist_items_sigma_R[key]['hash'][i]] += (
                    exact_freq_batch[key] * dist_items_sigma_R[key]['G'][i] )

    if streamLength[0] >= THRESHOLD:
        stopping_condition.set()
    

if __name__ == '__main__':
    assert len(sys.argv) == 7, "D, W, left, right, K, portExp"
    
    conf = SparkConf().setMaster("local[*]").setAppName("DistinctExample")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 1)  # Batch duration of 1 second
    ssc.sparkContext.setLogLevel("ERROR")
    
    stopping_condition = threading.Event()
    
    
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # INPUT READING
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    D = int(sys.argv[1])
    W = int(sys.argv[2])
    left = int(sys.argv[3])
    right = int(sys.argv[4])
    K = int(sys.argv[5])
    portExp = int(sys.argv[6])
    print("Receiving data from port =", portExp)
    
    
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # DEFINING THE REQUIRED DATA STRUCTURES TO MAINTAIN THE STATE OF THE STREAM
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    streamLength = [0,0] # Stream length [Σ, ΣR]
    exact_freq_sigma = {}
    exact_freq_sigma_R = {}
    dist_items_sigma_R = {}
    CountSketch = [[0] * W for _ in range(D)]
    

    # CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    # For each batch, to the following.
    # BEWARE: the `foreachRDD` method has "at least once semantics", meaning
    # that the same data might be processed multiple times in case of failure.
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))
    
    # MANAGING STREAMING SPARK CONTEXT
    print("Starting streaming engine")
    ssc.start()
    print("Waiting for shutdown condition")
    stopping_condition.wait()
    print("Stopping the streaming engine")
    # NOTE: You will see some data being processed even after the
    # shutdown command has been issued: This is because we are asking
    # to stop "gracefully", meaning that any outstanding work
    # will be done.
    ssc.stop(False, True)
    print("Streaming engine stopped")

    # COMPUTE AND PRINT FINAL STATISTICS

    F2_true = sum([freq**2 for freq in exact_freq_sigma_R.values()])
    F2_estimate = np.median(
        [sum([value**2 for value in CountSketch[i]]) for i in range(D)])
    F2_true = F2_true / (streamLength[1] **2)
    F2_estimate = F2_estimate / (streamLength[1] **2)
    exact_freq_sigma_R_sorted = sorted(exact_freq_sigma_R.items(), key=lambda x: x[1], reverse=True)
    top_k_frequent_items = exact_freq_sigma_R_sorted[:K]
    relative_error_items = [i[0] for i in top_k_frequent_items]

    estimated_frequency_of_error_items = [(key,
                    np.median([CountSketch[i][dist_items_sigma_R[key]["hash"][i] * dist_items_sigma_R[key]["G"][i]]for i in range(D)]))
                    for key in relative_error_items]

    avarage_relative_error = np.mean(
        [abs(exact_freq_sigma_R[i[0]] - i[1]) / exact_freq_sigma_R[i[0]]
            for i in estimated_frequency_of_error_items])

    if K <= 20:
        for e in estimated_frequency_of_error_items:
            print(f"Item {e[0]} Freq = ", exact_freq_sigma_R[e[0]], f"Est. Freq = {abs(e[1])}",)


    print(f"D = {D}, W = {W}, [left,right]:[{left}, {right}], K = {K}, Port = {portExp}")
    print("Total number of items = ", streamLength[0])
    print(f"Total number of items in [{left, right}] = ", streamLength[1])
    print(f"Number of distinct items in [{left, right}] =  ",len(dist_items_sigma_R),)
    print(f"Avg err for top {K} = ", avarage_relative_error)
    print("F2 = ", F2_true)
    print("F2 Estimate = ", F2_estimate)