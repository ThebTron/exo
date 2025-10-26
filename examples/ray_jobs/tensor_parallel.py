import ray
import socket
import time
import numpy as np

ray.init(address="auto")

@ray.remote(num_cpus=8) # just to force distribution across multiple workers
def matmul_slice(a_slice, b):
    # each worker handles part of A * B
    time.sleep(2)
    res = a_slice @ b
    return {
        "result": res,
        "host_name": socket.gethostname(),
        "host_ip_address": ray._private.services.get_node_ip_address(),
    }

# Example large matrices
A = np.random.randn(4000, 4000)
B = np.random.randn(4000, 4000)

# Split A into chunks along the row dimension
num_shards = 10
A_shards = np.array_split(A, num_shards, axis=0)

# Dispatch parallel matmul operations
futures = [matmul_slice.remote(shard, B) for shard in A_shards]

# Gather partial results and combine along rows
result = ray.get(futures)
for r in result:
    print(
        f"host={r['host_name']} ({r['host_ip_address']}), "
        f"result={r['result'].shape}"
    )
final_result = np.concatenate(
    [res["result"] for res in result], 
    axis=0
)

print(final_result.shape)