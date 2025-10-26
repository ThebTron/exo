import numpy as np
import ray
import socket
import time

# connect to existing cluster
ray.init(address="auto")

# define task
@ray.remote(num_cpus=4) # just to force distribution across multiple workers
def one_plus_one(task_id, init_array):
    res = init_array + np.ones(2)
    time.sleep(5)
    return {
        "result": res,
        "task_id": task_id,
        "host_name": socket.gethostname(),
        "host_ip_address": ray._private.services.get_node_ip_address(),
    }

# submit 10 independent tasks in parallel
futures = [
    one_plus_one.remote(task_id=i, init_array=np.ones(2)*i) 
    for i in range(10)
]

# block until all finish, then get results
results = ray.get(futures)

# pretty-print where each task ran
for r in results:
    print(
        f"task_id={r['task_id']}, host={r['host_name']} ({r['host_ip_address']}), "
        f"result={r['result']}"
    )