import numpy as np
import ray
import socket
import time

ray.init(address="auto")

@ray.remote(num_cpus=8) # just to force distribution across multiple workers
def one_plus_one(task_id, input_dict):
    res = input_dict["result"] + np.ones(2)
    time.sleep(2)
    return {
        "result": res,
        "task_id": task_id,
        "host_name": socket.gethostname(),
        "host_ip_address": ray._private.services.get_node_ip_address(),
    }

def layer_eval(label, init_array, n=10):
    futures = []
    prev = ray.put({"result": init_array}) # puts init in object store
    for i in range(n):
        f = (
            one_plus_one
            .options(name=f"{label}_{i}")
            .remote(task_id=f"{label}_{i}", input_dict=prev)
        )
        futures.append(f)
        prev = f # builds dependency between runs
    return futures

def print_results(futures):
    results = ray.get(futures)
    for r in results:
        print(f"task_id={r['task_id']}, host={r['host_name']} ({r['host_ip_address']}), "
              f"result={r['result']}")

zeros_futs = layer_eval("zeros", np.zeros(2), n=10)
ones_futs  = layer_eval("ones",  np.ones(2),  n=10)

print_results(zeros_futs)
print_results(ones_futs)