import argparse
import asyncio
import concurrent.futures
from pathlib import Path
import psutil
import os
import requests
import resource
import time

from fastapi import HTTPException
import ray
from ray.dashboard.modules.job.sdk import JobSubmissionClient
import uvloop

from exo import DEBUG
from exo.helpers import ensure_endpoint_alive
from exo.runner import ExoChatRunner, ExoRayComputeRunner

# TODO: figure out why this is happening
os.environ["GRPC_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Configure uvloop for maximum performance
def configure_uvloop():
    uvloop.install()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Increase file descriptor limits on Unix systems
    if not psutil.WINDOWS:
      soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
      try: resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
      except ValueError:
        try: resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))
        except ValueError: pass

    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) * 4)))
    return loop

def run(runner):
    loop = None
    try:
        loop = configure_uvloop()
        loop.run_until_complete(runner.main())
    except KeyboardInterrupt:
        print("\nShutdown requested... exiting")
    finally:
        if loop: loop.close()

def run_chat():
    exo_chat = ExoChatRunner.from_cli()
    run(runner=exo_chat)

def run_ray():
    exo_ray = ExoRayComputeRunner.from_cli()
    run(runner=exo_ray)

def submit_ray_job():
    parser = argparse.ArgumentParser()
    parser.add_argument("python_file_path", type=str)
    parser.add_argument("--api-port", type=int, default=52415)
    parser.add_argument("--setup-wait", type=int, default=2)
    args = parser.parse_args()

    _filepath = Path(args.python_file_path)

    if _filepath.suffix != ".py":
       raise ValueError(f"Only python scripts are supported as valid ray jobs")

    # kill other ray cluster
    ensure_endpoint_alive(args.api_port)
    response = requests.post(f"http://localhost:{args.api_port}/stop_ray").json()

    time.sleep(args.setup_wait)

    # start fresh ray cluster
    response = requests.post(f"http://localhost:{args.api_port}/start_ray").json()    

    time.sleep(args.setup_wait)

    if DEBUG >= 1:
        print(response)

    if response["status"] == "failure":
        raise HTTPException(
            f"Starting ray failed, message: {response['message']}"
        )
    
    # get job submission client
    head_node_address = response["message"]["head_node_address"]
    client = JobSubmissionClient(head_node_address)
    _entrypoint = f"python {_filepath}"
    job_id = client.submit_job(entrypoint=_entrypoint)

    print(f"Submitted ({job_id=}): {_entrypoint}")

    info = ray.init("auto", logging_level="ERROR")
    print(f"Job queue is available at: http://{info['webui_url']}")


if __name__ == "__main__":
  run_chat()
