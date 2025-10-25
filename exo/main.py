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
from exo.runner.ray_compute import submit_ray_job

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

def submit_ray_job_cmd():
   submit_ray_job()

if __name__ == "__main__":
  run_chat()
