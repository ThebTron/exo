import asyncio
import concurrent.futures
import psutil
import os
import resource
import uvloop

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
   

if __name__ == "__main__":
  run_chat()
