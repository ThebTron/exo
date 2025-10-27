import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import Field
import ray
from ray.dashboard.modules.job.sdk import JobSubmissionClient
import requests
import uvicorn

from exo import DEBUG
from exo.helpers import get_local_ip, ensure_endpoint_alive
from exo.orchestration import RayComputeNode
from exo.runner import ExoRunner
from exo.viz.topology_viz import TopologyViz

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"


async def start_head_and_connect(port=6379, dashboard_port=8265):
    if ray.is_initialized():
      ray.shutdown()

    head_ip = await get_local_ip()
    # start a cluster head
    cmd = f"ray stop --force && ray start --head --node-ip-address={head_ip} --port={port} --dashboard-port={dashboard_port}"
    
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        print(f"Ray head failed to start:\n{stderr.decode() or stdout.decode()}")
        return None
    else:
        if DEBUG >= 1:
            print(f"Ray head started on {head_ip}:{port}")
    
    loop = asyncio.get_running_loop()
    ray_info = await loop.run_in_executor(
        None,
        lambda: ray.init(address=f"{head_ip}:{port}", log_to_driver=False)
    )

    if DEBUG >=1:
        print(f"Connected to Ray cluster at {ray_info.address_info}")

    return ray_info

async def _start_ray_peers(node: RayComputeNode, response_timeout: int):
    info = await start_head_and_connect()
    request_id = str(uuid.uuid4())

    # launch peer nodes
    await asyncio.wait_for(
        asyncio.shield(
            asyncio.create_task(
                node.start_ray_peers(
                    request_id=request_id,
                    head_node_address=info['address'],
                )
            )
        ), 
        timeout=response_timeout
    )
    if DEBUG >= 1:
        print(f"[Node '{node.id}'] head of ray cluster: '{info['address']}'")
    return info


async def _stop_ray_peers(node: RayComputeNode, response_timeout: int):
    # launch peer nodes
    await asyncio.wait_for(
        asyncio.shield(
            asyncio.create_task(
                node.stop_ray_peers()
            )
        ), 
        timeout=response_timeout
    )
    ray.shutdown()
    if DEBUG >= 1:
        print(f"[Node '{node.id}'] stop ray cluster")


class ExoRayComputeRunner(ExoRunner):
    response_timeout: int = 90
    api_port: int = Field(52415, description="Port for job submission server.")

    def setup_node(self):
        self._topology_viz = (
            TopologyViz(
                add_info=["'Running ExoRayComputeRunner'"]
            ) 
            if not self.disable_tui else None
        )

        self._node = RayComputeNode(
            self.node_id,
            None,
            self._discovery,
            topology_viz=self._topology_viz,
        )
        self.setup_server(node=self._node)
        self._node.server = self._server

    def setup_api(self):
        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.post("/start_ray")
        async def start_ray():
            try:
                info = await _start_ray_peers(
                    node=self._node,
                    response_timeout=self.response_timeout,
                )
                return {
                    "status": "success",
                    "message": {
                        "head_node_address": info["address"],
                    }
                }
            except Exception as e:
                return {
                    "status": "failure",
                    "message": e,  
                }
                
        @app.post("/stop_ray")
        async def stop_ray():
            try:
                info = await _stop_ray_peers(
                    node=self._node,
                    response_timeout=self.response_timeout,
                )
                
                return {
                    "status": "success",
                    "message": "ray server stopped"
                }
            except Exception as e:
                return {
                    "status": "failure",
                    "message": e,  
                }
        
        # setup api
        config = uvicorn.Config(
            app, 
            host="0.0.0.0", 
            port=self.api_port, 
            loop="asyncio",
            log_level="warning",
            access_log=False,
        )
        self._api = uvicorn.Server(config)

    async def run(self):
        await self._api.serve()
        

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