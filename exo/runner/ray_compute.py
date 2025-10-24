import json
import os
import uuid

import asyncio
from fastapi import FastAPI
from pydantic import Field
import ray
import uvicorn

from exo import DEBUG
from exo.helpers import print_yellow_exo
from exo.orchestration import RayComputeNode
from exo.runner import ExoRunner
from exo.viz.topology_viz import TopologyViz

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"


async def _start_ray_peers(node: RayComputeNode, response_timeout: int):
    # launch head node
    info = ray.init(
        address=None, 
        include_dashboard=True, 
        log_to_driver=False,
    )
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
        print(f"[Node '{node.id}'] head of cluster: '{info['address']}'")
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
        print(f"[Node '{node.id}'] stop cluster'")


class ExoRayComputeRunner(ExoRunner):
    """TODO"""

    response_timeout: int = 90
    api_port: int = Field(52415, description="Port for job submission server.")

    def model_post_init(self, __context):
        print_yellow_exo()

        self.setup_node_meta()

        self.setup_discovery()

        self.setup_api()

        host, port = self._api.config.host, self._api.config.port

        self._topology_viz = (
            TopologyViz(
                add_info=["'Running ExoRayComputeRunner'"]
            ) 
            if not self.disable_tui else None
        )

        self.setup_node()


    def setup_api(self):
        app = FastAPI()

        @app.post("/start_ray")
        async def start_ray():
            try:
                if not ray.is_initialized():
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
                else:
                    return {
                        "status": "failure",
                        "message": "ray server is already up and running",
                    }
            except Exception as e:
                return {
                    "status": "failure",
                    "message": e,  
                }
                
        @app.post("/stop_ray")
        async def stop_ray():
            try:
                if ray.is_initialized():
                    info = await _stop_ray_peers(
                        node=self._node,
                        response_timeout=self.response_timeout,
                    )
                    
                    return {
                        "status": "success",
                        "message": "ray server stopped"
                    }
                else:
                    return {
                        "status": "failure",
                        "message": "no ray server running to shutdown",
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

    def setup_node(self):
        self._node = RayComputeNode(
            self.node_id,
            None,
            self._discovery,
            topology_viz=self._topology_viz,
        )
        self.setup_server(node=self._node)
        self._node.server = self._server


    async def run(self):
        await self._api.serve()
        