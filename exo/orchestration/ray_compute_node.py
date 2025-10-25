import numpy as np
import json
import asyncio
import uuid
import subprocess
import time
import traceback
from typing import List, Dict, Optional, Tuple

import ray

from exo import DEBUG
from exo.networking import Discovery, PeerHandle, Server
from exo.topology.topology import Topology
from exo.topology.device_capabilities import UNKNOWN_DEVICE_CAPABILITIES
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.helpers import AsyncCallbackSystem, get_local_ip
from exo.orchestration import Node
from exo.viz.topology_viz import TopologyViz
from exo.download.download_progress import RepoProgressEvent


async def init_ray_worker(head_node_addr: str):
    """
    Ensures a local raylet is running, joins the cluster at `head_node_addr`,
    and initializes Ray for this Python process.
    """
    local_ip = await get_local_ip()

    # start raylet
    proc = await asyncio.create_subprocess_shell(
        f"ray stop --force && ray start --address='{head_node_addr}' --node-ip-address={local_ip}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode() or stdout.decode())
    else:
      if DEBUG >= 1:
        print(f"Local Ray runtime started on {local_ip}, connecting to head {head_node_addr}")


async def shutdown_ray_worker():
    proc = await asyncio.create_subprocess_shell(
        f"ray stop --force",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode() or stdout.decode())
    else:
      if DEBUG >= 1:
        print(f"Local Ray runtime closed")


class RayComputeNode(Node):
  def __init__(
    self,
    _id: str,
    server: Server,
    discovery: Discovery,
    topology_viz: Optional[TopologyViz] = None,
  ):
    self.id = _id
    self.server = server
    self.discovery = discovery
    # TODO: refactor topology_viz to make partitioning optional
    self.partitioning_strategy = RingMemoryWeightedPartitioningStrategy()
    self.peers: List[PeerHandle] = {}
    self.topology: Topology = Topology()
    self.device_capabilities = UNKNOWN_DEVICE_CAPABILITIES
    self.buffered_token_output: Dict[str, Tuple[List[int], bool]] = {}
    self.buffered_logits: Dict[str, List[np.ndarray]] = {}
    self.buffered_inputs: Dict[str, List[np.ndarray]] = {}
    self.buffered_partials: Dict[str, List[np.ndarray]] = {}
    self.checkpoints: Dict[str, Dict[str, int]] = {}
    
    self.topology_viz = topology_viz
    self._on_opaque_status = AsyncCallbackSystem[str, Tuple[str, str]]()
    self._on_opaque_status.register("node_status").on_next(self.on_node_status)
    self.node_download_progress: Dict[str, RepoProgressEvent] = {}
    self.topology_inference_engines_pool: List[List[str]] = []
    self.outstanding_requests = {}

  async def start_ray_peers(
      self,
      head_node_address: str,
      request_id: Optional[str] = None,
      inference_state: Optional[dict] = {},
  ) -> Optional[np.ndarray]:
    
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "start_ray_compute",
          "request_id": request_id,
          "head_node_address": head_node_address,
        }),
      )
    )

  async def stop_ray_peers(
      self,
      request_id: Optional[str] = None,
      inference_state: Optional[dict] = {},
  ) -> Optional[np.ndarray]:
    
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "end_ray_compute",
        }),
      )
    )

  def on_node_status(self, request_id, opaque_status):
    try:
      status_data = json.loads(opaque_status)
      status_type = status_data.get("type", "")
      if status_type == "supported_inference_engines":
        node_id = status_data.get("node_id")
        engines = status_data.get("engines", [])
        self.topology_inference_engines_pool.append(engines)
      elif status_type == "node_status":
        if status_data.get("status", "").startswith("start_"):
          self.current_topology.active_node_id = active_node_id = status_data.get("node_id")
          if active_node_id != self.id:
            head_node_addr = status_data.get("head_node_address")

            asyncio.create_task(
              init_ray_worker(head_node_addr)
            )

            if DEBUG >= 1: print(f"[Node '{self.id}'] joined cluster: '{active_node_id}'")
        elif status_data.get("status", "").startswith("end_"):
          active_node_id = status_data.get("node_id")
          if active_node_id == self.current_topology.active_node_id:
            self.current_topology.active_node_id = None

          if active_node_id != self.id:
            head_node_addr = status_data.get("head_node_address")
            asyncio.create_task(
              shutdown_ray_worker()
            )
            if DEBUG >= 1: print(f"[Node '{self.id}'] shut down ray cluster of head node: '{active_node_id}'")
    except Exception as e:
      if DEBUG >= 1: print(f"Error on_node_status: {e}")
      if DEBUG >= 1: traceback.print_exc()