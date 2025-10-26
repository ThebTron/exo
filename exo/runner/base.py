import argparse
import asyncio
import atexit
import json
import platform
import os
import signal
import tqdm

from pydantic import BaseModel, Field
from typing import Literal, Optional, Union, get_args, get_origin

from exo.download.new_shard_download import has_exo_home_read_access, has_exo_home_write_access, ensure_exo_home, seed_models
from exo.helpers import print_yellow_exo, find_available_port, DEBUG, get_or_create_node_id, shutdown
from exo.networking.manual.manual_discovery import ManualDiscovery
from exo.networking.udp.udp_discovery import UDPDiscovery
from exo.networking.tailscale.tailscale_discovery import TailscaleDiscovery
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.networking.grpc.grpc_server import GRPCServer


class ExoRunner(BaseModel):
    """Base configuration to run node / discovery"""

    # Node configuration
    node_id: Optional[str] = Field(None, description="Unique ID of this node.")
    node_host: str = Field("0.0.0.0", description="Host address of this node.")
    node_port: Optional[int] = Field(None, description="Port number for node communication.")
    models_seed_dir: Optional[str] = Field(None, description="Directory containing model seeds.")

    # Discovery settings
    listen_port: int = Field(5678, description="Listening port for discovery service.")
    broadcast_port: int = Field(5678, description="Broadcast port for discovery.")
    discovery_module: Literal["udp", "tailscale", "manual"] = Field(
        "udp", description="Discovery module to use."
    )
    discovery_timeout: int = Field(30, description="Timeout for discovery, in seconds.")
    discovery_config_path: Optional[str] = Field(None, description="Path to discovery config JSON.")
    wait_for_peers: int = Field(0, description="Number of peers to wait for before starting.")

    # Tailscale integration
    tailscale_api_key: Optional[str] = Field(None, description="Tailscale API key.")
    tailnet_name: Optional[str] = Field(None, description="Tailscale tailnet name.")

    # Node filtering
    node_id_filter: Optional[str] = Field(
        None, description="Comma-separated list of allowed node IDs."
    )
    interface_type_filter: Optional[str] = Field(
        None, description="Comma-separated list of allowed network interface types."
    )

    # Miscellaneous
    disable_tui: Optional[bool] = Field(
        False, description="Disable text user interface (TUI) mode."
    )

    @classmethod
    def from_cli(cls):
        # TODO: we can also use pydantic-cli here

        def unwrap_type(annotation):
            """Return the inner type if annotation is Optional[...] or Union[..., NoneType]."""
            origin = get_origin(annotation)
            if origin is Union:
                args = [a for a in get_args(annotation) if a is not type(None)]
                if len(args) == 1:
                    return args[0]
            return annotation
        
        parser = argparse.ArgumentParser()
        for name, field in cls.model_fields.items():
            arg = f"--{name.replace('_','-')}"
            default = field.default
            _annotation = unwrap_type(field.annotation)
            
            if _annotation is bool:
                parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=default)
            else:
                if get_origin(_annotation) is Literal:
                    parser.add_argument(
                        arg, type=str, default=default, choices=get_args(_annotation)
                    )
                else:
                    parser.add_argument(
                        arg, type=_annotation, default=default
                    )

        parser.add_argument("--config", type=str, help="Optional JSON config file")
        args = parser.parse_args()

        data = {}
        if args.config:
            data = json.load(open(args.config))

        # CLI args override config
        for k, v in vars(args).items():
            if k in data:
                continue
            if v is not None and k != "config":
                data[k] = v
        return cls.model_validate(data)

    def model_post_init(self, __context):
        print_yellow_exo()

        self.setup_node_meta()

        self.setup_discovery()

        self.setup_node()

        self.setup_api()

    def setup_node_meta(self):
        # configure nodes
        if self.node_port is None:
            self.node_port = find_available_port(self.node_host)
            if DEBUG >= 1: print(f"Using available port: {self.node_port}")

        self.node_id = self.node_id or get_or_create_node_id()

    def setup_discovery(self):
        # configure discovery

        # Convert node-id-filter and interface-type-filter to lists if provided
        allowed_node_ids = self.node_id_filter.split(',') if self.node_id_filter else None
        allowed_interface_types = self.interface_type_filter.split(',') if self.interface_type_filter else None

        if self.discovery_module == "udp":
            self._discovery = UDPDiscovery(
                self.node_id,
                self.node_port,
                self.listen_port,
                self.broadcast_port,
                lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
                discovery_timeout=self.discovery_timeout,
                allowed_node_ids=allowed_node_ids,
                allowed_interface_types=allowed_interface_types
            )
        elif self.discovery_module == "tailscale":
            self._discovery = TailscaleDiscovery(
                self.node_id,
                self.node_port,
                lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
                discovery_timeout=self.discovery_timeout,
                tailscale_api_key=self.tailscale_api_key,
                tailnet=self.tailnet_name,
                allowed_node_ids=allowed_node_ids
            )
        elif self.discovery_module == "manual":
            if not self.discovery_config_path:
                raise ValueError(f"--discovery-config-path is required when using manual discovery. Please provide a path to a config json file.")
            self._discovery = ManualDiscovery(self.discovery_config_path, self.node_id, create_peer_handle=lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities))

    def setup_server(self, node):
        self._server = GRPCServer(node, self.node_host, self.node_port)

    def setup_node(self):
        raise NotImplementedError
    
    def setup_api(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    @staticmethod
    async def check_exo_home():
        home, has_read, has_write = await ensure_exo_home(), await has_exo_home_read_access(), await has_exo_home_write_access()
        if DEBUG >= 1: print(f"exo home directory: {home}")
        print(f"{has_read=}, {has_write=}")
        if not has_read or not has_write:
            print(f"""
                WARNING: Limited permissions for exo home directory: {home}.
                This may prevent model downloads from working correctly.
                {"❌ No read access" if not has_read else ""}
                {"❌ No write access" if not has_write else ""}
                """)

    @staticmethod
    def clean_path(path):
        """Clean and resolve path"""
        if path.startswith("Optional("):
            path = path.strip('Optional("').rstrip('")')
        return os.path.expanduser(path)

    async def main(self):
        loop = asyncio.get_running_loop()

        try: await self.check_exo_home()
        except Exception as e: print(f"Error checking exo home directory: {e}")

        if not self.models_seed_dir is None:
            try:
                models_seed_dir = self.clean_path(self.models_seed_dir)
                await seed_models(models_seed_dir)
            except Exception as e:
                print(f"Error seeding models: {e}")

        def restore_cursor():
            if platform.system() != "Windows":
                os.system("tput cnorm")  # Show cursor

        # Restore the cursor when the program exits
        atexit.register(restore_cursor)

        # Use a more direct approach to handle signals
        def handle_exit():
            asyncio.ensure_future(shutdown(signal.SIGTERM, loop, self._node.server))

        if platform.system() != "Windows":
            for s in [signal.SIGINT, signal.SIGTERM]:
                loop.add_signal_handler(s, handle_exit)

        await self._node.start(wait_for_peers=self.wait_for_peers)

        await self.run()

        if self.wait_for_peers > 0:
            print("Cooldown to allow peers to exit gracefully")
            for i in tqdm(range(50)):
                await asyncio.sleep(.1)
