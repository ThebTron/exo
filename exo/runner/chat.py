import asyncio
import json
import numpy as np
import time
import tqdm
import traceback
import uuid

from pydantic import Field
from typing import Literal, Optional

from exo.train.dataset import load_dataset, iterate_batches
from exo.orchestration import ChatNode
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.api import ChatGPTAPI
from exo.download.download_progress import RepoProgressEvent
from exo.download.new_shard_download import new_shard_downloader
from exo.download.shard_download import ShardDownloader, NoopShardDownloader
from exo.helpers import print_yellow_exo, DEBUG, get_system_info, get_all_ip_addresses_and_interfaces, terminal_link
from exo.inference.inference_engine import get_inference_engine
from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from exo.models import build_base_shard, get_repo
from exo.runner import ExoRunner
from exo.viz.topology_viz import TopologyViz


class ExoChatRunner(ExoRunner):
    # Core command
    command: Optional[Literal["run", "eval", "train"]] = Field(
        None, description="Main command to execute."
    )
    supported_model_name: Optional[str] = Field(
        None, description="Name of the model to run."
    )

    # Training parameters
    default_model: Optional[str] = Field(None, description="Default model name.")
    iters: int = Field(100, description="Number of training iterations.")
    save_every: int = Field(5, description="Save checkpoint every N iterations.")
    data: str = Field("exo/train/data/lora", description="Directory containing training data.")
    batch_size: int = Field(1, description="Mini-batch size.")
    resume_checkpoint: Optional[str] = Field(None, description="Path to resume checkpoint.")
    save_checkpoint_dir: str = Field("checkpoints", description="Directory for saving checkpoints.")

    # Download parameters
    download_quick_check: bool = Field(False, description="Quick check for model shard downloads.")
    max_parallel_downloads: int = Field(8, description="Maximum parallel shard downloads.")

    # ChatGPT API configuration
    chatgpt_api_port: int = Field(52415, description="Port for ChatGPT API web server.")
    chatgpt_api_response_timeout: int = Field(
        900, description="Timeout for ChatGPT API responses (seconds)."
    )
    max_generate_tokens: int = Field(10000, description="Maximum tokens to generate per request.")
    system_prompt: Optional[str] = Field(None, description="System prompt for ChatGPT API.")

    # Inference configuration
    inference_engine: Optional[str] = Field(
        None, description="Inference engine to use (mlx, tinygrad, dummy, etc.)."
    )
    run_model: Optional[str] = Field(None, description="Model name to run directly.")
    prompt: str = Field("Who are you?", description="Prompt for model when using --run-model.")
    default_temp: float = Field(0.0, description="Default token sampling temperature.")

    def setup_node(self):
        # setup inference engine
        print(f"Selected inference engine: {self.inference_engine}")

        system_info = get_system_info()
        print(f"Detected system: {system_info}")

        self.setup_shard_downloader()

        inference_engine_name = self.inference_engine or ("mlx" if system_info == "Apple Silicon Mac" else "tinygrad")
        print(f"Inference engine name after selection: {inference_engine_name}")

        self._inference_engine = get_inference_engine(inference_engine_name, self._shard_downloader)
        print(f"Using inference engine: {self._inference_engine.__class__.__name__} with shard downloader: {self._shard_downloader.__class__.__name__}")

        # setup viz
        chatgpt_api_endpoints = [f"http://{ip}:{self.chatgpt_api_port}/v1/chat/completions" for ip, _ in get_all_ip_addresses_and_interfaces()]
        web_chat_urls = [f"http://{ip}:{self.chatgpt_api_port}" for ip, _ in get_all_ip_addresses_and_interfaces()]
        if DEBUG >= 0:
            print("Chat interface started:")
            for web_chat_url in web_chat_urls:
                print(f" - {terminal_link(web_chat_url)}")
            print("ChatGPT API endpoint served at:")
            for chatgpt_api_endpoint in chatgpt_api_endpoints:
                print(f" - {terminal_link(chatgpt_api_endpoint)}")

        info_lines = []
        if len(web_chat_urls) > 0:
            info_lines.append(f"Web Chat URL (tinychat): {' '.join(web_chat_urls[:1])}")
        if len(chatgpt_api_endpoints) > 0:
            info_lines.append(f"ChatGPT API endpoint: {' '.join(chatgpt_api_endpoints[:1])}")

        self._topology_viz = (
            TopologyViz(add_info=info_lines) 
            if not self.disable_tui else None
        )

        # setup node
        self._node = ChatNode(
            self.node_id,
            None,
            self._inference_engine,
            self._discovery,
            self._shard_downloader,
            partitioning_strategy=RingMemoryWeightedPartitioningStrategy(),
            max_generate_tokens=self.max_generate_tokens,
            topology_viz=self._topology_viz,
            default_sample_temperature=self.default_temp
        )
        self.setup_server(node=self._node)
        self._node.server = self._server

        # setup topology visualization updates
        self._buffered_token_output = {}
        def update_topology_viz(req_id, tokens, __):
            if not self._topology_viz: return
            if not self._node.inference_engine.shard: return
            if self._node.inference_engine.shard.model_id == 'stable-diffusion-2-1-base': return
            if req_id in self._buffered_token_output: self._buffered_token_output[req_id].extend(tokens)
            else: self._buffered_token_output[req_id] = tokens
            self._topology_viz.update_prompt_output(req_id, self._node.inference_engine.tokenizer.decode(self._buffered_token_output[req_id]))
        self._node.on_token.register("update_topology_viz").on_next(update_topology_viz)

        def update_prompt_viz(request_id, opaque_status: str):
            if not self._topology_viz: return
            try:
                status = json.loads(opaque_status)
                if status.get("type") != "node_status" or status.get("status") != "start_process_prompt": return
                self._topology_viz.update_prompt(request_id, status.get("prompt", "corrupted prompt (this should never happen)"))
            except Exception as e:
                if DEBUG >= 2:
                    print(f"Failed to update prompt viz: {e}")
                    traceback.print_exc()
        self._node.on_opaque_status.register("update_prompt_viz").on_next(update_prompt_viz)

        self.setup_preemptively_load_shard()

        self.setup_throttled_broadcast()

    def setup_shard_downloader(self):
        self._shard_downloader: ShardDownloader = new_shard_downloader(self.max_parallel_downloads) if self.inference_engine != "dummy" else NoopShardDownloader()

    def setup_preemptively_load_shard(self):
        def preemptively_load_shard(request_id: str, opaque_status: str):
            try:
                status = json.loads(opaque_status)
                if status.get("type") != "node_status" or status.get("status") != "start_process_prompt": return
                current_shard = self._node.get_current_shard(Shard.from_dict(status.get("shard")))
                if DEBUG >= 2: print(f"Preemptively starting download for {current_shard}")
                asyncio.create_task(self._node.inference_engine.ensure_shard(current_shard))
            except Exception as e:
                if DEBUG >= 2:
                    print(f"Failed to preemptively start download: {e}")
                    traceback.print_exc()
        self._node.on_opaque_status.register("preemptively_load_shard").on_next(preemptively_load_shard)

    def setup_throttled_broadcast(self):
        self._last_events: dict[str, tuple[float, RepoProgressEvent]] = {}
        def throttled_broadcast(shard: Shard, event: RepoProgressEvent):
            current_time = time.time()
            if event.status == "not_started": return
            last_event = self._last_events.get(shard.model_id)
            if last_event and last_event[1].status == "complete" and event.status == "complete": return
            if last_event and last_event[0] == event.status and current_time - last_event[0] < 0.2: return
            self._last_events[shard.model_id] = (current_time, event)
            asyncio.create_task(self._node.broadcast_opaque_status("", json.dumps({"type": "download_progress", "node_id": self._node.id, "progress": event.to_dict()})))
        self._shard_downloader.on_progress.register("broadcast").on_next(throttled_broadcast)

    def setup_api(self):
        self._api = ChatGPTAPI(
            self._node,
            self._node.inference_engine.__class__.__name__,
            response_timeout=self.chatgpt_api_response_timeout,
            on_chat_completion_request=(
                lambda req_id, __, prompt: self._topology_viz.update_prompt(req_id, prompt) 
                if self._topology_viz else None
            ),
            default_model=self.default_model,
            system_prompt=self.system_prompt
        )

    async def run_model_cli(self, node: ChatNode, model_name: str, prompt: str):
        inference_class = node.inference_engine.__class__.__name__
        shard = build_base_shard(model_name, inference_class)
        if not shard:
            print(f"Error: Unsupported model '{model_name}' for inference engine {inference_class}")
            return
        tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
        request_id = str(uuid.uuid4())
        callback_id = f"cli-wait-response-{request_id}"
        callback = node.on_token.register(callback_id)
        if self._topology_viz:
            self._topology_viz.update_prompt(request_id, prompt)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

        try:
            print(f"Processing prompt: {prompt}")
            await node.process_prompt(shard, prompt, request_id=request_id)

            tokens = []
            def on_token(_request_id, _tokens, _is_finished):
                tokens.extend(_tokens)
                return _request_id == request_id and _is_finished
            await callback.wait(on_token, timeout=300)

            print("\nGenerated response:")
            print(tokenizer.decode(tokens))
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")
            traceback.print_exc()
        finally:
            node.on_token.deregister(callback_id)

    @staticmethod
    async def hold_outstanding(node: ChatNode):
        while node.outstanding_requests:
            await asyncio.sleep(.5)
        return
    
    @staticmethod
    async def run_iter(node: ChatNode, shard: Shard, train: bool, data, batch_size=1):
        losses = []
        tokens = []
        for batch in tqdm(iterate_batches(data, batch_size), total=len(data) // batch_size):
            _, _, lengths = batch
            losses.append(np.sum(lengths * await node.enqueue_example(shard, *batch, train=train)))
            tokens.append(np.sum(lengths))
        total_tokens = np.sum(tokens)
        total_loss = np.sum(losses) / total_tokens

        return total_loss, total_tokens

    async def eval_model_cli(self, node: ChatNode, model_name, dataloader, batch_size, num_batches=-1):
        inference_class = node.inference_engine.__class__.__name__
        shard = build_base_shard(model_name, inference_class)
        if not shard:
            print(f"Error: Unsupported model '{model_name}' for inference engine {inference_class}")
            return
        tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
        train, val, test = dataloader(tokenizer.encode)
        print(f"Evaluating {len(test)} examples with batch_size {batch_size}")
        loss, tokens = await self.run_iter(node, shard, False, test, batch_size)
        print(f"total | {loss=}, {tokens=}")
        print("Waiting for outstanding tasks")
        await self.hold_outstanding(node)

    async def train_model_cli(self, node: ChatNode, model_name, dataloader, batch_size, iters, save_interval=0, checkpoint_dir=None):
        inference_class = node.inference_engine.__class__.__name__
        shard = build_base_shard(model_name, inference_class)
        if not shard:
            print(f"Error: Unsupported model '{model_name}' for inference engine {inference_class}")
            return
        tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
        train, val, test = dataloader(tokenizer.encode)
        print(f"Training on {len(train)} examples with batch_size {batch_size} for {iters} epochs")
        for i in tqdm(range(3)):
            await asyncio.sleep(1)
        for epoch in range(iters):
            loss, tokens = await self.run_iter(node, shard, True, train, batch_size)
            print(f"epoch {epoch + 1}/{iters}\t| loss: {loss}, tokens: {tokens}")
            if save_interval > 0 and epoch > 0 and (epoch % save_interval) == 0 and checkpoint_dir is not None:
                await node.coordinate_save(shard, epoch, checkpoint_dir)
                await self.hold_outstanding(node)
        await self.hold_outstanding(node)

    async def run(self):
        if self.command == "run" or self.run_model:
            model_name = self.model_name or self.run_model
            if not model_name:
                print("Error: Model name is required when using 'run' command or --run-model")
                return
            await self.run_model_cli(self._node, model_name, self.prompt)
        elif self.command == "eval" or self.command == 'train':
            model_name = self.model_name
            dataloader = lambda tok: load_dataset(
                self.data, 
                preprocess=lambda item: tok(item), 
                loadline=lambda line: json.loads(line).get("text","")
            )
            if self.command == 'eval':
                if not model_name:
                    print("Error: Much like a human, I can't evaluate anything without a model")
                    return
                await self.eval_model_cli(
                    self._node, model_name, dataloader, self.batch_size
                )
            else:
                if not model_name:
                    print("Error: This train ain't leaving the station without a model")
                    return
                await self.train_model_cli(
                    self._node, 
                    model_name, 
                    dataloader, 
                    self.batch_size, 
                    self.iters, 
                    save_interval=self.save_every, 
                    checkpoint_dir=self.save_checkpoint_dir
                )
        else:
            asyncio.create_task(self._api.run(port=self.chatgpt_api_port))  # Start the API server as a non-blocking task
            await asyncio.Event().wait()