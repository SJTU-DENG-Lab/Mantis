import time
import enum
import rich
import tqdm
import tyro
import msgpack
import logging
import pathlib
import requests
import dataclasses

import polars as pl
import numpy as np


logger = logging.getLogger(__name__)

def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(data): return msgpack.packb(
    data, default=pack_array, use_bin_type=True)


def unpackb(data): return msgpack.unpackb(
    data, object_hook=unpack_array, raw=False)

class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8001
    use_https: bool = False
    num_steps: int = 20
    timing_file: pathlib.Path | None = None
    env: EnvMode = EnvMode.ALOHA_SIM


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        times = self._timings[key]
        return {"mean": float(np.mean(times)), "std": float(np.std(times)), "p25": float(np.quantile(times, 0.25)), "p50": float(np.quantile(times, 0.50)), "p75": float(np.quantile(times, 0.75)), "p90": float(np.quantile(times, 0.90)), "p95": float(np.quantile(times, 0.95)), "p99": float(np.quantile(times, 0.99)), }

    def print_all_stats(self) -> None:
        table = rich.table.Table(title="[bold blue]HTTP Client Timing Statistics[/bold blue]",
                                 show_header=True, header_style="bold white", border_style="blue", title_justify="center")
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)
        stat_columns = [("Mean", "yellow", "mean"), ("Std", "yellow", "std"), ("P25", "magenta", "p25"), ("P50", "magenta", "p50"),
                        ("P75", "magenta", "p75"), ("P90", "magenta", "p90"), ("P95", "magenta", "p95"), ("P99", "magenta", "p99")]
        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)

class MsgPackHttpClientPolicy:
    def __init__(self, host: str, port: int, use_https: bool = False):
        protocol = "https" if use_https else "http"
        if host.startswith("http"):
            self.base_url = host
        else:
            self.base_url = f"{protocol}://{host}:{port}"

        self.infer_url = f"{self.base_url}/infer"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/msgpack"})
        logger.info(
            f"MsgPack client configured to connect to: {self.base_url}")

    def get_server_metadata(self) -> dict:
        try:
            response = self.session.get(self.base_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get server metadata: {e}")
            return {"error": str(e)}

    def infer(self, observation: dict) -> dict:
        packed_observation = packb(observation)
        try:
            response = self.session.post(
                self.infer_url, data=packed_observation, timeout=80)
            response.raise_for_status()

            unpacked_response = unpackb(response.content)
            return unpacked_response
        except requests.exceptions.RequestException as e:
            logger.error(f"Inference request failed: {e}")
            return {"error": str(e)}


def main(args: Args) -> None:
    obs_fn = {
        EnvMode.ALOHA: _random_observation_aloha,
        EnvMode.ALOHA_SIM: _random_observation_aloha,
        EnvMode.DROID: _random_observation_droid,
        EnvMode.LIBERO: _random_observation_libero,
    }[args.env]

    policy = MsgPackHttpClientPolicy(
        host=args.host, port=args.port, use_https=args.use_https)

    logger.info("Checking server connection...")
    metadata = policy.get_server_metadata()
    # if "error" in metadata:
    #     logger.error("Could not connect to server. Exiting.")
    #     return
    logger.info(f"Successfully connected! Server metadata: {metadata}")

    logger.info("Warming up the server...")
    for _ in range(2):
        policy.infer(obs_fn())

    timing_recorder = TimingRecorder()

    logger.info(f"Running benchmark for {args.num_steps} steps...")
    for _ in tqdm.trange(args.num_steps, desc="Running policy"):
        inference_start_time = time.time()
        action = policy.infer(obs_fn())
        # print(action["actions"].shape)
        if "error" in action:
            logger.error("Stopping benchmark due to an inference error.")
            break
        timing_recorder.record("client_roundtrip_ms",
                               (time.time() - inference_start_time) * 1000)

    timing_recorder.print_all_stats()
    if args.timing_file is not None:
        timing_recorder.write_parquet(args.timing_file)



def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 512, 512), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 512, 512), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 256, 256), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 256, 256), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))