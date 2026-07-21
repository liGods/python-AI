from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ok_tasks.card_ai.features import RANKS, encode_candidate
from ok_tasks.card_ai.model_registry import ModelRegistry


def _encode_inputs(observation: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not candidates:
        return {}
    # Observation/history features are identical for every legal action. Encoding them
    # once avoids repeatedly hashing skills and walking up to 64 history entries.
    common = encode_candidate(observation, candidates[0])
    count = len(candidates)
    static = np.repeat(common["static"][None, :], count, axis=0)
    action_start = len(RANKS) * 2
    static[:, action_start : action_start + len(RANKS)] = 0.0
    rank_indices = {rank: index for index, rank in enumerate(RANKS)}
    for row, candidate in enumerate(candidates):
        for rank in candidate.get("ranks", candidate.get("cards", [])):
            index = rank_indices.get(rank)
            if index is not None:
                static[row, action_start + index] += 1.0
    return {
        "static": static.astype(np.float32, copy=False),
        "history": np.repeat(common["history"][None, :, :], count, axis=0).astype(np.float32, copy=False),
        "history_mask": np.repeat(common["history_mask"][None, :], count, axis=0).astype(np.bool_, copy=False),
        "role_index": np.full(count, common["role_index"], dtype=np.int64),
    }


def legacy_state_to_observation(state: dict[str, Any]) -> dict[str, Any]:
    history = []
    for index, value in enumerate(state.get("history", [])):
        if isinstance(value, dict):
            history.append(dict(value))
        else:
            history.append(
                {
                    "kind": "play" if value else "pass",
                    "actor": None,
                    "ranks": list(value or []),
                    "action_type": "unknown",
                    "turn": index,
                }
            )
    return {
        "game_id": state.get("game_id", state.get("round_id", "runtime")),
        "observer": state.get("position", "landlord_down"),
        "hand": [
            {"card_id": f"screen_{index}", "rank": rank, "source": "screen", "tags": []}
            for index, rank in enumerate(state.get("hand_cards", []))
        ],
        "current_player": state.get("position", "landlord_down"),
        "landlord": "landlord",
        "target_ranks": list(state.get("table_cards", [])),
        "target_action_type": state.get("table_action_type", "unknown"),
        "trick_owner": state.get("trick_owner"),
        "opponent_card_counts": list(state.get("opponent_card_counts", [17, 17])),
        "history": history,
        "hero": state.get("hero"),
        "hero_state": dict(state.get("hero_state", {})),
    }


class OpenVINOActionRankerV3:
    def __init__(self, model_path: str | Path):
        import openvino as ov

        self.model_path = Path(model_path)
        self.core = ov.Core()
        compile_config: dict[str, Any] = {"PERFORMANCE_HINT": "LATENCY"}
        thread_limit = os.environ.get("OPENVINO_CPU_THREADS_NUM")
        if thread_limit:
            compile_config.update({"INFERENCE_NUM_THREADS": int(thread_limit), "NUM_STREAMS": 1})
        self.compiled = self.core.compile_model(self.core.read_model(self.model_path), "CPU", compile_config)

    def score(self, observation: dict[str, Any], candidates: list[dict[str, Any]]) -> np.ndarray:
        if not candidates:
            return np.empty(0, dtype=np.float32)
        inputs = _encode_inputs(observation, candidates)
        result = self.compiled(inputs)
        try:
            output = result[self.compiled.output("action_score")]
        except RuntimeError:
            output = result[self.compiled.output(0)]
        return np.asarray(output, dtype=np.float32).reshape(-1)


class ONNXRuntimeActionRankerV3:
    """CUDA action ranker with an explicit failure when GPU execution is unavailable."""

    def __init__(self, model_path: str | Path):
        import onnxruntime as ort

        # This loads CUDA/cuDNN from the installed PyTorch package without importing
        # PyTorch itself, saving roughly 600 MB of RAM in every evaluation worker.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")
        self.model_path = Path(model_path)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "do_copy_in_default_stream": 1,
                    },
                ),
                "CPUExecutionProvider",
            ],
        )
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"ONNX Runtime silently fell back to {self.session.get_providers()}")
        output_names = {output.name for output in self.session.get_outputs()}
        self.output_name = "action_score" if "action_score" in output_names else self.session.get_outputs()[0].name

    def score(self, observation: dict[str, Any], candidates: list[dict[str, Any]]) -> np.ndarray:
        if not candidates:
            return np.empty(0, dtype=np.float32)
        output = self.session.run([self.output_name], _encode_inputs(observation, candidates))[0]
        return np.asarray(output, dtype=np.float32).reshape(-1)


def create_action_ranker(model_path: str | Path, backend: str = "auto"):
    """Create the requested ranker; auto prefers CUDA and safely falls back to OpenVINO."""
    path = Path(model_path)
    selected = backend.strip().lower()
    if selected not in {"auto", "cuda", "openvino"}:
        raise ValueError(f"Unsupported inference backend: {backend}")
    if selected in {"auto", "cuda"}:
        onnx_path = path if path.suffix.lower() == ".onnx" else path.with_name("student.onnx")
        if onnx_path.is_file():
            try:
                return ONNXRuntimeActionRankerV3(onnx_path)
            except (ImportError, OSError, RuntimeError):
                if selected == "cuda":
                    raise
        elif selected == "cuda":
            raise FileNotFoundError(onnx_path)
    xml_path = path if path.suffix.lower() == ".xml" else path.with_name("student.xml")
    return OpenVINOActionRankerV3(xml_path)


def load_stable_ranker(registry_root: str | Path) -> tuple[OpenVINOActionRankerV3 | None, dict[str, Any]]:
    registry = ModelRegistry(registry_root)
    pointers = registry._pointers()
    version_id = pointers.get("stable")
    if not version_id:
        return None, {"reason": "模型注册表没有稳定版本"}
    manifest = registry.load_manifest(version_id)
    xml_name = manifest.files.get("openvino_xml")
    if not xml_name:
        return None, {"reason": "稳定版本没有OpenVINO文件", "version_id": version_id}
    path = registry.versions / version_id / xml_name
    return OpenVINOActionRankerV3(path), {"version_id": version_id, "manifest": manifest.to_dict()}


def load_version_ranker(registry_root: str | Path, version_id: str) -> OpenVINOActionRankerV3:
    registry = ModelRegistry(registry_root)
    manifest = registry.load_manifest(version_id)
    xml_name = manifest.files.get("openvino_xml")
    if not xml_name:
        raise FileNotFoundError(f"模型版本 {version_id} 没有OpenVINO文件")
    return OpenVINOActionRankerV3(registry.versions / version_id / xml_name)


def read_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
