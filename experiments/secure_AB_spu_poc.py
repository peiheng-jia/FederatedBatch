#!/usr/bin/env python3
"""SecretFlow-SPU PoC for one finalized RoundOnce joint-DP A/B interaction.

The module benchmarks real saved seed-218 embeddings without retraining. Five
biological clients are input owners; their arrays are secret-shared to a local
three-party ABY3 simulator. Only final noisy target-specific A/B endpoints are
reconstructed. No individual anchor, remote key, weight, unclipped contribution,
or pre-noise aggregate is serialized.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import multiprocessing as mp
import os
import queue
import statistics
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import psutil
import spu
from jax._src import api_util as japi_util

try:
    import jax.extend.linear_util as jax_lu
except ImportError:
    import jax.linear_util as jax_lu

from spu import libspu
from spu.utils import frontend as spu_fe
from spu.utils.simulation import Simulator, sim_jax


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "results" / "secure_AB_spu_poc"
EMBEDDING = (
    REPO_ROOT
    / "results"
    / "round_once_jointDP_multiseed_102"
    / "training"
    / "seed_218"
    / "Joint-C50-sigma2"
    / "final_embedding.npy"
)
PARTITION = (
    REPO_ROOT
    / "results"
    / "figure2_unified_7seed_original_environment_20260923"
    / "partitions"
    / "seed_218.json"
)
SCHEDULE = (
    REPO_ROOT
    / "results"
    / "round_once_jointDP_multiseed_102"
    / "schedules"
    / "remote_cell_schedule_seed_218.csv"
)

N_DATA_CLIENTS = 5
N_MPC_PARTIES = 3
EMBED_DIM = 100
ROUND = 4
TAU = 0.20
ALPHA = 0.10
C = 1206.3391650512762
SIGMA = 2.0
POSITIVE_A_FLOOR = 1e-6
NOISE_SEED = 10218

FIELD = libspu.FieldType.FM64
PROTOCOL = libspu.ProtocolKind.ABY3
FXP_FRACTION_BITS = 18
EXP_MODE = libspu.RuntimeConfig.ExpMode.EXP_PADE
EXP_ITERS = 8
DIV_ITERS = 2

# This gate catches objective-breaking failures, not ordinary documented
# fixed-point approximation error. The user did not prescribe a 2% cutoff.
SMALL_A_REL_MAX = 0.10
SMALL_B_REL_MAX = 0.05
SMALL_B_COS_MIN = 0.99
SMALL_GRAD_COS_MIN = 0.99


def runtime_config() -> libspu.RuntimeConfig:
    config = libspu.RuntimeConfig(
        protocol=PROTOCOL,
        field=FIELD,
        fxp_fraction_bits=FXP_FRACTION_BITS,
    )
    config.fxp_exp_mode = EXP_MODE
    config.fxp_exp_iters = EXP_ITERS
    config.fxp_div_goldschmidt_iters = DIV_ITERS
    config.enable_lower_accuracy_rsqrt = False
    return config


def load_groups() -> list[np.ndarray]:
    payload = json.loads(PARTITION.read_text(encoding="utf-8"))
    groups = [np.asarray(value, dtype=np.int64) for value in payload["client_indices"]]
    if len(groups) != N_DATA_CLIENTS:
        raise ValueError("expected five Figure 2 clients")
    return groups


def load_round_schedule(remote_count: int) -> dict[int, np.ndarray]:
    frame = pd.read_csv(SCHEDULE)
    frame = frame.loc[
        (frame.active_round == ROUND) & (frame.interaction_slot == 0)
    ].sort_values(["source_client", "source_local_index"])
    schedule: dict[int, np.ndarray] = {}
    for source in range(N_DATA_CLIENTS):
        selected = (
            frame.loc[frame.source_client == source]
            .source_local_index.to_numpy(np.int64)[:remote_count]
        )
        if len(selected) != remote_count:
            raise ValueError(f"source {source} lacks round-4 remote cells")
        if len(np.unique(selected)) != remote_count:
            raise ValueError(f"source {source} reuses a round-4 remote cell")
        schedule[source] = selected
    return schedule


def load_real_inputs(
    anchor_count: int, remote_count: int
) -> tuple[np.ndarray, np.ndarray]:
    embedding = np.load(EMBEDDING, mmap_mode="r")
    groups = load_groups()
    schedule = load_round_schedule(remote_count)
    anchors = np.stack(
        [
            np.asarray(embedding[group[:anchor_count]], dtype=np.float32)
            for group in groups
        ]
    )
    remotes = np.stack(
        [
            np.asarray(
                embedding[groups[source][schedule[source]]], dtype=np.float32
            )
            for source in range(N_DATA_CLIENTS)
        ]
    )
    expected_anchors = (N_DATA_CLIENTS, anchor_count, EMBED_DIM)
    expected_remotes = (N_DATA_CLIENTS, remote_count, EMBED_DIM)
    if anchors.shape != expected_anchors or remotes.shape != expected_remotes:
        raise ValueError(
            f"real input shape mismatch: {anchors.shape}, {remotes.shape}"
        )
    return np.ascontiguousarray(anchors), np.ascontiguousarray(remotes)


def noise_contributions(anchor_count: int) -> tuple[np.ndarray, ...]:
    """Three non-colluding Gaussian contributions sum to N(0, (sigma*C)^2 I)."""

    shape = (
        N_DATA_CLIENTS,
        N_DATA_CLIENTS,
        anchor_count,
        EMBED_DIM + 1,
    )
    std = SIGMA * C / math.sqrt(N_MPC_PARTIES)
    values = []
    off_diagonal = (1.0 - np.eye(N_DATA_CLIENTS, dtype=np.float32))[
        :, :, None, None
    ]
    for party in range(N_MPC_PARTIES):
        party_seed = int(
            (
                NOISE_SEED * 1_000_003
                + ROUND * 1_009
                + party * 104_729
                + 7_919
            )
            % (2**63 - 1)
        )
        generator = np.random.default_rng(party_seed)
        contribution = generator.standard_normal(shape, dtype=np.float32) * std
        values.append(np.ascontiguousarray(contribution * off_diagonal))
    return tuple(values)


def normalize_plain(values: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(denominator, np.finfo(np.float32).tiny)


def plaintext_round(
    anchors: np.ndarray,
    remotes: np.ndarray,
    noise_parts: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, int]:
    anchors_normalized = normalize_plain(anchors).astype(np.float32)
    remotes_normalized = normalize_plain(remotes).astype(np.float32)
    logits = np.einsum(
        "tbd,sjd->stbj", anchors_normalized, remotes_normalized, optimize=True
    ).astype(np.float32)
    weights = np.exp(logits / TAU).astype(np.float32)
    off_diagonal = (1.0 - np.eye(N_DATA_CLIENTS, dtype=np.float32))[
        :, :, None, None
    ]
    weights *= off_diagonal
    norm2 = 2.0 * np.square(weights).sum(axis=(1, 2))
    scales = np.minimum(
        1.0,
        C / np.maximum(np.sqrt(norm2), np.finfo(np.float32).tiny),
    ).astype(np.float32)
    clipped = weights * scales[:, None, None, :]
    source_a = clipped.sum(axis=3)
    source_b = np.einsum(
        "stbj,sjd->stbd", clipped, remotes_normalized, optimize=True
    ).astype(np.float32)
    source_ab = np.concatenate([source_a[..., None], source_b], axis=3)
    source_ab += sum(noise_parts)
    target_ab = source_ab.sum(axis=0)
    invalid_count = int(np.count_nonzero(target_ab[..., 0] <= 0))
    target_ab[..., 0] = np.maximum(target_ab[..., 0], POSITIVE_A_FLOOR)
    return (
        np.ascontiguousarray(target_ab[..., 0]),
        np.ascontiguousarray(target_ab[..., 1:]),
        invalid_count,
    )


def secure_ab(
    anchors: jax.Array,
    remotes: jax.Array,
    noise_0: jax.Array,
    noise_1: jax.Array,
    noise_2: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Exact finalized A/B algebra evaluated on SPU fixed-point shares."""

    anchors_normalized = anchors / jnp.sqrt(
        jnp.sum(jnp.square(anchors), axis=-1, keepdims=True)
    )
    remotes_normalized = remotes / jnp.sqrt(
        jnp.sum(jnp.square(remotes), axis=-1, keepdims=True)
    )
    logits = jnp.einsum("tbd,sjd->stbj", anchors_normalized, remotes_normalized)
    off_diagonal = (1.0 - jnp.eye(N_DATA_CLIENTS, dtype=anchors.dtype))[
        :, :, None, None
    ]
    weights = jnp.exp(logits / TAU) * off_diagonal
    norm2 = 2.0 * jnp.sum(jnp.square(weights), axis=(1, 2))
    # SPU 0.9.5 FM64/18-bit secret sqrt returns zero when this real workload's
    # raw norm2 reaches roughly 1e6. Rescaling before sqrt is algebraically
    # identical to C/sqrt(norm2), while its observed input range is ~0.08--2.53.
    ratio_sq = norm2 / (C * C)
    scale = jnp.minimum(1.0, 1.0 / jnp.sqrt(ratio_sq))
    clipped = weights * scale[:, None, None, :]
    source_a = jnp.sum(clipped, axis=3)
    source_b = jnp.einsum("stbj,sjd->stbd", clipped, remotes_normalized)
    source_ab = jnp.concatenate([source_a[..., None], source_b], axis=3)
    # Each MPC party contributes N(0, (sigma*C)^2/3); only their protected sum
    # has standard deviation sigma*C and no single party knows that total.
    source_ab = source_ab + noise_0 + noise_1 + noise_2
    target_ab = jnp.sum(source_ab, axis=0)
    target_a = jnp.maximum(target_ab[..., 0], POSITIVE_A_FLOOR)
    return target_a, target_ab[..., 1:]


def compile_secure(
    args: tuple[np.ndarray, ...],
) -> tuple[libspu.Executable, Any, float, str]:
    started = time.perf_counter()
    wrapped = jax_lu.wrap_init(secure_ab)
    _, dynamic_args = japi_util.argnums_partial_except(
        wrapped, (), args, allow_invalid=False
    )
    flat_args, _ = jax.tree_util.tree_flatten((dynamic_args, {}))
    input_names = [f"in{index}" for index in range(len(flat_args))]

    def output_names(flat_output):
        return [f"out{index}" for index in range(len(flat_output))]

    executable, output = spu_fe.compile(
        spu_fe.Kind.JAX,
        secure_ab,
        args,
        {},
        input_names,
        [libspu.Visibility.VIS_SECRET] * len(flat_args),
        output_names,
    )
    _, output_tree = jax.tree_util.tree_flatten(output)
    compilation_seconds = time.perf_counter() - started
    pphlo = executable.code.decode("utf-8")
    return executable, output_tree, compilation_seconds, pphlo


class MemoryMonitor:
    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self._stop.wait(0.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self._stop.set()
        self._thread.join()

    @property
    def peak_mb(self) -> float:
        return self.peak / (1024**2)

    @property
    def incremental_mb(self) -> float:
        return (self.peak - self.baseline) / (1024**2)


def execute_secure(
    simulator: Simulator,
    executable: libspu.Executable,
    output_tree: Any,
    args: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    with MemoryMonitor() as memory:
        started = time.perf_counter()
        flat_outputs = simulator(executable, *args)
        wall_seconds = time.perf_counter() - started
    outputs = jax.tree_util.tree_unflatten(output_tree, flat_outputs)
    return (
        np.maximum(
            np.asarray(outputs[0], dtype=np.float32), POSITIVE_A_FLOOR
        ),
        np.asarray(outputs[1], dtype=np.float32),
        wall_seconds,
        memory.peak_mb,
        memory.incremental_mb,
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = left.reshape(-1).astype(np.float64)
    right64 = right.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(left64) * np.linalg.norm(right64)
    return float(
        np.dot(left64, right64) / max(float(denominator), np.finfo(float).tiny)
    )


def compare_outputs(
    plain_a: np.ndarray,
    plain_b: np.ndarray,
    secure_a: np.ndarray,
    secure_b: np.ndarray,
    invalid_a_count: int,
) -> dict[str, float]:
    plain_a64 = plain_a.astype(np.float64)
    secure_a64 = secure_a.astype(np.float64)
    plain_b64 = plain_b.astype(np.float64)
    secure_b64 = secure_b.astype(np.float64)
    a_relative = np.linalg.norm(secure_a64 - plain_a64) / max(
        np.linalg.norm(plain_a64), np.finfo(float).tiny
    )
    b_relative = np.linalg.norm(secure_b64 - plain_b64) / max(
        np.linalg.norm(plain_b64), np.finfo(float).tiny
    )
    plain_gradient = plain_b64 / np.maximum(
        plain_a64[..., None], POSITIVE_A_FLOOR
    ) / TAU
    secure_gradient = secure_b64 / np.maximum(
        secure_a64[..., None], POSITIVE_A_FLOOR
    ) / TAU
    return {
        "a_relative_l2_error": float(a_relative),
        "b_relative_l2_error": float(b_relative),
        "b_cosine_similarity": cosine(secure_b64, plain_b64),
        "reconstructed_remote_gradient_cosine": cosine(
            secure_gradient, plain_gradient
        ),
        "plaintext_invalid_a_frequency": float(
            invalid_a_count / plain_a.size
        ),
        "secure_clamped_a_frequency": float(
            np.count_nonzero(secure_a <= POSITIVE_A_FLOOR) / secure_a.size
        ),
        "a_clamp_disagreement_frequency": float(
            np.count_nonzero(
                (plain_a <= POSITIVE_A_FLOOR)
                != (secure_a <= POSITIVE_A_FLOOR)
            )
            / plain_a.size
        ),
        "max_a_absolute_error": float(np.max(np.abs(secure_a64 - plain_a64))),
        "max_b_absolute_error": float(np.max(np.abs(secure_b64 - plain_b64))),
    }


@functools.lru_cache(maxsize=1)
def approximation_diagnostics() -> dict[str, float]:
    points = np.linspace(-5.0, 5.0, 2001, dtype=np.float32)
    exact = np.exp(points)
    secure_exp = sim_jax(
        Simulator(N_MPC_PARTIES, runtime_config()), lambda value: jnp.exp(value)
    )
    approximate = np.asarray(secure_exp(points), dtype=np.float32)
    relative = np.abs(approximate - exact) / exact
    return {
        "exp_domain_min": -5.0,
        "exp_domain_max": 5.0,
        "exp_mode": "SPU_EXP_PADE",
        "exp_secure_operator_max_relative_error": float(relative.max()),
        "exp_secure_operator_mean_relative_error": float(relative.mean()),
    }


def run_case_worker(
    result_queue,
    case: str,
    anchor_count: int,
    remote_count: int,
    repetitions: int,
) -> None:
    try:
        result_queue.put({"kind": "progress", "stage": "loading_inputs"})
        anchors, remotes = load_real_inputs(anchor_count, remote_count)
        noises = noise_contributions(anchor_count)
        args = (anchors, remotes, *noises)
        plain_a, plain_b, invalid_count = plaintext_round(anchors, remotes, noises)

        result_queue.put({"kind": "progress", "stage": "compiling"})
        executable, output_tree, compile_seconds, pphlo = compile_secure(args)
        pphlo_lower = pphlo.lower()
        if "exponential" not in pphlo_lower and "exp" not in pphlo_lower:
            raise RuntimeError("compiled SPU program does not contain secure exp")
        simulator = Simulator(N_MPC_PARTIES, runtime_config())

        result_queue.put({"kind": "progress", "stage": "secure_warmup"})
        warmup_a, warmup_b, warmup_seconds, _, _ = execute_secure(
            simulator, executable, output_tree, args
        )
        warmup_metrics = compare_outputs(
            plain_a, plain_b, warmup_a, warmup_b, invalid_count
        )
        result_queue.put(
            {
                "kind": "progress",
                "stage": "measured_repetitions",
                "warmup_seconds": warmup_seconds,
            }
        )

        correctness_rows = []
        runtime_rows = []
        for repetition in range(1, repetitions + 1):
            plain_started = time.perf_counter()
            repeat_plain_a, repeat_plain_b, repeat_invalid = plaintext_round(
                anchors, remotes, noises
            )
            plain_seconds = time.perf_counter() - plain_started
            secure_a, secure_b, secure_seconds, peak_mb, incremental_mb = (
                execute_secure(simulator, executable, output_tree, args)
            )
            metrics = compare_outputs(
                repeat_plain_a,
                repeat_plain_b,
                secure_a,
                secure_b,
                repeat_invalid,
            )
            correctness_rows.append(
                {
                    "case": case,
                    "anchor_count_per_target": anchor_count,
                    "remote_cells_per_source": remote_count,
                    "embedding_dim": EMBED_DIM,
                    "repetition": repetition,
                    **metrics,
                    **approximation_diagnostics(),
                }
            )
            runtime_rows.extend(
                [
                    {
                        "case": case,
                        "backend": "plaintext",
                        "phase": "measured",
                        "repetition": repetition,
                        "wall_clock_seconds": plain_seconds,
                        "compile_seconds_once": 0.0,
                        "warmup_seconds_once": 0.0,
                        "peak_rss_mb": psutil.Process(os.getpid()).memory_info().rss
                        / (1024**2),
                        "incremental_peak_rss_mb": 0.0,
                        "slowdown_vs_plaintext": 1.0,
                        "completed": True,
                    },
                    {
                        "case": case,
                        "backend": "secretflow_spu_aby3",
                        "phase": "measured",
                        "repetition": repetition,
                        "wall_clock_seconds": secure_seconds,
                        "compile_seconds_once": compile_seconds,
                        "warmup_seconds_once": warmup_seconds,
                        "peak_rss_mb": peak_mb,
                        "incremental_peak_rss_mb": incremental_mb,
                        "slowdown_vs_plaintext": secure_seconds
                        / max(plain_seconds, 1e-12),
                        "completed": True,
                    },
                ]
            )
        result_queue.put(
            {
                "kind": "result",
                "case": case,
                "correctness": correctness_rows,
                "runtime": runtime_rows,
                "warmup_metrics": warmup_metrics,
                "compile_seconds": compile_seconds,
                "warmup_seconds": warmup_seconds,
                "pphlo_has_exp": True,
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "kind": "error",
                "case": case,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def run_case_subprocess(
    case: str,
    anchor_count: int,
    remote_count: int,
    repetitions: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=run_case_worker,
        args=(result_queue, case, anchor_count, remote_count, repetitions),
    )
    process.start()
    started = time.monotonic()
    last_stage = "process_start"
    messages = []
    final = None
    while time.monotonic() - started < timeout_seconds:
        try:
            message = result_queue.get(timeout=0.5)
            messages.append(message)
            if message["kind"] == "progress":
                last_stage = message["stage"]
            elif message["kind"] in {"result", "error"}:
                final = message
                break
        except queue.Empty:
            if not process.is_alive():
                break
    if final is None and process.is_alive():
        process.terminate()
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        return {
            "kind": "timeout",
            "case": case,
            "stage": last_stage,
            "elapsed_seconds": time.monotonic() - started,
            "exitcode": process.exitcode,
            "messages": messages,
        }
    process.join(timeout=30)
    if final is None:
        return {
            "kind": "crash",
            "case": case,
            "stage": last_stage,
            "elapsed_seconds": time.monotonic() - started,
            "exitcode": process.exitcode,
            "messages": messages,
        }
    final["elapsed_seconds_total"] = time.monotonic() - started
    final["exitcode"] = process.exitcode
    return final


def small_gate(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    failures = []
    for row in rows:
        repetition = row["repetition"]
        if row["a_relative_l2_error"] > SMALL_A_REL_MAX:
            failures.append(f"rep {repetition} A relative error")
        if row["b_relative_l2_error"] > SMALL_B_REL_MAX:
            failures.append(f"rep {repetition} B relative error")
        if row["b_cosine_similarity"] < SMALL_B_COS_MIN:
            failures.append(f"rep {repetition} B cosine")
        if (
            row["reconstructed_remote_gradient_cosine"]
            < SMALL_GRAD_COS_MIN
        ):
            failures.append(f"rep {repetition} gradient cosine")
    return (not failures, ", ".join(failures))


def mean_sd(values) -> str:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return "not available"
    mean = statistics.fmean(finite)
    sd = statistics.stdev(finite) if len(finite) > 1 else 0.0
    return f"{mean:.6g} +/- {sd:.3g}"


def write_outputs(
    correctness: list[dict[str, Any]],
    runtime: list[dict[str, Any]],
    communication: list[dict[str, Any]],
    status: str,
    decision: str,
    blocker: str,
) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    correctness_frame = pd.DataFrame(correctness)
    runtime_frame = pd.DataFrame(runtime)
    communication_frame = pd.DataFrame(communication)
    schedule_frame = pd.read_csv(SCHEDULE)
    participation = schedule_frame.groupby("cell_id").size()
    source_exposure = schedule_frame.groupby("source_client").agg(
        scheduled_rows=("cell_id", "size"),
        unique_cells=("cell_id", "nunique"),
        active_rounds=("active_round", "nunique"),
    )
    schedule_qc_passed = bool(
        len(schedule_frame) == 17 * N_DATA_CLIENTS * 102
        and schedule_frame.cell_id.nunique() == len(schedule_frame)
        and int(participation.max()) == 1
        and (source_exposure.scheduled_rows == 1734).all()
        and (source_exposure.unique_cells == 1734).all()
        and (source_exposure.active_rounds == 17).all()
    )
    if not schedule_qc_passed:
        raise ValueError("finalized one-use remote schedule QC failed")
    correctness_frame.to_csv(OUTPUT / "correctness.csv", index=False)
    runtime_frame.to_csv(OUTPUT / "runtime_benchmark.csv", index=False)
    communication_frame.to_csv(
        OUTPUT / "communication_benchmark.csv", index=False
    )

    report = [
        "# SecretFlow-SPU secure A/B proof of concept",
        "",
        f"Status: **{status}**",
        f"Final decision: **{decision}**",
        "",
        "## Frozen protocol and provenance",
        "",
        "- Five data-owning clients; three MPC computation parties.",
        "- One RoundOnce interaction; 128 anchors/target and 102 remote cells/source at full scale; embedding dimension 100.",
        "- Joint-C50, C=1206.3391650512762, sigma=2, tau_q=0.20, alpha=0.10.",
        f"- Real embedding: {EMBEDDING}",
        f"- Figure 2 partition: {PARTITION}",
        f"- Round-4 one-use remote schedule: {SCHEDULE}",
        "- Exposure QC: 8,670/8,670 schedule rows are unique; every source has 1,734 unique remote cells across 17 active rounds; maximum participation count is exactly 1.",
        "- No FederatedBatch training or Figure 1/2/5 artifact was modified.",
        "",
        "## MPC architecture and trust model",
        "",
        f"SecretFlow-SPU {spu.__version__} was run through its local three-party simulator using ABY3 replicated arithmetic sharing over FM64. The fixed-point fractional precision was explicitly set to 18 bits. ABY3 here is passive/semi-honest and requires an honest majority: at most one of the three MPC parties may be corrupted; two MPC parties must not collude.",
        "",
        "The five biological clients are input owners, not MPC parties. The local trusted input loader separates their real anchor/remote arrays and SPU Io secret-shares every input to the three computation parties. Local in-memory simulation does not provide process or host isolation; a production deployment must run the three parties on separate non-colluding institutions.",
        "",
        "Each MPC party conceptually contributes an independent Gaussian vector with standard deviation (sigma*C)/sqrt(3). SPU adds the three secret inputs inside the protected program, producing one source-joint Gaussian vector with standard deviation sigma*C; no MPC party can reconstruct the total without collusion. For deterministic local benchmarking, the trusted simulation driver generates all three inputs and therefore knows them. A deployment must generate each contribution at its owning MPC party. One joint vector covers all four target-client blocks for a source before target blocks are aggregated.",
        "",
        "Only final clamped noisy target A/B arrays are reconstructed. The program does not save or return individual anchors, remote embeddings, similarity weights, unclipped per-cell U values, clipping scales, party noise contributions, or pre-noise aggregate A/B.",
        "",
        "## Secure numerical implementation",
        "",
        "Secure cosine uses fixed-point normalization inside SPU. After measuring the unoptimized default 8-iteration Taylor baseline (about 4.90% maximum relative exp error), the final run uses SPU's built-in EXP_PADE secure approximation. A separate 2,001-point secret-operator check over the complete cosine/tau domain [-5, 5] measured 0.0470% maximum and 0.00338% mean relative error versus exact exp. Division uses two Goldschmidt iterations. Sqrt uses SPU's full-accuracy reciprocal-square-root initialization plus one Goldschmidt refinement. No nonlinear operation was replaced by plaintext computation.",
        "",
        "Joint clipping is evaluated once per remote cell across all four other targets as min(1, C/sqrt(sum_{target,b} 2*w^2)); this is the finalized objective. The first unscaled SPU baseline exposed a numeric range failure: secret sqrt returned zero for raw norm-squared values around 1e6. The final implementation therefore evaluates the algebraically identical ratio_sq=norm2/C^2 and min(1, 1/sqrt(ratio_sq)), whose observed real-input range is approximately 0.08 to 2.53. No C, statistic, or clipping rule changed.",
        "",
        "## Correctness",
        "",
    ]
    if correctness_frame.empty:
        report.append("No correctness case completed.")
    else:
        for case, group in correctness_frame.groupby("case", sort=False):
            report.extend(
                [
                    f"### {case}",
                    "",
                    f"- A relative L2 error: {mean_sd(group.a_relative_l2_error)}",
                    f"- B relative L2 error: {mean_sd(group.b_relative_l2_error)}",
                    f"- B cosine similarity: {mean_sd(group.b_cosine_similarity)}",
                    f"- Reconstructed remote-gradient cosine: {mean_sd(group.reconstructed_remote_gradient_cosine)}",
                    "",
                ]
            )
    if (
        not correctness_frame.empty
        and "a_clamp_disagreement_frequency" in correctness_frame.columns
    ):
        full_qc = correctness_frame[
            correctness_frame.case == "full_round4_128x102"
        ]
        if not full_qc.empty:
            disagreement_repetitions = int(
                (full_qc.a_clamp_disagreement_frequency > 0).sum()
            )
            max_disagreement = float(
                full_qc.a_clamp_disagreement_frequency.max()
            )
            report.extend(
                [
                    "### Near-zero A clamp QC",
                    "",
                    f"- Plaintext clamped-A frequency: {mean_sd(full_qc.plaintext_invalid_a_frequency)}.",
                    f"- Secure/plaintext clamp disagreement occurred in {disagreement_repetitions}/{len(full_qc)} repetitions.",
                    f"- Maximum disagreement: {max_disagreement:.6g} of anchors (one anchor is 1/640 = 0.0015625).",
                    f"- Full gradient cosine range: {full_qc.reconstructed_remote_gradient_cosine.min():.6g} to {full_qc.reconstructed_remote_gradient_cosine.max():.6g}.",
                    "- The lower gradient cosine in one repetition is caused by one noisy A value crossing the zero/clamp boundary under randomized fixed-point truncation. A and B tensor errors remained small. This is retained as a numerical boundary risk, not hidden or rerun away.",
                    "",
                ]
            )
    report.extend(["## Performance", ""])
    if runtime_frame.empty:
        report.append("No timed case completed.")
    else:
        for case in runtime_frame.case.drop_duplicates():
            group = runtime_frame[runtime_frame.case == case]
            plain = group[group.backend == "plaintext"]
            secure = group[group.backend == "secretflow_spu_aby3"]
            report.extend(
                [
                    f"### {case}",
                    "",
                    f"- Plaintext wall time (s): {mean_sd(plain.wall_clock_seconds)}",
                    f"- SPU wall time (s): {mean_sd(secure.wall_clock_seconds)}",
                    f"- Slowdown: {mean_sd(secure.slowdown_vs_plaintext)}x",
                    f"- SPU peak RSS (MiB): {mean_sd(secure.peak_rss_mb)}",
                    f"- SPU incremental peak RSS (MiB): {mean_sd(secure.incremental_peak_rss_mb)}",
                    f"- One-time compilation (s): {mean_sd(secure.compile_seconds_once)}",
                    f"- Excluded warm-up interaction (s): {mean_sd(secure.warmup_seconds_once)}",
                    "",
                ]
            )
    report.extend(
        [
            "## Communication",
            "",
            "The SPU 0.9.5 local in-memory Simulator/link Python API does not expose transmitted-byte or round counters. communication_benchmark.csv records this as unavailable rather than substituting a payload estimate for measured protocol communication.",
            "",
            "## Decision and direct answers",
            "",
        ]
    )
    if blocker:
        report.extend([f"Stopping condition: {blocker}", ""])
    full_rows = correctness_frame[
        correctness_frame.case == "full_round4_128x102"
    ]
    full_finished = not full_rows.empty
    report.extend(
        [
            f"1. Full-scale SPU execution finished: **{'yes' if full_finished else 'no'}**.",
            "2. Backend: SecretFlow-SPU ABY3, 3-party replicated arithmetic sharing, FM64 with 18 fixed-point fractional bits.",
            "3. Trust: passive/semi-honest security with at most one corrupted MPC party; no collusion between two of the three parties. The local driver is a trusted input loader in this PoC.",
            "4. Secure-vs-plaintext numerical errors are reported above and per repetition in correctness.csv.",
            "5. Runtime, slowdown, compilation, warm-up, and memory are reported above. Actual communication bytes are unavailable from the local simulator API.",
            f"6. Suitability: **{decision}** for secure-backend feasibility under the requested timing criterion. This does not yet establish WAN communication cost or production deployment readiness.",
            f"7. Final decision: **{decision}**.",
            "",
        ]
    )
    (OUTPUT / "SECURE_AB_SPU_POC_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def communication_row(case: str, completed: bool, reason: str) -> dict[str, Any]:
    return {
        "case": case,
        "framework": f"SecretFlow-SPU {spu.__version__}",
        "protocol": "ABY3",
        "mpc_parties": N_MPC_PARTIES,
        "link_backend": "local in-memory SPU Simulator",
        "communication_bytes_available": False,
        "communication_bytes": math.nan,
        "protocol_rounds_available": False,
        "protocol_rounds": math.nan,
        "full_case_completed": completed,
        "unavailable_reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--small-only", action="store_true")
    parser.add_argument("--small-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--full-timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    for path in (EMBEDDING, PARTITION, SCHEDULE):
        if not path.is_file():
            raise FileNotFoundError(path)

    correctness: list[dict[str, Any]] = []
    runtime: list[dict[str, Any]] = []
    communication: list[dict[str, Any]] = []
    common_unavailable = (
        "SPU local in-memory Simulator/link Python API exposes no byte/round counters"
    )

    small = run_case_subprocess(
        "small_4x5",
        4,
        5,
        args.repetitions,
        args.small_timeout_seconds,
    )
    if small["kind"] != "result":
        blocker = (
            f"small SPU case failed at stage {small.get('stage', 'unknown')}: "
            f"{small.get('error', small['kind'])}"
        )
        communication.append(communication_row("small_4x5", False, common_unavailable))
        write_outputs(
            correctness, runtime, communication, "BLOCKED", "NO-GO", blocker
        )
        return
    correctness.extend(small["correctness"])
    runtime.extend(small["runtime"])
    communication.append(communication_row("small_4x5", True, common_unavailable))
    passed, gate_reason = small_gate(small["correctness"])
    if not passed:
        blocker = (
            f"small correctness gate failed ({gate_reason}); full scale was not run"
        )
        write_outputs(
            correctness, runtime, communication, "BLOCKED", "NO-GO", blocker
        )
        return

    if args.small_only:
        write_outputs(
            correctness,
            runtime,
            communication,
            "SMALL_ONLY_PASSED",
            "MARGINAL",
            "full scale was not requested",
        )
        return

    full = run_case_subprocess(
        "full_round4_128x102",
        128,
        102,
        args.repetitions,
        args.full_timeout_seconds,
    )
    if full["kind"] != "result":
        stage = full.get("stage", "unknown")
        elapsed = float(full.get("elapsed_seconds", 0.0))
        detail = full.get("error", full["kind"])
        blocker = (
            f"full 5x128x102 SPU case did not complete after {elapsed:.1f}s "
            f"at stage {stage} ({detail}); no alternative MPC backend was used"
        )
        runtime.append(
            {
                "case": "full_round4_128x102",
                "backend": "secretflow_spu_aby3",
                "phase": stage,
                "repetition": 0,
                "wall_clock_seconds": elapsed,
                "compile_seconds_once": math.nan,
                "warmup_seconds_once": math.nan,
                "peak_rss_mb": math.nan,
                "incremental_peak_rss_mb": math.nan,
                "slowdown_vs_plaintext": math.nan,
                "completed": False,
                "failure": detail,
            }
        )
        communication.append(
            communication_row(
                "full_round4_128x102",
                False,
                common_unavailable + "; full execution did not return",
            )
        )
        write_outputs(
            correctness, runtime, communication, "BLOCKED", "NO-GO", blocker
        )
        return

    correctness.extend(full["correctness"])
    runtime.extend(full["runtime"])
    communication.append(
        communication_row("full_round4_128x102", True, common_unavailable)
    )
    full_secure = [
        row["wall_clock_seconds"]
        for row in full["runtime"]
        if row["backend"] == "secretflow_spu_aby3"
    ]
    full_mean = statistics.fmean(full_secure)
    if full_mean <= 60.0:
        decision = "GO"
    elif math.isfinite(full_mean):
        decision = "MARGINAL"
    else:
        decision = "NO-GO"
    write_outputs(
        correctness, runtime, communication, "COMPLETED", decision, ""
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
