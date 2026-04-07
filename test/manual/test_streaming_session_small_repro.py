"""Manual reproducer for streaming-session KV accounting failures.

Runs a small ungated model with streaming sessions enabled and issues mixed
streaming/non-streaming chunked-prefill traffic in concurrent batches.

Example:
    uv run python test/manual/test_streaming_session_small_repro.py \
        --model-path Qwen/Qwen2.5-1.5B-Instruct \
        --rounds 12 \
        --sessions 4 \
        --turns 6 \
        --mem-fraction-static 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server


FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "Sphinx of black quartz, judge my vow. "
    "The five boxing wizards jump quickly. "
    "Jackdaws love my big sphinx of quartz. "
    "A wizard's job is to vex chumps quickly in fog. "
    "We promptly judged antique ivory buckles for the next prize. "
) * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.getenv("SGLANG_REPRO_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
    )
    parser.add_argument("--base-url", default=DEFAULT_URL_FOR_TEST)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--non-streaming-per-turn", type=int, default=2)
    parser.add_argument("--gen-len", type=int, default=16)
    parser.add_argument("--chunked-prefill-size", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--skip-server-launch", action="store_true")
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--offset-stride", type=int, default=5000)
    return parser.parse_args()


def tail_text(path: Path, num_lines: int = 80) -> str:
    if not path.exists():
        return f"<missing {path}>"
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-num_lines:])


async def generate(
    base_url: str,
    session: aiohttp.ClientSession,
    input_ids: list[int],
    max_new_tokens: int,
    session_params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload: Dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "no_stop_trim": True,
            "skip_special_tokens": False,
        },
    }
    if session_params:
        payload["session_params"] = session_params

    async with session.post(base_url + "/generate", json=payload) as resp:
        text = await resp.text()
        assert resp.status == 200, f"generate failed: {resp.status} {text}"
        return await resp.json()


async def run_rounds(args: argparse.Namespace, tokenizer: Any) -> None:
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        session_ids = []
        for _ in range(args.sessions):
            async with http.post(
                args.base_url + "/open_session",
                json={"capacity_of_str_len": 50000, "streaming": True},
            ) as resp:
                assert resp.status == 200, await resp.text()
                session_ids.append(await resp.json())

        for round_idx in range(args.rounds):
            for turn in range(args.turns):
                tasks = []
                for session_idx, session_id in enumerate(session_ids):
                    offset = (
                        args.client_id * args.offset_stride
                        + (round_idx * args.sessions + session_idx) * 250
                        + turn * 200
                    )
                    text = (
                        f"[client={args.client_id} round={round_idx} session={session_idx} turn={turn}] "
                        f"{FILLER[offset:offset + 1800]}"
                    )
                    input_ids = tokenizer.encode(text)
                    tasks.append(
                        generate(
                            args.base_url,
                            http,
                            input_ids,
                            args.gen_len,
                            session_params={"id": session_id, "rid": None},
                        )
                    )

                for ns_idx in range(args.non_streaming_per_turn):
                    offset = (
                        args.client_id * args.offset_stride
                        + round_idx * 97
                        + ns_idx * 113
                        + turn * 53
                    ) % max(1, len(FILLER) - 900)
                    text = (
                        f"[client={args.client_id} round={round_idx} ns={ns_idx} turn={turn}] "
                        f"{FILLER[offset:offset + 900]}"
                    )
                    input_ids = tokenizer.encode(text)
                    tasks.append(
                        generate(
                            args.base_url,
                            http,
                            input_ids,
                            args.gen_len,
                        )
                    )

                await asyncio.gather(*tasks)

                health = requests.get(args.base_url + "/health", timeout=10)
                if health.status_code != 200:
                    raise RuntimeError(
                        f"server unhealthy after round={round_idx} turn={turn}: "
                        f"{health.status_code} {health.text}"
                    )

        for session_id in session_ids:
            async with http.post(
                args.base_url + "/close_session", json={"session_id": session_id}
            ) as resp:
                assert resp.status == 200, await resp.text()


def main() -> int:
    args = parse_args()
    stdout_log = None
    stderr_log = None
    process = None
    start = time.time()

    try:
        if not args.skip_server_launch:
            stdout_log = tempfile.NamedTemporaryFile(
                prefix="streaming-session-repro.",
                suffix=".stdout.log",
                delete=False,
                mode="w+",
                encoding="utf-8",
            )
            stderr_log = tempfile.NamedTemporaryFile(
                prefix="streaming-session-repro.",
                suffix=".stderr.log",
                delete=False,
                mode="w+",
                encoding="utf-8",
            )
            process = popen_launch_server(
                args.model_path,
                args.base_url,
                timeout=args.startup_timeout,
                device=args.device,
                return_stdout_stderr=(stdout_log, stderr_log),
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    str(args.chunked_prefill_size),
                    "--page-size",
                    str(args.page_size),
                    "--mem-fraction-static",
                    str(args.mem_fraction_static),
                    "--max-running-requests",
                    str(args.max_running_requests),
                    "--tp-size",
                    str(args.tp_size),
                    "--log-level",
                    "info",
                ],
            )

        tokenizer = get_tokenizer(args.model_path)
        asyncio.run(run_rounds(args, tokenizer))
        duration = time.time() - start
        print(f"completed successfully in {duration:.1f}s")
        if stdout_log is not None and stderr_log is not None:
            print(f"stdout log: {stdout_log.name}")
            print(f"stderr log: {stderr_log.name}")
        return 0
    except Exception as exc:
        print(f"reproducer failed: {exc}", file=sys.stderr)
        if stdout_log is not None and stderr_log is not None:
            print(f"stdout log: {stdout_log.name}", file=sys.stderr)
            print(f"stderr log: {stderr_log.name}", file=sys.stderr)
            print("---- stderr tail ----", file=sys.stderr)
            print(tail_text(Path(stderr_log.name)), file=sys.stderr)
        return 1
    finally:
        if stdout_log is not None:
            stdout_log.close()
        if stderr_log is not None:
            stderr_log.close()
        if process is not None:
            try:
                kill_process_tree(process.pid)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
