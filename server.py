"""
LiteRT-LM OpenAI-Compatible API Server

Serves a .litertlm model behind standard OpenAI API endpoints with
comprehensive per-request inference statistics logged to stdout.

Usage:
  python server.py -hf <model> [options]

Endpoints:
  GET  /v1/models              -> list available model(s)
  POST /v1/chat/completions    -> streaming & non-streaming chat
"""

import argparse
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import litert_lm
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from huggingface_hub import hf_hub_download, list_repo_files
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("litert-lm-server")

# ---------------------------------------------------------------------------
# Configuration (set via argparse before engine starts)
# ---------------------------------------------------------------------------

CONFIG = {
    "hf_model": "model.litertlm",
    "model_file": None,
    "backend": litert_lm.Backend.CPU,
    "backend_str": "cpu",
    "host": "0.0.0.0",
    "port": 8000,
    "speculative": False,
    "context_length": 4096,
    "cache_dir": None,
}

# ---------------------------------------------------------------------------
# Pydantic schemas (OpenAI-compatible wire format)
# ---------------------------------------------------------------------------


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None

    model_config = {"extra": "allow"}


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------

engine: Optional[litert_lm.Engine] = None
model_id: str = "unknown"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_model_path(hf_repo: str, model_file: Optional[str] = None) -> str:
    """Return a local path to the .litertlm file.

    If *hf_repo* is already a local file it is returned as-is.
    Otherwise it is treated as a HuggingFace repo ID, optionally pinned
    to a specific *model_file* inside the repo (auto-detected otherwise).
    """
    p = Path(hf_repo)
    if p.exists():
        return str(p.resolve())

    if model_file:
        filename = model_file
    else:
        log.info("Listing files in HF repo:  %s", hf_repo)
        files = [f for f in list_repo_files(hf_repo) if f.endswith(".litertlm")]
        if not files:
            raise RuntimeError(f"No .litertlm file found in {hf_repo}")
        # Pick the one without qualcomm/extra qualifiers as default
        plain = [f for f in files if "/" not in f.replace("\\", "/")]
        filename = (plain or files)[0]
        log.info("Auto-detected model file:  %s", filename)

    log.info("Downloading from HF:  %s / %s", hf_repo, filename)
    return hf_hub_download(repo_id=hf_repo, filename=filename)


def _to_litert_msgs(messages: List[ChatMessage]) -> List[Dict]:
    result = []
    for m in messages:
        if isinstance(m.content, str):
            content = [{"type": "text", "text": m.content}]
        else:
            content = [p.model_dump() for p in m.content]
        result.append({"role": m.role, "content": content})
    return result


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars / token for English text)."""
    return max(1, len(text) // 4)


def _sse_chunk(
    completion_id: str,
    delta_content: str = "",
    finish: bool = False,
) -> str:
    data = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": (
                    {}
                    if finish
                    else ({"role": "assistant"} if delta_content == "__role__" else {"content": delta_content})
                ),
                "finish_reason": ("stop" if finish else None),
            }
        ],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stats_line(**kw: Any) -> str:
    parts = [f"[model={v}]" if k == "model" else f"{k}={v}" for k, v in kw.items()]
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# FastAPI app (lifespan)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, model_id
    cfg = CONFIG
    log.info(
        "Engine config:  model=%s  backend=%s  speculative=%s  context=%s",
        cfg["hf_model"], cfg["backend_str"], cfg["speculative"], cfg["context_length"],
    )
    if cfg["model_file"]:
        log.info("Model file:  %s", cfg["model_file"])
    if cfg["cache_dir"]:
        log.info("Cache dir: %s", cfg["cache_dir"])

    model_path = _resolve_model_path(cfg["hf_model"], cfg.get("model_file"))
    engine = litert_lm.Engine(
        model_path,
        backend=cfg["backend"],
        max_num_tokens=cfg["context_length"],
        enable_speculative_decoding=cfg["speculative"],
        cache_dir=cfg["cache_dir"] or "",
    )

    model_id = Path(model_path).stem
    log.info("Model loaded:  %s", model_id)

    yield

    if engine:
        del engine
        engine = None
        log.info("Engine released")


app = FastAPI(title="LiteRT-LM Server", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models() -> Dict:
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "litert-lm",
            }
        ],
    }


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest) -> Any:
    if engine is None:
        raise HTTPException(503, "Engine not initialised")

    msgs = _to_litert_msgs(body.messages)
    if not msgs:
        raise HTTPException(400, "messages must not be empty")

    # last message is the new user prompt; everything before is history
    prompt = msgs.pop()

    # estimate input tokens from all textual content
    input_tokens = _estimate_tokens(" ".join(m["content"][0]["text"] for m in [*msgs, prompt]))

    if body.stream:
        return StreamingResponse(
            _stream_completion(msgs, prompt, input_tokens),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return _non_stream_completion(msgs, prompt, input_tokens)


# ---------------------------------------------------------------------------
# Non-streaming handler
# ---------------------------------------------------------------------------


def _non_stream_completion(history: List[Dict], prompt: Dict, input_tokens: int) -> Dict:
    with engine.create_conversation(messages=history) as conversation:
        t0 = time.perf_counter()
        try:
            response = conversation.send_message(prompt)
        except RuntimeError as exc:
            msg = str(exc)
            log.warning("Inference error (non-stream): %s", msg)
            raise HTTPException(400, detail=msg) from exc
        elapsed = time.perf_counter() - t0

    text = "".join(item["text"] for item in response.get("content", []) if item.get("type") == "text")
    output_tokens = _estimate_tokens(text)

    out_speed = output_tokens / elapsed if elapsed > 0 else 0.0

    log.info(
        "INFERENCE  %s",
        _stats_line(
            model=model_id,
            mode="non-stream",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_time_ms=round(elapsed * 1000),
            output_speed=f"{out_speed:.1f} tok/s",
        ),
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Streaming handler
# ---------------------------------------------------------------------------


def _stream_completion(history: List[Dict], prompt: Dict, input_tokens: int) -> Iterator[str]:
    with engine.create_conversation(messages=history) as conversation:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"

        yield _sse_chunk(completion_id, delta_content="__role__")

        t0 = time.perf_counter()
        ttft: Optional[float] = None
        chunk_count = 0
        full_text = ""

        try:
            for chunk in conversation.send_message_async(prompt):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                    prefill_speed = input_tokens / ttft if ttft > 0 else 0.0
                    log.info(
                        "PREFILL    %s",
                        _stats_line(
                            model=model_id,
                            ttft_ms=round(ttft * 1000),
                            prefill_speed=f"{prefill_speed:.1f} tok/s",
                            input_tokens=input_tokens,
                        ),
                    )

                for item in chunk.get("content", []):
                    if item.get("type") == "text":
                        t = item["text"]
                        full_text += t
                        chunk_count += 1
                        yield _sse_chunk(completion_id, delta_content=t)
        except RuntimeError as exc:
            msg = str(exc)
            log.warning("Inference error (stream): %s", msg)
            yield f"data: {json.dumps({'error': msg})}\n\n"
            yield _sse_chunk(completion_id, finish=True)
            yield "data: [DONE]\n\n"
            return

        decode_time = time.perf_counter() - t0 - (ttft or 0)
        total_time = time.perf_counter() - t0
        output_tokens = _estimate_tokens(full_text)

        decode_speed = output_tokens / decode_time if decode_time > 0 else 0.0

        log.info(
            "INFERENCE  %s",
            _stats_line(
                model=model_id,
                mode="stream",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                chunks=chunk_count,
                ttft_ms=round((ttft or 0) * 1000),
                decode_time_ms=round(decode_time * 1000),
                total_time_ms=round(total_time * 1000),
                prefill_speed=f"{prefill_speed:.1f} tok/s" if ttft else "N/A",
                decode_speed=f"{decode_speed:.1f} tok/s",
            ),
        )

        yield _sse_chunk(completion_id, finish=True)
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LiteRT-LM OpenAI-Compatible API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python server.py -hf litert-community/gemma-4-E2B-it-litert-lm --gpu --speculative\n"
            "  python server.py -hf /path/to/model.litertlm --context 4096 --port 8080\n"
        ),
    )
    parser.add_argument(
        "-hf", "--hf-model",
        required=True,
        help="HuggingFace model ID (e.g. litert-community/gemma-4-E2B-it-litert-lm) or local .litertlm path",
    )
    parser.add_argument(
        "-c", "--context", type=int, default=4096,
        help="Context length in tokens (default: 4096)",
    )
    parser.add_argument(
        "--speculative", action="store_true",
        help="Enable MTP speculative decoding (GPU recommended)",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use GPU backend (default: CPU)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Bind port (default: 8000)",
    )
    parser.add_argument(
        "--model-file",
        help="Filename inside the HF repo (auto-detected if not given)",
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory for compiled-artifact cache",
    )
    args = parser.parse_args()

    CONFIG["hf_model"] = args.hf_model
    CONFIG["model_file"] = args.model_file
    CONFIG["backend"] = litert_lm.Backend.GPU if args.gpu else litert_lm.Backend.CPU
    CONFIG["backend_str"] = "gpu" if args.gpu else "cpu"
    CONFIG["host"] = args.host
    CONFIG["port"] = args.port
    CONFIG["speculative"] = args.speculative
    CONFIG["context_length"] = args.context
    CONFIG["cache_dir"] = args.cache_dir or None

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
