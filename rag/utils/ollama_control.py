#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Helpers for pausing / resuming Ollama models during heavy OCR work."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterable, List

import requests


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11435")
MODELS_TO_PAUSE = [m.strip() for m in os.getenv("OLLAMA_PAUSE_MODELS", "").split(",") if m.strip()]
WARMUP_PROMPT = os.getenv("OLLAMA_WARMUP_PROMPT", "")
REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "5"))


def _stop_model(name: str) -> bool:
    if not name:
        return False
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/stop",
            json={"name": name},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.ok:
            logging.info("Ollama model '%s' stopped to free GPU memory", name)
            return True
        logging.warning("Failed to stop Ollama model '%s': %s", name, resp.text)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Exception stopping Ollama model '%s': %s", name, exc)
    return False


def _warmup_model(name: str) -> None:
    if not name:
        return
    prompt = WARMUP_PROMPT or " "
    payload = {
        "model": name,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=max(REQUEST_TIMEOUT, 10),
        )
        if resp.ok:
            logging.info("Ollama model '%s' warmed up after chunking", name)
        else:
            logging.warning("Failed to warm up Ollama model '%s': %s", name, resp.text)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Exception warming up Ollama model '%s': %s", name, exc)


@contextmanager
def pause_ollama_models(models: Iterable[str] | None = None):
    """Temporarily stop a set of Ollama models to release GPU memory."""

    if models is None:
        models = MODELS_TO_PAUSE
    models = [m for m in (models or []) if m]

    if not models or not OLLAMA_BASE_URL:
        yield
        return

    stopped: List[str] = []
    for model in models:
        if _stop_model(model):
            stopped.append(model)

    try:
        yield
    finally:
        for model in stopped:
            _warmup_model(model)


__all__ = ["pause_ollama_models"]
