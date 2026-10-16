"""MobileWorld 端到端评测 harness 的纯工具函数。"""
from __future__ import annotations

import os

DEFAULT_ANTHROPIC_BASE_URL = "http://model.mify.ai.srv/anthropic"
DEFAULT_MIDSCENE_BASE_URL = "http://model.mify.ai.srv/v1"
DEFAULT_MODEL = "xiaomi/mimo-v2.6-flash"


def score_success(score: float) -> bool:
    """MobileWorld 打分为 0.0/1.0；容差 0.999 判定成功。"""
    return score >= 0.999


def extract_host_port(ports: dict, container_port: str) -> str | None:
    """从 docker inspect 的 Ports dict 提取容器端口对应的首个 HostPort。

    ports 形如 {"6800/tcp": [{"HostPort": "6800"}], ...}；container_port 如 "6800"。
    """
    for key, host_bindings in ports.items():
        if key.split("/", 1)[0] == container_port and host_bindings:
            host_port = host_bindings[0].get("HostPort", "")
            if host_port:
                return host_port
    return None


def build_agent_env(adb_port: int, overrides: dict[str, str] | None = None) -> dict[str, str]:
    """构造 agent 子进程环境：继承当前环境 + 网关配置 + ANDROID_SERIAL。

    API key 一律来自当前进程环境（overrides 传入），不写死。
    """
    env = os.environ.copy()
    env.setdefault("ANTHROPIC_BASE_URL", DEFAULT_ANTHROPIC_BASE_URL)
    env.setdefault("MIDSCENE_MODEL_BASE_URL", DEFAULT_MIDSCENE_BASE_URL)
    env.setdefault("MIDSCENE_MODEL_NAME", DEFAULT_MODEL)
    env.setdefault("MIDSCENE_MODEL_FAMILY", "xiaomi-mimo")
    env["ANDROID_SERIAL"] = f"localhost:{adb_port}"
    if overrides:
        env.update(overrides)
    return env
