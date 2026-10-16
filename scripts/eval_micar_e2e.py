"""端到端评测自研 micar_app_ui_agent（真机式驱动，仅复用 MobileWorld 的任务与打分）。

每个容器 env 上：
  1. AndroidEnvClient.initialize_task(task_name) → 触发 initialize_task_hook 布置状态
  2. get_task_goal(task_name)                      → 自然语言 goal
  3. adb connect localhost:{adb_port}              → 暴露容器内模拟器
  4. 子进程启动真实 agent：main.py --mw-goal "<goal>" --device localhost:{adb_port}
  5. agent 退出后 get_task_score(task_name)        → MobileWorld 唯一打分来源
  6. 结果写 results.jsonl

不经过 BaseAgent / JSONAction / core.runner 的 predict 循环。
API key 一律通过环境变量传入（ANTHROPIC_API_KEY / MIDSCENE_MODEL_API_KEY），不落盘。

用法示例（先 sudo uv run mw env run --count 1 起一个容器）：
    uv run python scripts/eval_micar_e2e.py --task OpenFlightModeTask --log-dir traj_logs/micar_e2e
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import jsonlines

from mobile_world.runtime.client import AndroidEnvClient
from mobile_world.runtime.utils.docker import docker_inspect, list_containers_by_image_substring
from mobile_world.runtime.utils.e2e import build_agent_env, extract_host_port, score_success

AGENT_DIR = "/Users/mi/git/micartest-mobile/micar_app_ui_agent"
AGENT_MAIN = f"{AGENT_DIR}/main.py"


def discover_targets(image_filter: str = "mobile_world", prefix: str = "mobile_world_env") -> list[tuple[str, int]]:
    """返回 [(backend_url, adb_port)]，每个运行中的容器一个。"""
    containers = list_containers_by_image_substring(image_filter, include_all=False)
    targets: list[tuple[str, int]] = []
    for c in containers:
        name = c.get("Names", "")
        if not name or not name.startswith(prefix):
            continue
        info = docker_inspect(name)
        if not info:
            continue
        ports = info.get("NetworkSettings", {}).get("Ports", {})
        backend = extract_host_port(ports, "6800")
        adb = extract_host_port(ports, "5556")
        if backend and adb:
            targets.append((f"http://localhost:{backend}", int(adb)))
    return targets


def run_one_task(target: tuple[str, int], task_name: str, args: argparse.Namespace) -> dict:
    backend_url, adb_port = target
    env = AndroidEnvClient(backend_url, device="emulator-5554", step_wait_time=1.0)
    env.switch_suite_family("mobile_world")
    start = time.time()
    try:
        env.initialize_task(task_name)
        goal = env.get_task_goal(task_name)

        adb = subprocess.run(["adb", "connect", f"localhost:{adb_port}"], capture_output=True, text=True, check=False)
        if adb.returncode != 0:
            print(f"[adb] connect failed rc={adb.returncode}: {adb.stderr.strip()}", file=sys.stderr)

        cmd = [
            args.agent_python, AGENT_MAIN,
            "--mw-goal", goal,
            "--device", f"localhost:{adb_port}",
            "--platform", "android",
            "--model", args.model,
            "--max-turns", str(args.max_turns),
            "--cache", "false",
        ]
        agent_exit: int | str
        try:
            proc = subprocess.run(
                cmd,
                cwd=AGENT_DIR,
                env=build_agent_env(adb_port),
                timeout=args.agent_timeout,
                check=False,
            )
            agent_exit = proc.returncode
        except subprocess.TimeoutExpired:
            print(f"[timeout] agent exceeded {args.agent_timeout}s on {task_name}", file=sys.stderr)
            agent_exit = "timeout"
        score, reason = env.get_task_score(task_name)
        return {
            "task": task_name,
            "goal": goal,
            "adb_port": adb_port,
            "agent_exit": agent_exit,
            "score": score,
            "success": score_success(score),
            "reason": reason,
            "duration_seconds": round(time.time() - start, 1),
        }
    finally:
        try:
            env.tear_down_task(task_name)
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="任务名（class name），逗号分隔多个；或用 env.get_suite_task_list 前先跑单题")
    ap.add_argument("--log-dir", default="traj_logs/micar_e2e", help="结果 JSONL 输出目录")
    ap.add_argument("--model", default="xiaomi/mimo-v2.6-flash", help="规划模型")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--agent-timeout", type=int, default=900, help="单个 agent 子进程超时（秒）")
    ap.add_argument("--agent-python", default="python", help="agent 侧 python 解释器")
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()

    targets = discover_targets()
    if not targets:
        print("未发现运行中的 MobileWorld 容器，请先 `sudo uv run mw env run --count N`", file=sys.stderr)
        return 1
    print(f"发现 {len(targets)} 个容器: {targets}")

    task_names = [t.strip() for t in args.task.split(",") if t.strip()]
    pairs = [(targets[i % len(targets)], t) for i, t in enumerate(task_names)]
    if not pairs:
        print("未解析到任何任务名（--task 为空或仅逗号）", file=sys.stderr)
        return 1

    os.makedirs(args.log_dir, exist_ok=True)
    out_path = f"{args.log_dir}/results.jsonl"

    results: list[dict] = []
    with jsonlines.open(out_path, mode="a") as writer:
        if len(pairs) == 1:
            results.append(run_one_task(pairs[0][0], pairs[0][1], args))
            writer.write(results[-1])
        else:
            with ThreadPoolExecutor(max_workers=min(args.concurrency, len(pairs))) as ex:
                futs = [ex.submit(run_one_task, tgt, name, args) for tgt, name in pairs]
                for fut in as_completed(futs):
                    r = fut.result()
                    results.append(r)
                    writer.write(r)

    ok = sum(1 for r in results if r["success"])
    print(f"\n== 完成 {len(results)} 题，成功 {ok}，失败 {len(results) - ok} ==")
    for r in results:
        mark = "✓" if r["success"] else "✗"
        print(f"  {mark} {r['task']}  score={r['score']}  exit={r['agent_exit']}  {r['reason'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
