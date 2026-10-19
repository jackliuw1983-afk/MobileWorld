"""端到端评测自研 micar_app_ui_agent（真机式驱动，仅复用 MobileWorld 的任务与打分）。

每个容器 env 上：
  1. AndroidEnvClient.initialize_task(task_name) → 触发 initialize_task_hook 布置状态
  2. get_task_goal(task_name)                      → 自然语言 goal
  3. adb connect localhost:{adb_port} + 唤醒/常亮   → 暴露容器内模拟器并防熄屏空截图
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


def wake_and_keep_awake(adb_port: int) -> None:
    """唤醒并保持屏幕常亮，避免无头模拟器熄屏后 screencap 输出空截图。"""
    serial = f"localhost:{adb_port}"
    cmds = (
        ["shell", "input", "keyevent", "224"],  # KEYCODE_WAKEUP
        ["shell", "input", "keyevent", "4"],     # KEYCODE_BACK（锁屏时解滑动锁；已解锁时 no-op，避免 MENU 弹桌面菜单）
        ["shell", "settings", "put", "system", "screen_off_timeout", "2147483647"],
        ["shell", "svc", "power", "stayon", "true"],
    )
    for args in cmds:
        subprocess.run(
            ["adb", "-s", serial] + args, capture_output=True, text=True, check=False
        )
    time.sleep(1)  # 等屏幕真正亮起，再交给 agent 截图


def run_one_task(target: tuple[str, int], task_name: str, args: argparse.Namespace) -> dict:
    backend_url, adb_port = target
    agent_main = os.path.join(args.agent_dir, "main.py")
    start = time.time()
    env: AndroidEnvClient | None = None
    try:
        env = AndroidEnvClient(backend_url, device="emulator-5554", step_wait_time=1.0)
        env.switch_suite_family("mobile_world")
        env.initialize_task(task_name)
        goal = env.get_task_goal(task_name)

        adb = subprocess.run(["adb", "connect", f"localhost:{adb_port}"], capture_output=True, text=True, check=False)
        if adb.returncode != 0:
            print(f"[adb] connect failed rc={adb.returncode}: {adb.stderr.strip()}", file=sys.stderr)
        wake_and_keep_awake(adb_port)

        agent_argv = [
            "--mw-goal", goal,
            "--device", f"localhost:{adb_port}",
            "--platform", "android",
            "--model", args.model,
            "--max-turns", str(args.max_turns),
            "--cache", "false",
        ]
        if args.agent_bin and os.path.isfile(args.agent_bin):
            cmd = [args.agent_bin] + agent_argv
            cwd: str | None = None  # 二进制按自身路径定位 _internal，无需指定 cwd
        else:
            cmd = [args.agent_python, agent_main] + agent_argv
            cwd = args.agent_dir
        agent_exit: int | str | None
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
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
    except Exception as e:  # noqa: BLE001
        print(f"[harness_error] {task_name}: {e}", file=sys.stderr)
        return {
            "task": task_name,
            "goal": "",
            "adb_port": adb_port,
            "agent_exit": None,
            "score": 0.0,
            "success": False,
            "reason": f"harness_error: {e}",
            "duration_seconds": round(time.time() - start, 1),
        }
    finally:
        if env is not None:
            try:
                env.tear_down_task(task_name)
            except Exception:  # noqa: BLE001
                pass


def run_target_group(
    target: tuple[str, int], task_names: list[str], args: argparse.Namespace
) -> list[dict]:
    """串行跑同一容器上的全部任务（同一 env 不可并发），返回结果行列表。

    不写 jsonlines —— 写文件只能发生在主线程。
    """
    return [run_one_task(target, name, args) for name in task_names]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="任务名（class name），逗号分隔多个；或用 env.get_suite_task_list 前先跑单题")
    ap.add_argument("--log-dir", default="traj_logs/micar_e2e", help="结果 JSONL 输出目录")
    ap.add_argument("--model", default="xiaomi/mimo-v2.6-flash", help="规划模型")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--agent-timeout", type=int, default=900, help="单个 agent 子进程超时（秒）")
    ap.add_argument("--agent-dir", default=AGENT_DIR, help="agent 代码目录（含 main.py）")
    ap.add_argument(
        "--agent-python",
        default="python",
        help="agent 侧 python 解释器，须为已安装 agent 依赖的解释器（勿用 MobileWorld venv）",
    )
    ap.add_argument(
        "--agent-bin",
        default=None,
        help="agent 打包二进制路径（默认取 <agent-dir>/dist/micar-agent-macos-arm64/micar-agent）；存在则优先于 --agent-python 执行",
    )
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()

    if not args.agent_bin:
        args.agent_bin = os.path.join(args.agent_dir, "dist", "micar-agent-macos-arm64", "micar-agent")

    targets = discover_targets()
    if not targets:
        print("未发现运行中的 MobileWorld 容器，请先 `sudo uv run mw env run --count N`", file=sys.stderr)
        return 1
    print(f"发现 {len(targets)} 个容器: {targets}")

    if args.task.strip().upper() == "ALL":
        # 用第一个容器的 client 拉 GUI-only 任务列表（默认排除 mcp / user-interaction）
        first_url, _ = targets[0]
        probe = AndroidEnvClient(first_url, device="emulator-5554", step_wait_time=1.0)
        probe.switch_suite_family("mobile_world")
        task_names = probe.get_suite_task_list(enable_mcp=False, enable_user_interaction=False)
        print(f"GUI-only 任务共 {len(task_names)} 题")
    else:
        task_names = [t.strip() for t in args.task.split(",") if t.strip()]
    task_groups: dict[tuple[str, int], list[str]] = {}
    for i, name in enumerate(task_names):
        task_groups.setdefault(targets[i % len(targets)], []).append(name)
    if not task_groups:
        print("未解析到任何任务名（--task 为空或仅逗号）", file=sys.stderr)
        return 1

    os.makedirs(args.log_dir, exist_ok=True)
    out_path = f"{args.log_dir}/results.jsonl"

    # 按目标分组：同一容器内任务串行（一个 worker），跨容器并行；
    # worker 只返回结果行，jsonlines 写入仅发生在主线程。
    results: list[dict] = []
    with jsonlines.open(out_path, mode="w") as writer:
        max_workers = min(args.concurrency, len(task_groups))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(run_target_group, tgt, names, args) for tgt, names in task_groups.items()]
            for fut in as_completed(futs):
                for r in fut.result():
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
