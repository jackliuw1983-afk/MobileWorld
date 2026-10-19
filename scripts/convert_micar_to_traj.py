#!/usr/bin/env python3
"""把自研 micar agent 的执行产物转成官方 MobileWorld 的 traj.json（+ 可选 screenshots）。

micar 的评测 harness（scripts/eval_micar_e2e.py）绕过 core/runner.py 与 TrajLogger，
每个任务只产出：
  - <run>/<Task>/result.txt                    （score: / reason:，已是官方格式）
  - <run>/agent_logs/<Task>-<port>.log         （每轮 Agent / Thinking / Tool 调用）
  - <run>/midscene_reports/<Task>-<port>/<...>.html （midscene 报告：动作坐标 + 截图）
  - <run>/results.jsonl(.XXX)                  （goal / score / adb_port，聚合）

官方 bundle_trajs.py 需要每个任务目录里有 traj.json（{"0":{"traj":[step],...}}），
本脚本把 agent_log 的一轮一次「动作类工具调用」映射成一个官方 step，写入
<run>/<Task>/traj.json；--with-screenshots 时从 midscene HTML 里抽取 base64 截图
写到 <run>/<Task>/screenshots/<Task>-0-<step>.png，供 bundle_trajs.py 合成 mp4。

用法（在 MobileWorld 仓库内、有数据的那台机器上）：
    uv run python scripts/convert_micar_to_traj.py traj_logs/micar_e2e_1018 --with-screenshots
    uv run python site/bundle_trajs.py traj_logs/micar_e2e_1018 \
        -o site/trajs/micar.json.gz --with-screenshots \
        --video-base-url https://tongyi-mai.github.io/MAI-UI-blog/MobileWorld/trajs

产物写在原 run 目录内（原地补 traj.json / screenshots），可重复运行；已有 traj.json
的任务默认跳过，除非加 --force。支持 --out-dir 把补齐后的目录结构镜像到新位置。
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import re
import sys
from typing import Any

# ── 工具名 → 官方 action_type ──────────────────────────────────────────────
TOOL_ACTION_TYPE: dict[str, str] = {
    "tap": "click",
    "doubul_click": "double_tap",
    "long_press": "long_press",
    "input_text": "input_text",
    "scroll": "scroll",
    "swipe": "swipe",
    "launch_app": "open_app",
    "tap_by_xpath": "click",
    "input_by_xpath": "input_text",
    "bring_app_to_front": "open_app",
    "answer": "answer",
    "finish": "finished",
}

# 被动/观察工具：不生成 step（等价于官方循环里每次 predict 前的自动截图/等待）
PASSIVE_TOOLS = {"wait", "take_screenshot", "read_file", "dump_ui", "pull_file"}

# 从 agent_log 提取每轮信息的正则（log 行形如 "13:19:27 [INFO] loop_kernel:   Tool: tap({...})"）
RE_TOOL = re.compile(r"\[INFO\]\s+loop_kernel:\s+Tool:\s+(\w+)\((.*)\)\s*$")
RE_AGENT = re.compile(r"\[INFO\]\s+loop_kernel:\s+Agent:\s*(.*)$")
RE_THINKING = re.compile(r"\[INFO\]\s+loop_kernel:\s+Thinking:\s*(.*)$")
RE_TOKEN = re.compile(r"\bin=(\d+),\s*out=(\d+)\s+tokens")
RE_GOAL = re.compile(r"请完成以下目标：\\n(.+?)\\?\"", re.S)


def load_results(run_dir: str) -> dict[str, dict]:
    """从 results.jsonl（或最大的一条 results.jsonl.* 备份）读 goal/score/adb_port。

    返回 {task_name: {"goal": str, "score": float, "adb_port": int, ...}}；读不到则空。
    """
    candidates = [os.path.join(run_dir, "results.jsonl")]
    candidates += sorted(
        glob.glob(os.path.join(run_dir, "results.jsonl.*")),
        key=lambda p: os.path.getsize(p),
        reverse=True,
    )
    best: dict[str, dict] = {}
    for path in candidates:
        rows: list[dict] = []
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if len(rows) > len(best):
            best = {r["task"]: r for r in rows if isinstance(r, dict) and r.get("task")}
    return best


def parse_agent_log(log_path: str) -> tuple[str | None, list[dict]]:
    """解析 agent log，返回 (goal, turns)。

    turns 每个元素 = {"tool": str, "args": dict, "agent": str, "thinking": str,
                      "prompt_tokens": int, "completion_tokens": int}
    """
    goal: str | None = None
    turns: list[dict] = []
    cur: dict[str, Any] | None = None
    prompt_tokens = 0
    completion_tokens = 0

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if goal is None:
                m = RE_GOAL.search(line)
                if m:
                    goal = m.group(1).replace("\\n", "\n").replace("\\'", "'")

            mt = RE_TOKEN.search(line)
            if mt:
                prompt_tokens += int(mt.group(1))
                completion_tokens += int(mt.group(2))

            if "Tool:" in line and "[INFO] loop_kernel:" in line:
                m = RE_TOOL.search(line)
                if m:
                    if cur is not None:
                        turns.append(cur)
                    name, args_raw = m.group(1), m.group(2)
                    args: dict = {}
                    try:
                        parsed = json.loads(args_raw)
                        if isinstance(parsed, dict):
                            args = parsed
                    except json.JSONDecodeError:
                        args = {"_raw": args_raw}
                    cur = {"tool": name, "args": args, "agent": "", "thinking": ""}
                    continue

            if cur is None:
                continue
            ma = RE_AGENT.search(line)
            if ma:
                cur["agent"] += (ma.group(1) + "\n") if cur["agent"] else ma.group(1)
                continue
            mth = RE_THINKING.search(line)
            if mth:
                cur["thinking"] += (mth.group(1) + "\n") if cur["thinking"] else mth.group(1)

    if cur is not None:
        turns.append(cur)

    for t in turns:
        t["prompt_tokens"] = prompt_tokens
        t["completion_tokens"] = completion_tokens
    return goal, turns


def parse_midscene_report(html_path: str) -> tuple[dict | None, dict[str, bytes]]:
    """解析 midscene HTML，返回 (report_json, {screenshot_id: image_bytes})。

    report_json 顶层含 executions[] → tasks[]，其中 type=="Action Space" 的任务
    对应一次真实 UI 动作，param.locate.center 是点击坐标，recorder[].screenshot.id
    是动作后的截图 id；截图本体在 <script type="midscene-image" data-id=…> 里
    base64 内联。
    """
    if not html_path or not os.path.isfile(html_path):
        return None, {}
    with open(html_path, encoding="utf-8", errors="replace") as f:
        html = f.read()

    report: dict | None = None
    m = re.search(
        r'<script type="midscene_web_dump"[^>]*data-screenshot-mode="inline"[^>]*>(.*?)</script>',
        html,
        re.S,
    )
    if m:
        try:
            report = json.loads(m.group(1))
        except json.JSONDecodeError:
            report = None

    images: dict[str, bytes] = {}
    for m in re.finditer(
        r'<script type="midscene-image" data-id="([0-9a-f-]+)">(.*?)</script>', html, re.S
    ):
        img_id, content = m.group(1), m.group(2)
        if not content.startswith("data:"):
            continue
        try:
            b64 = content.split(",", 1)[1]
            images[img_id] = base64.b64decode(b64)
        except (IndexError, ValueError):
            continue
    return report, images


def extract_action_tasks(report: dict | None) -> list[dict]:
    """按顺序抽出 type=="Action Space" 的任务：{subType, center, after_id, before_id}。"""
    actions: list[dict] = []
    if not report:
        return actions
    for ex in report.get("executions", []) or []:
        for t in ex.get("tasks", []) or []:
            if t.get("type") != "Action Space":
                continue
            param = t.get("param") or {}
            locate = param.get("locate") if isinstance(param, dict) else None
            center = None
            if isinstance(locate, dict) and isinstance(locate.get("center"), (list, tuple)):
                center = list(locate["center"])[:2]
            ui = t.get("uiContext") or {}
            before_id = (ui.get("screenshot") or {}).get("id")
            after_id = before_id
            for r in t.get("recorder", []) or []:
                sc = (r or {}).get("screenshot") or {}
                if sc.get("type") == "midscene_screenshot_ref" and sc.get("id"):
                    after_id = sc["id"]
                    break
            actions.append(
                {
                    "subType": t.get("subType"),
                    "center": center,
                    "after_id": after_id,
                    "before_id": before_id,
                }
            )
    return actions


def build_action(tool: str, args: dict, report_act: dict | None) -> dict:
    """工具名 + 参数 + （可选）报告坐标 → 官方 action dict。"""
    action_type = TOOL_ACTION_TYPE.get(tool, tool)
    action: dict[str, Any] = {"action_type": action_type}

    if tool in ("input_text", "input_by_xpath"):
        text = args.get("value", args.get("text"))
        if text is not None:
            action["text"] = text
    elif tool == "answer":
        if args.get("answer") is not None:
            action["text"] = str(args["answer"])
    elif tool == "scroll":
        if args.get("direction"):
            action["direction"] = args["direction"]
    elif tool == "swipe":
        if args.get("direction"):
            action["direction"] = args["direction"]
        if args.get("distance") is not None:
            action["distance"] = args["distance"]
    elif tool in ("launch_app", "bring_app_to_front"):
        pkg = args.get("package", args.get("app_name"))
        if pkg:
            action["app_name"] = pkg

    # 点击类：从 midscene 报告拿中心坐标（agent log 里只有 locate 描述，没有坐标）
    if tool in ("tap", "doubul_click", "long_press", "tap_by_xpath"):
        if report_act and report_act.get("center"):
            action["x"], action["y"] = report_act["center"]
    return action


def build_prediction(turn: dict) -> str:
    parts: list[str] = []
    if turn.get("agent"):
        parts.append(turn["agent"].strip())
    if turn.get("thinking"):
        parts.append(f"<thinking>\n{turn['thinking'].strip()}\n</thinking>")
    parts.append(
        f"<tool_call>\n{turn['tool']}({json.dumps(turn['args'], ensure_ascii=False)})\n</tool_call>"
    )
    return "\n\n".join(p for p in parts if p)


def process_task(
    run_dir: str,
    out_dir: str,
    task: str,
    info: dict | None,
    with_screenshots: bool,
    force: bool,
) -> str | None:
    """处理单个任务，返回状态字符串；无动作时返回 None。"""
    out_task_dir = os.path.join(out_dir, task)
    out_traj = os.path.join(out_task_dir, "traj.json")
    if os.path.isfile(out_traj) and not force:
        return "skip"

    # 定位 agent_log（<Task>-<port>.log）
    logs = sorted(glob.glob(os.path.join(run_dir, "agent_logs", f"{task}-*.log")))
    log_path = logs[0] if logs else None
    if not log_path:
        return "no-log"

    goal, turns = parse_agent_log(log_path)
    if info and info.get("goal"):
        goal = info["goal"]

    # 定位 midscene 报告（<Task>-<port>/<...>.html）
    report, images = None, {}
    action_tasks: list[dict] = []
    if with_screenshots:
        reports = sorted(
            glob.glob(os.path.join(run_dir, "midscene_reports", f"{task}-*", "*.html"))
        )
        if reports:
            report, images = parse_midscene_report(reports[0])
            action_tasks = extract_action_tasks(report)

    # 逐轮：动作类工具 → step
    steps: list[dict] = []
    tool_names: list[str] = []
    # token 总量：parse_agent_log 已在每轮上都写入了整段日志累计的总量（RE_TOKEN 跨全日志累加），
    # 取首轮即可，切勿逐轮相加（会 N 倍膨胀）。
    prompt_tokens = turns[0].get("prompt_tokens", 0) or 0 if turns else 0
    completion_tokens = turns[0].get("completion_tokens", 0) or 0 if turns else 0
    report_idx = 0
    for turn in turns:
        tool = turn["tool"]
        if tool in PASSIVE_TOOLS:
            continue
        if tool not in tool_names:
            tool_names.append(tool)

        report_act = None
        if report_idx < len(action_tasks):
            report_act = action_tasks[report_idx]
            report_idx += 1

        action = build_action(tool, turn["args"], report_act)
        steps.append(
            {
                "task_goal": goal or "",
                "step": len(steps) + 1,
                "prediction": build_prediction(turn),
                "action": action,
                "ask_user_response": None,
                "tool_call": {"name": tool, "input": turn["args"]},
            }
        )

    if not steps:
        return "no-action"

    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    os.makedirs(out_task_dir, exist_ok=True)
    traj_data = {"0": {"tools": tool_names, "traj": steps, "token_usage": token_usage}}
    with open(out_traj, "w", encoding="utf-8") as f:
        json.dump(traj_data, f, ensure_ascii=False, indent=4)

    # result.txt：优先从原任务目录复制；否则由 info 生成
    src_result = os.path.join(run_dir, task, "result.txt")
    out_result = os.path.join(out_task_dir, "result.txt")
    if os.path.isfile(src_result):
        with open(src_result, encoding="utf-8") as f:
            result_txt = f.read()
    elif info is not None:
        result_txt = f"score: {info.get('score', 0.0)}\nreason: {info.get('reason', '')}"
    else:
        result_txt = "score: 0.0\nreason: unknown"
    with open(out_result, "w", encoding="utf-8") as f:
        f.write(result_txt)

    # 截图：按 step 顺序，从 midscene 图片表解出动作后截图 → <Task>-0-<step>.png
    if with_screenshots and images:
        from PIL import Image  # noqa: F401  (MobileWorld 依赖已含 PIL)

        shots_dir = os.path.join(out_task_dir, "screenshots")
        os.makedirs(shots_dir, exist_ok=True)
        written = 0
        for idx, act in enumerate(action_tasks):
            step_num = idx + 1
            if step_num > len(steps):
                break
            img_id = act.get("after_id") or act.get("before_id")
            data = images.get(img_id) if img_id else None
            if not data:
                continue
            try:
                img = Image.open(io.BytesIO(data))
                img.save(os.path.join(shots_dir, f"{task}-0-{step_num}.png"))
                written += 1
            except Exception:  # noqa: BLE001
                continue
        return f"ok:{len(steps)}steps:{written}shots"
    return f"ok:{len(steps)}steps"


def discover_tasks(run_dir: str) -> list[str]:
    """枚举 run 目录下的任务目录（含 result.txt 的），排除 agent_logs / midscene_reports。"""
    skip = {"agent_logs", "midscene_reports"}
    tasks = []
    for name in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, name)
        if not os.path.isdir(p) or name in skip or "_backup_" in name:
            continue
        if os.path.isfile(os.path.join(p, "result.txt")):
            tasks.append(name)
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="轨迹根目录，如 traj_logs/micar_e2e_1018")
    ap.add_argument("--with-screenshots", action="store_true", help="从 midscene HTML 抽截图")
    ap.add_argument("--out-dir", default=None, help="补齐后的输出目录（默认原地写入 run_dir）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 traj.json")
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")
    if not os.path.isdir(run_dir):
        sys.exit(f"错误：{run_dir} 不是目录")

    out_dir = args.out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)

    results = load_results(run_dir)
    tasks = discover_tasks(run_dir)
    if not tasks:
        sys.exit(f"错误：{run_dir} 下没找到任何含 result.txt 的任务目录")

    print(f"发现 {len(tasks)} 个任务（results.jsonl 命中 {len(results)} 条）")
    stats: dict[str, int] = {}
    for task in tasks:
        info = results.get(task)
        status = (
            process_task(run_dir, out_dir, task, info, args.with_screenshots, args.force) or "?"
        )
        stats[status.split(":")[0]] = stats.get(status.split(":")[0], 0) + 1

    print("完成：", ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    if out_dir != run_dir:
        print(f"输出目录：{out_dir}（提示：bundle 时把该目录当 traj_dir 传入）")
    else:
        print(f"已原地写入 {run_dir}，可直接 bundle_trajs.py {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
