from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from minecraft_vla.config import Step3Config, load_step3_config
from minecraft_vla.integration.minecraft_client import MinecraftClientConfig, create_client
from minecraft_vla.utils.io import ensure_dir, write_json
from minecraft_vla.utils.seed import set_seed


class ActionPolicy:
    def __init__(self, num_actions: int, default_action_id: int) -> None:
        self.num_actions = max(1, num_actions)
        self.default_action_id = max(0, default_action_id)

    def predict(self, step_idx: int, _observation: Dict[str, Any]) -> int:
        if self.num_actions <= 1:
            return self.default_action_id
        return int((self.default_action_id + step_idx) % self.num_actions)



def _infer_num_actions(step3_config: Step3Config) -> int:
    if step3_config.policy.source == "fixed":
        return max(1, int(step3_config.policy.max_action_id) + 1)

    report_path = Path(step3_config.policy.step2_report_path)
    if not report_path.exists():
        return max(1, int(step3_config.policy.max_action_id) + 1)

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    model_info = report.get("model", {})
    num_actions = model_info.get("num_actions")
    if isinstance(num_actions, int) and num_actions > 0:
        return num_actions
    return max(1, int(step3_config.policy.max_action_id) + 1)



def _save_trace_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def run_step3_pipeline(config: Step3Config) -> Dict[str, Any]:
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)

    num_actions = _infer_num_actions(config)
    policy = ActionPolicy(num_actions=num_actions, default_action_id=config.policy.default_action_id)

    client_config = MinecraftClientConfig(
        mode=config.server.mode,
        host=config.server.host,
        port=config.server.port,
        connect_timeout_sec=config.server.connect_timeout_sec,
        read_timeout_sec=config.server.read_timeout_sec,
        username=config.server.username,
    )
    client = create_client(client_config, dry_run=config.dry_run)

    connect_info = client.connect()
    trace_rows: List[Dict[str, Any]] = []
    rewards: List[float] = []
    action_hist: Dict[int, int] = {}

    try:
        for episode in range(config.evaluation.episodes):
            obs = client.reset()
            for step in range(config.evaluation.steps_per_episode):
                action_id = policy.predict(step, obs)
                result = client.step(action_id)
                reward = float(result.get("reward", 0.0))

                rewards.append(reward)
                action_hist[action_id] = action_hist.get(action_id, 0) + 1

                row = {
                    "episode": episode,
                    "step": step,
                    "action_id": action_id,
                    "reward": reward,
                    "tick": int(result.get("tick", 0)),
                }
                trace_rows.append(row)

                obs = result.get("observation", {})
                if bool(result.get("done", False)):
                    break
    finally:
        client.close()

    total_steps = len(trace_rows)
    mean_reward = float(sum(rewards) / total_steps) if total_steps > 0 else 0.0

    report = {
        "run_name": config.run_name,
        "dry_run": config.dry_run,
        "server": {
            "mode": config.server.mode,
            "host": config.server.host,
            "port": config.server.port,
            "username": config.server.username,
            "connect": connect_info,
        },
        "policy": {
            "source": config.policy.source,
            "step2_report_path": config.policy.step2_report_path,
            "num_actions": num_actions,
            "default_action_id": config.policy.default_action_id,
        },
        "evaluation": {
            "episodes": config.evaluation.episodes,
            "steps_per_episode": config.evaluation.steps_per_episode,
            "total_steps": total_steps,
            "mean_reward": mean_reward,
            "action_histogram": action_hist,
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "trace_jsonl": str(output_dir / "step3_trace.jsonl"),
        },
    }

    _save_trace_jsonl(output_dir / "step3_trace.jsonl", trace_rows)
    write_json(output_dir / "step3_report.json", report)
    write_json(output_dir / "step3_config_used.json", json.loads(json.dumps(asdict(config))))

    return report



def run_step3_pipeline_from_file(config_path: str | Path) -> Dict[str, Any]:
    config = load_step3_config(config_path)
    return run_step3_pipeline(config)
