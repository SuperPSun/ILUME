from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stage3.hpo.workflow import Workflow
from stage3.hpo.scheduler import BudgetExhausted


def main():
    parser = argparse.ArgumentParser(description="ILUME 独立 HPO v4 研究控制器。")
    parser.add_argument("action", choices=("audit", "prepare", "sensitivity", "search", "confirm", "splits", "lock", "repeat", "test", "report"))
    parser.add_argument("--config", default="configs/HPO_v4/study.yaml")
    parser.add_argument("--study-id")
    parser.add_argument("--branch", choices=("all", "joint", "posttraining"), default="all")
    parser.add_argument("--search-phase", choices=("broad", "local"), default="broad")
    parser.add_argument("--confirm-phase", choices=("folds", "seeds"), default="folds")
    parser.add_argument("--measure", action="store_true", help="audit 后执行资源测量；会训练完整 Base/边界候选。")
    parser.add_argument("--devices", help="逗号分隔，如 cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--jobs-per-gpu", type=int)
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--gpu-memory-reserve-gb", type=float)
    parser.add_argument("--max-gpu-hours", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="仅 prepare/sensitivity 生成完整 YAML 和作业清单，不训练。")
    args = parser.parse_args()
    if args.dry_run and args.action not in {"prepare", "sensitivity"}:
        parser.error("--dry-run is supported by prepare and sensitivity")
    if args.measure and args.action != "audit":
        parser.error("--measure requires audit")
    study = __import__("yaml").safe_load(Path(args.config).read_text())
    study_id = args.study_id or study["study_id"]
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", study_id):
        parser.error("Invalid study_id")
    lock_file = ROOT / "outputs" / "HPO_v4" / study_id / "studies" / "controller.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another HPO controller owns this study")
        workflow = Workflow(args.config, args)
        try:
            getattr(workflow, args.action)()
        except BudgetExhausted as error:
            workflow.persist()
            workflow.report()
            print(str(error), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
