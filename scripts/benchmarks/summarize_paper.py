from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.paper import publish_paper_summary
from common.outputs import repository_path


def main() -> None:
    parser = argparse.ArgumentParser(description="从完整预测生成独立的论文比较表和图。")
    parser.add_argument("--config", default="configs/v4/stage3/optimized_3407.yaml", help="与数据匹配的 Stage3 YAML；仅读取任务与划分。")
    parser.add_argument("--ilume", required=True, help="明确选择的 ILUME 运行根，包含 valid/fold1..5。")
    parser.add_argument("--baselines", required=True, help="包含十个完整 catalog_v4_3407 基线的根目录。")
    parser.add_argument("--with-test", action="store_true", help="同时要求并汇总 ILUME test/ 与基线五模型集成测试预测。")
    parser.add_argument("--output", required=True, help="新的仓库相对输出目录，已存在则拒绝。")
    args = parser.parse_args()
    publish_paper_summary(
        repository_path(args.config), repository_path(args.ilume),
        repository_path(args.baselines), repository_path(args.output), ROOT,
        with_test=args.with_test,
    )
    print(f"论文报告已写入 {args.output}")


if __name__ == "__main__":
    main()
