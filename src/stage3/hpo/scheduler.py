"""单控制进程的设备队列；worker 只调用现有 Stage 入口。"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from common.io import atomic_json, sha256_file


@dataclass(frozen=True)
class Job:
    key: str
    argv: tuple[str, ...]
    output: Path
    completion: str = "."
    memory_gb: float = 0


class BudgetExhausted(RuntimeError):
    pass


def free_memory_gb(devices):
    if devices == ("cpu",):
        return {"cpu": float("inf")}
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    return {f"cuda:{index.strip()}": float(memory) / 1024
            for index, memory in (line.split(",") for line in result.stdout.splitlines())}


def completed_output(output):
    metadata, summary = output / "metadata.json", output / "summary.json"
    return metadata.is_file() and summary.is_file() and json.loads(metadata.read_text()).get("status") == "completed"


def output_hashes(output):
    # Bind accepted final models and prediction manifests, including their referenced tensors/CSVs.
    files = [file for file in output.rglob("*") if file.is_file() and
             (file.suffix in {".json", ".yaml"} or file.name.endswith("final.pt") or file.name == "regression_head.pt" or file.suffix == ".csv")]
    return {str(file.relative_to(output)): sha256_file(file) for file in sorted(files)}


def process_usage(pid):
    """Observe worker and spawn children without adding a runtime dependency."""
    pending, observed = [pid], {"rss_bytes": 0, "cpu_seconds": 0, "read_bytes": 0, "write_bytes": 0}
    while pending:
        child = pending.pop()
        root = Path("/proc") / str(child)
        try:
            pending.extend(int(value) for value in (root / "task" / str(child) / "children").read_text().split())
            status = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in (root / "status").read_text().splitlines()}
            observed["rss_bytes"] += int(status.get("VmRSS", "0 kB").split()[0]) * 1024
            fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
            observed["cpu_seconds"] += (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
            io = {line.split(":")[0]: int(line.split(":")[1]) for line in (root / "io").read_text().splitlines()}
            for name in ("read_bytes", "write_bytes"):
                observed[name] += io[name]
        except PermissionError:
            observed["io_unavailable"] = True
        except (FileNotFoundError, ProcessLookupError):
            continue
    return observed


class Scheduler:
    def __init__(self, root, *, devices, jobs_per_gpu=1, max_parallel=None, reserve_gb=6,
                 max_gpu_hours=None, resume=False, poll_seconds=1):
        if not devices or len(set(devices)) != len(devices) or jobs_per_gpu < 1:
            raise ValueError("Provide distinct devices and positive jobs-per-gpu")
        if any(device != "cpu" and (not device.startswith("cuda:") or not device[5:].isdigit()) for device in devices):
            raise ValueError("Devices must be cpu or cuda:N")
        if "cpu" in devices and devices != ["cpu"] and devices != ("cpu",):
            raise ValueError("CPU cannot share a GPU queue")
        self.devices = tuple(devices)
        self.quota = jobs_per_gpu
        self.limit = len(devices) * jobs_per_gpu if max_parallel is None else max_parallel
        if self.limit < 1 or self.limit > len(devices) * jobs_per_gpu or reserve_gb < 0:
            raise ValueError("Invalid global parallel or memory limit")
        if max_gpu_hours is not None and max_gpu_hours <= 0:
            raise ValueError("GPU-hour budget must be positive")
        self.root = Path(root)
        self.path = self.root / "studies" / "jobs.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"gpu_hours": 0, "jobs": {}}
        self.resume, self.reserve, self.hours, self.poll = resume, reserve_gb, max_gpu_hours, poll_seconds

    def run(self, jobs):
        pending = {job.key: job for job in jobs}
        if len(pending) != len(jobs):
            raise ValueError("Duplicate HPO job key")
        running, accepted = {}, {}
        for key, job in list(pending.items()):
            signature = {"argv": [Path(job.argv[0]).name, *job.argv[1:]], "config_sha256": None, "completion": job.completion}
            if "--config" in job.argv:
                signature["config_sha256"] = sha256_file(job.argv[job.argv.index("--config") + 1])
            previous = self.state["jobs"].get(key)
            if previous and previous["signature"] != signature:
                raise ValueError("HPO job configuration changed")
            if previous and previous["status"] == "running" and Path(f"/proc/{previous['pid']}").exists():
                raise RuntimeError("Previous HPO worker is still alive")
            if previous and previous["status"] == "completed":
                if not self.resume:
                    raise FileExistsError("Use --resume for existing HPO jobs")
                output = Path(previous["output"])
                if not completed_output(output / job.completion) or output_hashes(output / job.completion) != previous["hashes"]:
                    raise ValueError(f"HPO accepted output damaged: {output}")
                accepted[key] = output
                pending.pop(key)
            elif previous and previous["status"] == "failed" and previous.get("failure") != "oom":
                if not self.resume:
                    raise FileExistsError("Use --resume to inspect existing failures")
                accepted[key] = None
                pending.pop(key)
        last = time.monotonic()
        last_saved = last
        try:
            while pending or running:
                now = time.monotonic()
                self.state["gpu_hours"] += (now - last) * len({item[2] for item in running.values() if item[2] != "cpu"}) / 3600
                last = now
                if now - last_saved >= 10:
                    atomic_json(self.path, self.state)
                    last_saved = now
                for key, (job, process, device, log, started) in list(running.items()):
                    row = self.state["jobs"][key]
                    usage = process_usage(process.pid)
                    row["observed_resources"] = {name: max(value, row.get("observed_resources", {}).get(name, 0)) for name, value in usage.items()}
                    status = process.poll()
                    if status is None:
                        continue
                    log.close()
                    row = self.state["jobs"][key]
                    row.update(status="completed" if status == 0 and completed_output(Path(row["output"]) / job.completion) else "failed",
                               exit_code=status, wall_seconds=now - started)
                    running.pop(key)
                    row["host_load_end"] = os.getloadavg()
                    performance = Path(row["output"]) / "performance.jsonl"
                    if performance.is_file():
                        rows = [json.loads(line) for line in performance.read_text().splitlines() if line]
                        row["peak_allocated_bytes"] = max(item.get("train_peak_allocated_bytes", 0) for item in rows)
                    if row["status"] == "completed":
                        accepted[key] = Path(row["output"])
                        row["hashes"] = output_hashes(accepted[key] / job.completion)
                    else:
                        tail = Path(row["log"]).read_text(errors="replace")[-20000:].lower()
                        failure = "oom" if "out of memory" in tail else "nonfinite" if any(word in tail for word in ("non-finite", "nonfinite", "nan gradient")) else "contract"
                        row["failure"] = failure
                        if failure == "oom" and row["attempt"] < 2:
                            self.quota = 1
                            pending[key] = job
                        elif failure in {"oom", "nonfinite"}:
                            accepted[key] = None
                        else:
                            atomic_json(self.path, self.state)
                            raise RuntimeError(f"HPO data/artifact/worker failure: {key}; {row['log']}")
                    atomic_json(self.path, self.state)
                exhausted = self.hours is not None and self.state["gpu_hours"] >= self.hours
                if exhausted:
                    raise BudgetExhausted("GPU-hour budget exhausted; outputs and pending jobs preserved")
                if len(running) < self.limit and pending:
                    available = free_memory_gb(self.devices)
                    for key, job in list(pending.items()):
                        device = next((candidate for candidate in self.devices
                                       if sum(item[2] == candidate for item in running.values()) < self.quota
                                       and available.get(candidate, 0) >= self.reserve + job.memory_gb), None)
                        if device is None or len(running) >= self.limit:
                            continue
                        old = self.state["jobs"].get(key, {})
                        attempt = old.get("attempt", 0) + 1
                        output = job.output if attempt == 1 else job.output.parent / f"{job.output.name}_attempt{attempt}"
                        argv = [str(output) if value == str(job.output) else value for value in job.argv]
                        output.parent.mkdir(parents=True, exist_ok=True)
                        log_path = output.parent / f"{output.name}.attempt{attempt}.log"
                        log = log_path.open("x")
                        env = dict(os.environ)
                        if device != "cpu":
                            env["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]
                        if any("stage1/regression" in value for value in argv):
                            argv += ["--device", "cpu" if device == "cpu" else "cuda"]
                        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                        signature = {"argv": [Path(job.argv[0]).name, *job.argv[1:]], "config_sha256": None, "completion": job.completion}
                        if "--config" in job.argv:
                            signature["config_sha256"] = sha256_file(job.argv[job.argv.index("--config") + 1])
                        self.state["jobs"][key] = {"signature": signature, "output": str(output), "device": device,
                                                   "attempt": attempt, "pid": process.pid, "status": "running", "log": str(log_path),
                                                   "host_load_start": os.getloadavg(),
                                                   "previous_attempts": old.get("previous_attempts", []) + ([{key: value for key, value in old.items() if key != "previous_attempts"}] if old else [])}
                        running[key] = (job, process, device, log, now)
                        pending.pop(key)
                        available[device] -= job.memory_gb
                        atomic_json(self.path, self.state)
                time.sleep(self.poll)
        finally:
            for key, (_, process, _, log, _) in running.items():
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                log.close()
                self.state["jobs"][key]["status"] = "interrupted"
            atomic_json(self.path, self.state)
        return accepted
