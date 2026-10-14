import os
import sys
import time
import argparse

from ray.job_submission import JobStatus, JobSubmissionClient

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(THIS_DIR, "jobs")

ENTRIES = {
    "bootstrap": "bootstrap.py",
    "raydata-tabular": "raydata_tabular.py",
    "daft-tabular": "daft_tabular.py",
    "raydata-multimodal": "raydata_multimodal.py",
    "daft-multimodal": "daft_multimodal.py",
}
MULTIMODAL_DEMOS = {"raydata-multimodal", "daft-multimodal"}

NAS_ROOT = os.environ.get("RAY_DATA_NAS_ROOT", "/nas/data/ray-data-demo")
TABULAR_DATA_PATH = os.environ.get("TABULAR_DATA_PATH", f"{NAS_ROOT}/data/products.parquet")
FACE_MODEL_PATH = os.environ.get("FACE_MODEL_PATH", f"{NAS_ROOT}/models/yolov11s-face.pt")
AWS_ENV_KEYS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL"]


def build_runtime_env(args):
    env = {
        "working_dir": JOBS_DIR,
        "excludes": ["__pycache__", "*.pyc", "out", "models"],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TABULAR_DATA_PATH": TABULAR_DATA_PATH,
            "FACE_MODEL_PATH": FACE_MODEL_PATH,
        },
    }
    if args.demo in MULTIMODAL_DEMOS:
        for key in AWS_ENV_KEYS:
            if os.environ.get(key):
                env["env_vars"][key] = os.environ[key]
    return env


DEMO_ARGS = {
    "tabular_out_dir": ({"raydata-tabular", "daft-tabular"}, "--out-dir"),
    "tabular_max_cpus": ({"raydata-tabular"}, "--max-cpus"),
    "multimodal_input": (MULTIMODAL_DEMOS, "--input"),
    "multimodal_output": (MULTIMODAL_DEMOS, "--output"),
}


def build_entrypoint(args):
    parts = [f"python {ENTRIES[args.demo]}"]
    if args.demo.startswith("daft"):
        parts.append("--runner ray")
    for dest, (demos, flag) in DEMO_ARGS.items():
        value = getattr(args, dest)
        if args.demo in demos and value is not None:
            parts.append(f"{flag} {value}")
    return " ".join(parts)


def _wait_job(client, job_id):
    last_status = None
    while True:
        status = client.get_job_status(job_id)
        if status != last_status:
            print(f"Job status: {status}", flush=True)
            last_status = status
        if status.is_terminal():
            logs = client.get_job_logs(job_id)
            if logs:
                print(logs, end="" if logs.endswith("\n") else "\n")
            return status
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="提交 Ray Data / Daft demo 到 EMR Ray 集群")
    parser.add_argument("--demo", choices=sorted(ENTRIES), default="raydata-tabular",
                        help="选一个 job 提交；首次或集群扩容后先提交 bootstrap")
    tabular = parser.add_argument_group("表格 demo（raydata-tabular / daft-tabular）")
    tabular.add_argument("--tabular-out-dir", default=None,
                         help="结果目录；多节点集群须指向共享存储，如 /nas/data/...")
    tabular.add_argument("--tabular-max-cpus", type=float, default=None,
                         help="仅 raydata-tabular：本作业 Ray Data 最多占用的 CPU 数")
    multimodal = parser.add_argument_group("多模态 demo（raydata-multimodal / daft-multimodal）")
    multimodal.add_argument("--multimodal-input", default=None, help="必填，OSS 输入目录，如 s3://bucket/videos/")
    multimodal.add_argument("--multimodal-output", default=None, help="必填，OSS 输出目录，如 s3://bucket/masked/")
    args = parser.parse_args()

    address = os.environ.get("RAY_DASHBOARD_ADDRESS")
    token = os.environ.get("RAY_TOKEN")
    if not address or not token:
        sys.exit("请先设置 RAY_DASHBOARD_ADDRESS 和 RAY_TOKEN")

    client = JobSubmissionClient(address, headers={"ray-token": token})
    entrypoint = build_entrypoint(args)
    print(f"提交入口: {entrypoint}")
    job_id = client.submit_job(entrypoint=entrypoint, runtime_env=build_runtime_env(args))
    print(f"Job ID: {job_id}")

    status = _wait_job(client, job_id)
    print(f"\nJob status: {status}", flush=True)
    sys.exit(0 if status == JobStatus.SUCCEEDED else 1)


if __name__ == "__main__":
    main()
