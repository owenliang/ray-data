import os
import random
import shutil
import socket
import subprocess
import sys
import urllib.request

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from utils import DATA_PATH, FACE_MODEL

PACKAGES = [
    "pandas<3", "daft>=0.7", "av>=14,<20", "numpy>=1.23,<2", "torch==2.6.0", "torchvision==0.21.0",
    "opencv-python-headless==4.10.0.84", "matplotlib>=3.3", "pillow>=7.1.2", "pyyaml>=5.3.1",
    "requests>=2.23", "psutil>=5.8", "polars>=0.20", "nvidia-ml-py>=12", "ultralytics-thop>=2.2.2",
    "ultralytics-platform>=0.1.45",
]
NO_DEPS_PACKAGES = ["ultralytics==8.4.176"]
CHECK = "import av, cv2, daft, torch, torchvision, ultralytics; print('torch', torch.__version__)"

FACE_MODEL_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11s-face.pt"
CATEGORIES = ["数码", "服饰", "美妆", "家居", "食品"]


def install():
    check = [sys.executable, "-c", CHECK]
    r = subprocess.run(check, capture_output=True, text=True, errors="replace")
    if not r.returncode:
        return f"{socket.gethostname()}: {r.stdout.strip()}（已安装）"

    pip = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
    for cmd in (pip + PACKAGES, pip + ["--no-deps"] + NO_DEPS_PACKAGES, check):
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if r.returncode:
            raise RuntimeError(f"{socket.gethostname()} 失败：{' '.join(cmd[:4])}…\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return f"{socket.gethostname()}: {r.stdout.strip()}（新安装）"


def install_on_all_nodes():
    task = ray.remote(num_cpus=0)(install)
    nodes = [n["NodeID"] for n in ray.nodes() if n["Alive"]]
    for line in ray.get([task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n, soft=False)).remote()
                         for n in nodes]):
        print(line)
    print(f"全部 {len(nodes)} 个存活节点依赖就绪")


def _atomic_write(path, write):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def ensure_face_model():
    if os.path.exists(FACE_MODEL):
        print(f"模型已就绪: {FACE_MODEL}")
        return

    def download(tmp):
        with urllib.request.urlopen(FACE_MODEL_URL, timeout=300) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)

    _atomic_write(FACE_MODEL, download)
    print(f"模型已下载: {FACE_MODEL}")


def ensure_data(n=200):
    if os.path.exists(DATA_PATH):
        print(f"数据集已就绪: {DATA_PATH}")
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = random.Random(42)
    rows = {k: [] for k in ("id", "category", "price", "rating", "sales")}
    for i in range(n):
        rows["id"].append(i)
        rows["category"].append(CATEGORIES[i % len(CATEGORIES)])
        rows["price"].append(round(rng.uniform(9.9, 999.0), 1))
        rows["rating"].append(round(rng.uniform(3.0, 5.0), 1))
        rows["sales"].append(rng.randrange(0, 10000))
    _atomic_write(DATA_PATH, lambda tmp: pq.write_table(pa.table(rows), tmp))
    print(f"数据集已生成: {DATA_PATH}（{n} 行 × {len(rows)} 列）")


def main():
    ray.init()
    install_on_all_nodes()
    ensure_face_model()
    ensure_data()


if __name__ == "__main__":
    main()
