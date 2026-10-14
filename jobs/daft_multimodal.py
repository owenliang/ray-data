import argparse
import os

import daft
from daft import DataType, col
from daft.io import IOConfig, S3Config

from utils import load_face_net, mask_video, oss_endpoint


def build_io_config():
    return IOConfig(s3=S3Config(
        endpoint_url=oss_endpoint(),
        key_id=os.environ["AWS_ACCESS_KEY_ID"],
        access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        force_virtual_addressing=True,
    ))


@daft.cls(gpus=1, cpus=4, max_concurrency=1)
class FaceMosaic:
    def __init__(self, output_uri):
        self.net = load_face_net()
        self.output_uri = output_uri
        print("人脸检测模型已加载（每个 GPU 实例一次）")

    @daft.method(return_dtype=DataType.struct({"output": DataType.string(), "num_frames": DataType.int64()}),
                 unnest=True)
    def mask(self, path, data):
        output, num_frames = mask_video(self.net, data, path, self.output_uri)
        return {"output": output, "num_frames": num_frames}


def main():
    parser = argparse.ArgumentParser(description="多模态（Daft 版）：OSS 视频人脸马赛克")
    parser.add_argument("--input", required=True, help="OSS 输入目录，如 s3://bucket/videos/")
    parser.add_argument("--output", required=True, help="OSS 输出目录，如 s3://bucket/masked/")
    parser.add_argument("--runner", choices=["local", "ray"], default="local",
                        help="ray=跑在 Ray 集群（提交端自动传）")
    args = parser.parse_args()

    if args.runner == "ray":
        daft.set_runner_ray()
    io_config = build_io_config()

    df = daft.from_glob_path(args.input, io_config=io_config)
    df = df.with_column("bytes", col("path").download(io_config=io_config))

    df = df.select(col("path").alias("video_path"), FaceMosaic(args.output).mask(col("path"), col("bytes")))

    logs_uri = args.output.rstrip("/") + "-logs/"
    df.write_parquet(logs_uri, io_config=io_config)
    print(f"③ 处理清单已写出 → {logs_uri}")


if __name__ == "__main__":
    main()
