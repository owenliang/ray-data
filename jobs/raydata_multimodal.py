import argparse

import ray

from utils import load_face_net, mask_video, oss_filesystem, strip_scheme


class FaceMosaic:
    def __init__(self, output_uri):
        self.net = load_face_net()
        self.output_uri = output_uri
        print("人脸检测模型已加载（每个 GPU Actor 一次）")

    def __call__(self, row):
        output, num_frames = mask_video(self.net, row["bytes"], row["path"], self.output_uri)
        return {"video_path": f"s3://{row['path']}", "output": output, "num_frames": num_frames}


def main():
    parser = argparse.ArgumentParser(description="多模态（Ray Data 版）：OSS 视频人脸马赛克")
    parser.add_argument("--input", required=True, help="OSS 输入目录，如 s3://bucket/videos/")
    parser.add_argument("--output", required=True, help="OSS 输出目录，如 s3://bucket/masked/")
    args = parser.parse_args()

    ray.init()
    oss_fs = oss_filesystem()

    ds = ray.data.read_binary_files(strip_scheme(args.input), include_paths=True, filesystem=oss_fs)

    ds = ds.map(FaceMosaic, fn_constructor_kwargs={"output_uri": args.output},
                concurrency=2, num_gpus=1, num_cpus=4)

    logs_path = strip_scheme(args.output).rstrip("/") + "-logs/"
    ds.write_parquet(logs_path, filesystem=oss_fs)
    print(f"③ 处理清单已写出 → s3://{logs_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
