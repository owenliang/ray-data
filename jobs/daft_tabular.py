import argparse
import glob
import os

import daft
from daft import col

from utils import DATA_PATH

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="表格数据（Daft 版）：同一任务的声明式写法")
    parser.add_argument("--out-dir", default=os.path.join(THIS_DIR, "out", "daft"),
                        help="结果目录；集群上可指向 /nas/data/... 以持久化")
    parser.add_argument("--runner", choices=["local", "ray"], default="local",
                        help="ray=跑在 Ray 集群（提交端自动传）")
    args = parser.parse_args()

    if args.runner == "ray":
        daft.set_runner_ray()

    df = daft.read_parquet(DATA_PATH)
    print(f"\n① schema:\n{df.schema()}")

    df = df.where(col("rating") > 4).where(col("price") < 500)

    df = df.with_column("discount_price", col("price") * 0.9)
    df = df.select("id", "category", "price", "discount_price")

    print("\n执行计划:")
    df.explain(show_all=True)

    df = df.collect()

    agg_df = df.groupby("category").agg(
        col("price").mean().alias("mean(price)"),
        col("id").count().alias("count()"),
    ).sort("category")
    print("\n⑤ 按类目聚合（rating>4 且 price<500）:")
    agg_df.show()

    print("⑥ 预览前 5 行:")
    df.show(5)

    for stale in glob.glob(os.path.join(args.out_dir, "*.parquet")):
        os.remove(stale)
    files_df = df.write_parquet(args.out_dir)
    print(f"⑦ 已写出 parquet → {args.out_dir}")
    files_df.show()


if __name__ == "__main__":
    main()
