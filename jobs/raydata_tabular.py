import argparse
import os

import pandas as pd
import ray
from ray.data import DataContext, ExecutionResources
from ray.data.aggregate import Count, Mean

from utils import DATA_PATH

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def add_discount(batch: pd.DataFrame) -> pd.DataFrame:
    batch["discount_price"] = batch["price"] * 0.9
    return batch


def main():
    parser = argparse.ArgumentParser(description="表格数据（Ray Data 版）：商品表批处理管线")
    parser.add_argument("--out-dir", default=os.path.join(THIS_DIR, "out", "raydata"),
                        help="结果目录；集群上可指向 /nas/data/... 以持久化")
    parser.add_argument("--max-cpus", type=float, default=None,
                        help="本作业 Ray Data 最多占用的 CPU 数，不设则用满集群")
    args = parser.parse_args()

    ray.init()
    if args.max_cpus:
        DataContext.get_current().execution_options.resource_limits = ExecutionResources(cpu=args.max_cpus)
    res = {k: v for k, v in ray.cluster_resources().items() if k in ("CPU", "GPU")}
    print(f"Ray 已启动 | 集群资源: {res}")

    ds = ray.data.read_parquet(DATA_PATH, override_num_blocks=4)
    print(f"\n① schema:\n{ds.schema()}")

    ds = ds.filter(expr="rating > 4 and price < 500")

    ds = ds.map_batches(add_discount, batch_format="pandas")
    ds = ds.select_columns(["id", "category", "price", "discount_price"])

    ds = ds.materialize()

    agg_ds = ds.groupby("category").aggregate(Mean("price"), Count()).sort("category")
    print("\n⑤ 按类目聚合（rating>4 且 price<500）:")
    for row in agg_ds.take_all():
        print(f"   {row}")

    print("\n⑥ 预览前 5 行:")
    print(ds.limit(5).to_pandas().to_string(index=False))

    ds.write_parquet(args.out_dir, mode="overwrite")
    files = [f for f in os.listdir(args.out_dir) if f.endswith(".parquet")]
    print(f"\n⑦ 已写出 {len(files)} 个 parquet 分块 → {args.out_dir}")

    ray.shutdown()


if __name__ == "__main__":
    main()
