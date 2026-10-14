# Ray Data 与 Daft：表格与多模态的双引擎对照

| Demo | 入口 | 做什么 |
| --- | --- | --- |
| `raydata-tabular` | `jobs/raydata_tabular.py` | 商品表：读 parquet → 过滤 `rating>4 且 price<500` → 加折扣价列 → 按类目聚合 → 写出 parquet；Ray Data 显式编排（函数 UDF、`override_num_blocks`、`materialize`） |
| `daft-tabular` | `jobs/daft_tabular.py` | 同一任务的 Daft 写法：惰性表达式，过滤与列裁剪由优化器下推进扫描；结果与 Ray 版逐值一致 |
| `raydata-multimodal` | `jobs/raydata_multimodal.py` | OSS 视频人脸脱敏：一行一个视频，GPU Actor 内逐帧 YOLO 检测 → 扩框 → 邻帧补框 → 马赛克 → H.264 回传 OSS（全帧、pts 原样、不带音轨），另写处理清单 |
| `daft-multimodal` | `jobs/daft_multimodal.py` | 同一任务的 Daft 写法：`from_glob_path` + `download` + `@daft.cls` 常驻模型；输出视频与 Ray 版字节相同 |

## 启动

```powershell
$env:RAY_DASHBOARD_ADDRESS = "<Job 网关>"; $env:RAY_TOKEN = "<Token>"
# 集群准备，单独提交一次：全节点装依赖、Driver 在 NAS 上准备人脸模型和表格数据（已存在就跳过）；集群扩容后重跑
python submit_data_job.py --demo bootstrap

python submit_data_job.py --demo raydata-tabular --tabular-out-dir /nas/data/ray-data-demo/raydata
python submit_data_job.py --demo daft-tabular    --tabular-out-dir /nas/data/ray-data-demo/daft

$env:AWS_ACCESS_KEY_ID = "<AK>"; $env:AWS_SECRET_ACCESS_KEY = "<SK>"
python submit_data_job.py --demo raydata-multimodal --multimodal-input s3://rayoss-data/videos/ --multimodal-output s3://rayoss-data/masked/raydata/
python submit_data_job.py --demo daft-multimodal    --multimodal-input s3://rayoss-data/videos/ --multimodal-output s3://rayoss-data/masked/daft/
```

