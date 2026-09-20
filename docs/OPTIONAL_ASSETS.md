# 原始仓库与按需下载

八个内置场景直接随源码提供。四个可选场景使用三份共享资源，不需要本项目发布数据 ZIP：

| 资源 | 原始地址 | 下载体积 | 处理 |
|---|---|---:|---|
| PBMC3k processed | [Scanpy 官方列出的 CZI 数据源](https://raw.githubusercontent.com/chanzuckerberg/cellxgene/main/example-dataset/pbmc3k.h5ad) | 24.7 MB | 读取 2638 × 1838 表达矩阵、归一化 |
| SciFact | [BEIR 原始仓库](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip) | 2.8 MB | 官方 test qrels、TF-IDF、512 维 LSA |
| Qwen3.5-4B | [Qwen 官方仓库](https://huggingface.co/Qwen/Qwen3.5-4B) | 约 9.35 GB | 两个语言场景共用；只接受清单中的固定版本与文件摘要 |

下载地址、版本、文件长度、SHA-256 和许可保存在 `assets/optional_assets.json`。PBMC3k 摘要与 Scanpy 的官方数据注册表一致。两个模型权重分片及随包任务输入的 SHA-256 与已有 Qwen3.5-4B 实验记录一致；这不是新客户端在全部平台已通过实卡测试的声明。

## 同意规则

- `guided` 的 10 秒倒计时只用于 PBMC3k 与 SciFact，输入 c 可取消。
- LLM 下载必须另外回答“是”，默认拒绝。无人值守需单独的 `--accept-model-download`。
- `--accept-download`、选择全部场景、同意上传均不构成模型下载同意。
- `--offline` 禁止网络；即使带有同意参数也不会下载。已安装且通过校验的资产可以复用。
- 下载同意不授权结果上传或私有深度诊断。

```bash
python -m pip install '.[llm,datasets]'
# 两个数据场景，不下载 LLM
python -m computeproof check --scenarios singlecell_neighbors,biomed_rag --accept-download
# 只下载官方模型，不启动 GPU 检测
python -m computeproof assets download language --accept-model-download
# 全部场景，需要两项独立授权
python -m computeproof check --scenarios all --accept-download --accept-model-download
```

下载先写到资产目录内的临时目录；验证大小和 SHA-256 后才安装。失败、中止不会留下可被误认为完整模型的目录，不覆盖不同内容的旧目录。模型推理使用 local_files_only，不会自行补下载。缺依赖、磁盘不足、显存不足或校验失败均记录为未完成，不换模型、不构造随机数据。

原文件可以离线复制到 `ARITHEIA_ASSETS/models/qwen3.5-4b/`，保留官方文件名，随后程序逐文件校验。其他模型或 adapter 被拒绝。历史小模型 ZIP 不再支持。旧的 singlecell / literature ZIP 仍可用 `assets install` 离线核验导入，但不再是默认在线下载方式。

## 预处理与数值解释

PBMC3k 和 SciFact 在用户主机做数据准备，GPU 执行检索矩阵计算。SciFact 的 SVD 与归一化可随 NumPy、SciPy、BLAS 和 scikit-learn 版本产生差异，因此保存输入摘要、原文件摘要、处理方法和依赖版本；不把这些输入冒充归档特征的逐位相同副本，也不使用旧任务分数直接判 GPU 故障。算术判断仍来自独立的精确探针。

模型文件完整性与模型输出数值校验是不同层次。Qwen3.5-4B 使用 BF16、关闭 thinking、eager attention 和无 KV cache 的 A/B/C/D 评分；任何选项出现 NaN/Inf 的题目不生成有效选择。历史 68 项精确探针的数值参考不变，其语言相关形状是通用工作负载，不能声称覆盖 Qwen3.5-4B 的所有算子。
