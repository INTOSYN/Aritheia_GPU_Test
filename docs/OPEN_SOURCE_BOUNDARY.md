# 常规检测开源与专项深测边界

从 0.5.0rc5 起，常规检测源码完整公开，不再要求单独安装 `computeproof-core`。用户可以检查数值比较和判断过程，运行十二个场景，并根据公开 CUDA 源码重建原生探针。只有异常后的专项深度诊断实现可以作为独立编译扩展另行分发，不作为基础客户端的隐藏依赖。

## 公开的数值比较与探针

| 公开文件 | 内容 |
|---|---|
| `src/agrel_public/exact.py` | 精确输入构造、算术条件、输出比较与错误记录 |
| `src/agrel_public/reference.py`、`frozen.py` | 固定参考、构建身份、输入/输出及分块摘要核对 |
| `src/agrel_public/probes.py` | 常规探针与公开后端执行 |
| `src/agrel_public/native_smid.py` | CUDA Driver API 加载、原生探针调度和逻辑 SM 覆盖 |
| `src/agrel_public/assets/native/source/smid_probe.cu` | 原生 BF16 MMA、FP32 FMA 探针源码 |
| `src/agrel_public/assets/native/source/build_native.py` | 原生代码构建和指令清单生成 |
| `src/agrel_public/assets/` | 运行所需固定参考、参数、摘要、许可和内置场景数据 |

Python 包没有平台专用的主机扩展，常规客户端可制作为 `py3-none-any` wheel；原生 GPU 代码以 `fatbin` 数据文件随包提供。通用 wheel 不表示任意 GPU 或操作系统都已实测支持：驱动、CUDA PyTorch、架构和执行路径仍需实际检查。重编译原生代码后必须重新核验相应参考和指令路径，不能只重写哈希并沿用官方验证结论。

## 完整公开的十二场景代码

用户可以从公开源码复查每个场景读取了什么数据、执行了什么计算，以及如何得到最终指标。场景没有隐藏的远程执行步骤，也不会把未发布实现替换成合成结果。

| 公开文件 | 内容 |
|---|---|
| `src/agrel_public/scenarios/registry.py` | 12 个场景标识、显示名称、两档配置、数据层级、输入规模和代表性探针形状 |
| `src/agrel_public/scenarios/engine.py` | BF16 自动混合精度推理、短程 AdamW 训练、视觉特征缓存、Tanimoto/稠密检索、分类/回归/排序指标和完整分派逻辑 |
| `src/agrel_public/scenarios/llm.py` | 固定语言模型 revision、本地加载、提示构造、A/B/C/D 受约束评分、非有限值检查和指标 |
| `src/agrel_public/scenarios/assets.py` | 指定版本数据与模型的定位、逐文件 SHA-256、可选包下载、ZIP 安全检查和离线安装 |
| `src/agrel_public/selection.py` | 内置/可选场景选择、缺失原因和下载授权处理 |
| `src/agrel_public/worker.py` | 一张 GPU 上的场景、矩阵探针、显存往返和原生探针执行顺序 |
| `src/agrel_public/processes.py`、`src/agrel_public/cli.py` | 多卡隔离、进度、退出状态、报告和用户交互 |

场景标识与实现路径：

| 场景标识 | 实现入口 |
|---|---|
| [`drug_screen/`](../src/agrel_public/scenarios/drug_screen/) | `engine.supervised`，固定模型的 MLP 推理与回归/排序指标 |
| [`drug_finetune/`](../src/agrel_public/scenarios/drug_finetune/) | `engine.supervised` 的训练分支，统一初始检查点与 AdamW 参数更新 |
| [`molecule_neighbors/`](../src/agrel_public/scenarios/molecule_neighbors/) | `engine.retrieval`，分块 Tanimoto top-k |
| [`cytology/`](../src/agrel_public/scenarios/cytology/) | `engine.supervised`，表格分类 |
| [`ocr_cache/`](../src/agrel_public/scenarios/ocr_cache/) | `engine.supervised` 的缓存分支，GPU 编码、主机缓存和 GPU 分类头 |
| [`genomics_splice/`](../src/agrel_public/scenarios/genomics_splice/) | `engine.supervised`，序列编码分类 |
| [`medical_ultrasound/`](../src/agrel_public/scenarios/medical_ultrasound/) | `engine.supervised`，医疗影像分类 |
| [`materials_screen/`](../src/agrel_public/scenarios/materials_screen/) | `engine.supervised`，材料性质回归 |
| [`singlecell_neighbors/`](../src/agrel_public/scenarios/singlecell_neighbors/) | `engine.retrieval`，单细胞近邻检索 |
| [`biomed_rag/`](../src/agrel_public/scenarios/biomed_rag/) | `engine.retrieval`，生物医学知识检索 |
| [`llm_biomed_tables/`](../src/agrel_public/scenarios/llm_biomed_tables/) | `llm.evaluate`，结构化信息理解 |
| [`agent_tools/`](../src/agrel_public/scenarios/agent_tools/) | `llm.evaluate`，智能体工具决策 |

共享函数用于避免十二份重复代码；注册表和分派器明确展示每个场景进入哪个实现。每个独立文件夹内有 SPEC、实际 execute 入口、__main__.py 和 README；共享函数也全部公开。两档配置从各目录汇总到注册表，普通十二场景不由服务器暗改参数。后续签名深测是单独预览、确认和标记的新实验，不是隐式修改场景。

网络下载、研究摘要/详细白名单、深测验签与同意逻辑同样公开，见 `telemetry.py`、`research_evidence.py`、`deep_cases.py` 与 `network.py`。公开客户端沿用现有 GPL 许可；另行编译分发不消除既有许可证义务。

## 另行授权的专项深测扩展

私有代码边界仅限于常规检查发现异常后用于针对性复核的专项深测实现。它不接管基础数值比较，不隐藏十二场景的算法，也不在常规 `check`、`guided` 或可选场景下载过程中自动安装。

如维护者提供深测包，发行描述必须绑定报告、目标设备、包摘要、有效期、操作系统、处理器和 CPython ABI。用户分别预览并确认下载、执行和结果上传。原生代码不是沙箱内的不可信脚本；只有信任发行者后才应执行。当前实现不代表已提供 Windows 等各平台的深测二进制，也不代表生产服务已经部署。

## 不进入用户仓库的内容

- 维护者实卡日志和验收报告；
- 发布门、构建流水线和内部审计结果；
- 私有专项深测实现、其构建中间产物和未发布包；
- 研究接收服务、数据库结构、部署配置、凭据与签名私钥。

用户仓库包括可安装客户端、数值比较、完整场景实现、常规原生探针源码和构建脚本、公开说明、资产许可及公共行为测试。维护资料保存在独立目录，不作为产品内容分发。`computeproof-core` 拆包说明及 rc4 以前的候选仅是历史发行记录，不代表本版的源码边界。
