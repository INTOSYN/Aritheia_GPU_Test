# Aritheia — GPU Numerical Reliability Test

Aritheia 是本工具的公开名称；当前 Python 包及命令入口保留 `computeproof`，版本为 0.5.0rc6。

本公开版本使用 `~/.aritheia` 保存配置和缓存，可通过 `ARITHEIA_HOME` 和 `ARITHEIA_ASSETS` 指定目录。旧安装的配置不会自动迁移。默认研究摘要和详细记录仍使用现有 `computeproof.summary.v3` / `computeproof.detail.v2` 协议；自行配置的旧式 `/v1` 接收服务及签名包需支持新的 `aritheia.*` 格式标识，不能直接假定兼容。

**用一张或多张 NVIDIA GPU 检查科研计算是否出现数值异常。**

训练能结束、loss 在下降，甚至实验成绩更好，都不能证明每一步计算正确。ComputeProof 提供 12 个真实科研场景，并用固定输入、精确算术合同、分块摘要和原生逻辑 SM 探针检查 BF16/FP32 计算链路。

每张卡独立完成计算和判断。用户不需要 CPU 矩阵回算、健康对照卡、账户或数据库连接；主机 CPU 只负责准备输入、核对摘要、计算场景指标和保存报告。

## 源码边界

常规检测客户端完整公开源码，不再依赖 `computeproof-core` 平台扩展包。数值对比、参考摘要校验、矩阵及显存探针、原生 CUDA 探针和十二个场景均可检查、修改和自行构建。只把异常发生后的专项深度诊断扩展留作单独分发；它不是正常检测的必装依赖。

12 个场景的运行代码完整公开，包括数据读取、模型结构、两档配置、训练与推理、检索、语言模型提示与评分、指标计算、资产校验和场景调度：

- [`scenarios/`](src/agrel_public/scenarios/)：十二个独立命名文件夹，每个都有配置、可执行入口和说明；注册表汇总两档配置与探针形状。
- [`scenarios/engine.py`](src/agrel_public/scenarios/engine.py)：MLP 推理、AdamW 微调、特征缓存、相似性检索、指标和场景分派。
- [`scenarios/llm.py`](src/agrel_public/scenarios/llm.py)：本地语言模型加载、结构化信息理解、智能体决策和选项评分。
- [`scenarios/assets.py`](src/agrel_public/scenarios/assets.py)：内置及可选数据的定位、完整性校验、下载和离线安装。
- [`worker.py`](src/agrel_public/worker.py) 与 [`cli.py`](src/agrel_public/cli.py)：单卡/多卡执行、进度、结果汇总和研究记录控制。

- [`exact.py`](src/agrel_public/exact.py)、[`reference.py`](src/agrel_public/reference.py)、[`frozen.py`](src/agrel_public/frozen.py)、[`probes.py`](src/agrel_public/probes.py)：精确输入、固定参考、数值比较、复核及探针执行。
- [`native_smid.py`](src/agrel_public/native_smid.py) 与 [`assets/native/source/`](src/agrel_public/assets/native/source/)：CUDA 驱动接口、原生逻辑 SM 探针源码及构建脚本。随包提供可加载的 `fatbin`、参考和完整性清单。

服务端、数据库凭据、签名私钥及未公开的专项深度诊断扩展不在用户仓库。详细边界见 [OPEN_SOURCE_BOUNDARY](docs/OPEN_SOURCE_BOUNDARY.md)。

用户目录不包含维护者实测报告、内部日志、数据库服务、私钥或私有深度诊断源码。常规探针的构建源码则随客户端公开。

## 安装

使用已有的 CUDA PyTorch 环境。基础客户端新增依赖只有 NumPy 与 cryptography；工具不重装 Torch、不安装驱动、不改频率、电压或功率。Python 客户端没有 Cython、`.so` 或 `.pyd` 必装模块，可从源码安装，也可安装通用 Python wheel。随包的原生 GPU 探针已编译为 `fatbin`，正常使用不需要 nvcc；需要修改探针时可使用公开 CUDA 源码自行构建。

```bash
git clone https://github.com/INTOSYN/Aritheia_GPU_Test.git
cd Aritheia_GPU_Test
python -m pip install '.[llm,datasets]'
python -m computeproof doctor
python -m computeproof guided --device cuda:0
```

从旧保护核心版升级时建议使用新环境。旧环境中的 `.so`/`.pyd` 可能遮盖同名公开 `.py`；如需继续使用旧环境，应先显式卸载旧 `computeproof-core`，再重新安装本版客户端。程序会检查这种混装，不会静默替用户移除已有包。

当前请使用上述源码安装方式。仓库中的 Linux wheel 安装脚本需在正式 Release 发布并配置下载地址及 SHA-256 后使用；本次源码发布不代表 wheel 或可选数据包已上线。

### Linux

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m computeproof doctor
python -m computeproof check --device cuda:0 --out runs/linux-card-001
xdg-open runs/linux-card-001/report.html
```

### Windows PowerShell

Windows 需要 64 位 Python ≥3.10、NVIDIA 驱动和可用的 CUDA PyTorch 环境。在下载并解压源码后进入项目根目录安装，不需要等待 Windows 专用常规检测核心 wheel：

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install ".[llm,datasets]"
python -m computeproof doctor
python -m computeproof check --device cuda:0 --out .\runs\windows-card-001
Start-Process .\runs\windows-card-001\report.html
```

也可以使用随源码提供的 [PowerShell 安装脚本](install-windows.ps1)。它默认安装本地源码到独立环境，复用指定 Python 的 CUDA PyTorch，不自动安装驱动或 Torch；不加 `-Run` 就不启动 GPU 检测：

```powershell
.\install-windows.ps1 -Python python
# 明确启动引导式检测；可选数据下载和深测仍分别遵循各自同意流程
.\install-windows.ps1 -Python python -Run
```

后续发布通用 wheel 后，脚本也支持同时指定 `-WheelUrl` 和 `-WheelSha256` 校验下载，不接受没有摘要的下载包。

这是 Windows 源码安装路径，不是 Windows 实卡兼容性认证。当前版本及脚本尚未完成 Windows GPU 全流程测试；驱动、CUDA PyTorch、原生探针加载、子进程和报告路径须在实际环境核验。不支持的功能应报告未完成，不能算作通过。当前没有可承诺下载的 Windows 私有深测二进制。

修改或重编译原生探针会改变其构建身份，必须重新核验指令路径及对应参考，不能只更新文件摘要就沿用官方探针的验证结论。

退出码 0 表示本次要求的项目全部完成且未观察到异常，1 表示检出数值异常，2 表示未完成，用户中止为 130。驱动错误、缺少资产或不支持的执行路径会得到 `INCOMPLETE`，不会记作通过。

### 多卡

每张 GPU 使用独立子进程和结果目录；一张卡失败不会覆盖另一张卡的结果。

```bash
python -m computeproof check --all-devices --parallel-gpus 2 --out runs/all-cards
python -m computeproof check --devices cuda:0,cuda:1 --out runs/two-cards
```

## 十二个科研场景

| 场景 | 研究方向 | 固定数据 | 安装方式 |
|---|---|---|---|
| 药物候选批量评分 | 计算药物发现 | 1,017 个化合物 | 内置 |
| 药物性质模型微调 | 分子模型训练与优化 | 同一数据，短程 AdamW 更新 | 内置 |
| 分子相似性检索 | 化学信息学与候选筛选 | 4,991 个分子 | 内置 |
| 乳腺细胞学特征分类 | 智能细胞学分析 | 569 样本、30 特征 | 内置 |
| 视觉特征编码与缓存 | 视觉表征与特征复用 | 1,797 张数字图 | 内置 |
| 基因组序列分类 | 剪接位点识别 | 3,190 条序列 | 内置 |
| 医疗影像分类 | 智能医疗影像研究 | BreastMNIST 780 张图 | 内置 |
| 材料性能预测 | 超导材料筛选 | 21,263 条材料记录 | 内置 |
| 单细胞图谱分析 | 单细胞组学 | 2,638 个细胞 | 原始仓库约 24.7 MB，下载后预处理 |
| 生物医学知识检索 | 医学文献检索与 RAG | 5,183 文档、300 查询 | BEIR 原始仓库约 2.8 MB，下载后预处理 |
| 大语言模型结构化信息理解 | 科学信息抽取 | 32 个固定任务 | Qwen 官方 Qwen3.5-4B，约 9.35 GB，单独同意后下载 |
| 大语言模型智能体决策 | 工具选择与任务决策 | 32 个固定任务 | 同上，仅下载一次 |

每个场景有两档预设配置。四个可选场景直接使用原始数据或模型仓库，不依赖本项目 Release 数据包。交互式 `guided` 的可取消 10 秒倒计时只授权 PBMC3k 与 SciFact 数据下载；Qwen3.5-4B 另行询问，默认拒绝，沉默或超时不视为同意。无人值守使用 `--accept-download` 授权数据下载，另用 `--accept-model-download` 授权模型下载；前者绝不包含后者。两个语言场景只支持官方未量化 Qwen3.5-4B 的固定版本，以 BF16、关闭 thinking、eager attention、无 KV cache 进行选项评分；不支持其他模型、量化版或 adapter，不会偷偷换成小模型。下载完成后同一次运行继续相应场景，拒绝下载不会阻止其他场景。已存在且通过校验的模型可离线复用。

安装 `.[llm,datasets]` 提供 Transformers、单细胞读取和检索预处理依赖；安装依赖不下载模型。约 9.35 GB 是模型文件下载大小，不是显存需求。显存不足或不支持 BF16 时如实记录未完成。数据预处理的版本和输入摘要会保存；不同环境重新生成的特征不能当作旧实验的逐位参考。任务分数不用于直接认定 GPU 故障。

十二个独立文件夹与运行方式见 [源码入口表](docs/OPEN_SOURCE_BOUNDARY.md)。例如只运行公开场景、不联网也不上传：

```bash
python -m agrel_public.scenarios.drug_screen --device cuda:0 --out runs/drug-only
```

这个独立入口仅执行场景，不代替整套精确探针；缺失可选资产需先下载，推理本身不联网。

```bash
# 八个内置应用及其探针
python -m computeproof check --large-scenarios skip --out runs/light
# 12 个场景的精确探针，不需要可选大包
python -m computeproof check --scenarios all --probes-only --out runs/probes
# 经授权下载资产并运行全部场景
python -m computeproof check --scenarios all --accept-download --accept-model-download --out runs/full
```

各场景的科学问题、具体算法和解释边界见 [SCENARIOS](docs/SCENARIOS.md)，文件体积见 [DATA_SIZES](docs/DATA_SIZES.md)，可选包规则见 [OPTIONAL_ASSETS](docs/OPTIONAL_ASSETS.md)。

## 数值判断

探针把每个非零乘积限制为 ±1，每个输出最多累加 128 个非零乘积。相关输入、乘积、部分和与输出在声明的 BF16/FP32 普通乘加路径中可精确表示，因此判断不依赖经验误差阈值。

完整矩阵由受测 GPU 计算。客户端核对预先保存的输入摘要、整矩阵摘要和每 4096 个输出值的分块摘要；不使用 CPU GEMM，也不调用另一张 GPU 回算。68 个矩阵探针各运行 8 个固定种子，另检查 128 MiB 显存字节往返；原生 BF16 MMA 与 FP32 FMA 探针记录逻辑 SM 覆盖。

```bash
python -m computeproof reference verify
python -m computeproof report runs/card-001
python -m computeproof replay runs/card-001/card-1/probes/probe-001
```

这些检查证明的是本次声明预算内是否观察到不一致，不是显卡永久健康认证。场景分数用于显示应用影响，精确探针负责数值判定；两者分别报告。完整数学条件见 [MATH_AND_EVIDENCE](docs/MATH_AND_EVIDENCE.md)。

## 联网、离线和研究贡献

软件可以完全断网运行。联网时默认发送每卡不超过 16 KiB 的研究摘要：安装范围设备哈希、GPU 型号、显存与计算能力、测试前频率、Torch/CUDA/驱动、构建与参考摘要、请求预算及 24 档配置的选择/完成范围。正常样本也保留这些范围，避免只收失败样本或混淆不同版本。默认不收场景数值、排名、用户数据或完整张量；异常/未完成才附加探针不一致计数及逻辑 SM 诊断。场景 `MONITOR_ALERT` 不会写成 `ALL_PASS`，但也不直接等同于硬件故障。

`--telemetry off` 关闭研究记录；`--offline` 禁止检测程序的一切网络行为。这两个控制项默认不显示在 `--help`，只在本 README 说明。`--offline` 不会改变 pip 自身的联网行为。

详细研究 JSON 先写入运行目录的 `contribution/`。只有用户查看内容、可选填写姓名或邮箱并最终确认后才上传：

```bash
python -m computeproof research preview-detail runs/card-001
python -m computeproof research upload-detail runs/card-001 --confirm SHA256
python -m computeproof research withdraw runs/card-001
```

设备哈希是同一安装内可关联的假名标识，不能称为完全匿名。原始 GPU UUID、主机名、用户名、用户文件、完整反例数组和自由文本日志不上传。字段、期限、撤回和服务器容量说明见 [PRIVACY_AND_CONSENT](docs/PRIVACY_AND_CONSENT.md)。

## 异常后的可选深测

维护者可针对已接收的异常/未完成报告分配有签名的 case。未完成本身不等于硬件故障。客户端只信任预先安装的发行公钥，不信任服务器临时给出的密钥；case 绑定原报告及同一安装下的原显卡，并限制有效期、重复次数、内存估算与超时。

常规检测与数据包不能自行加载私有扩展。专项深测可以使用已安装的公开后端，或在用户查看包的发行者、用途、适配平台、大小和摘要并确认后，另行下载签名绑定的编译扩展。私有扩展必须匹配操作系统、处理器架构和 CPython ABI；不会通过 `pip install`、任意 Python/Shell 脚本或安装钩子替代这一流程，也不改变驱动、功率或频率。加载原生扩展仍是在本机执行维护者提供的代码，并不是安全沙箱，用户应自行判断是否信任发行者。

下载、执行、结果上传分别确认；普通摘要默认上传和可选场景下载授权都不能代替深测授权。流程为：获取签名说明 → `deep preview-download` → 确认下载 → `deep preview` → 确认执行 → 预览结果 → 确认上传。具体参数与命令见 [DEEP_CASES](docs/DEEP_CASES.md)。深测是后续单独实验，不能倒写成原自然任务中的错误证据。编译与签名分别用于提高直接取得源码的成本和验证来源，不保证不可逆向或数值正确。

当前客户端提供该分发流程，不代表已有适配所有平台的私有深测包或已经部署生产服务。没有匹配且通过验证的编译包时，只运行公开检查，不自动安装其他后端。

## 支持范围与许可

主要面向支持原生 BF16 + FP32 常规训练的 NVIDIA GPU，例如 RTX A6000、RTX 4090、RTX PRO 6000 Blackwell、A100/A800、H100 和 B200。30、40、50 系列中具备相应计算能力、且本机驱动与 PyTorch 支持的卡可以运行；具体架构或算子不支持时明确返回未完成。

常规客户端、数值比较、公开探针及 12 个场景源码按根目录 [LICENSE](LICENSE) 分发。数据与模型遵循各自许可，见 [THIRD_PARTY](src/agrel_public/assets/scenarios/licenses/THIRD_PARTY.md)。另行分发的私有深测扩展须在发布前明确其许可并完成相应审查；移出公开目录或编译本身不改变已有授权，此前已发布的 GPL 源码权利不受影响。ComputeProof 是科研计算可靠性工具，不是临床工具或硬件厂商认证。
