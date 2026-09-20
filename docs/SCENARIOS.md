# 十二个科研计算场景

显卡不崩溃、训练能收敛、分数甚至更好，都不足以单独证明计算正确。ComputeProof 把这个问题放进 12 个可以实际运行的小项目：从分子排行榜到语言模型的工具选择，观察数值变化怎样抵达应用结果。场景在 BF16 自动混合精度下运行（累加、loss、softmax、优化器状态和输出处理为 FP32）；场景指标、非有限值检查和精确探针分别报告，不把"应用成绩下降"和"硬件故障"画等号。

每个场景两档冻结配置。批量场景：batch 8 与 512/1536；语言场景：short 与 long 上下文。配置决定代表性精确探针的 GEMM 形状：MLP 场景为 `[b,nin]×[nin,512]`、`[b,512]×[512,256]`、`[b,256]×[256,nout]`；检索场景为 `[b,dim]×[dim,docs]`；语言场景保留历史通用精确探针：维度 576 / 1536 / 9 × 64 / 49,152、名义长度 256 / 1024，包括一个 `bmm`。这些探针不表示 Qwen3.5-4B 的网络结构，也不证明其全部路径正确。这些是代表性形状，不保证与实际 token 长度、尾批次或融合路径一一对应。旧记录中读表提示约 281/1361 token，智能体约 179–185/1331–1337 token；实际长度随输出 detail 记录。

## 可用性分层

| 层 | 场景 | 处理 |
|---|---|---|
| 随包 `bundled` | drug_screen, drug_finetune, molecule_neighbors, cytology, ocr_cache | 冻结数据 + 模型已在 wheel 内（约 6 MB），来自 Aritheia 0.3.0 归档，SHA-256 记录在 `assets/scenarios/ASSET_MANIFEST.json` |
| 内置 `frozen_asset` | genomics_splice, medical_ultrasound, materials_screen | 与 0.3.5 相同的内置场景；冻结数据/模型已从参考环境导入并随 wheel 发布（公开副本去除了准备/训练设备标识）。若缺失则报告 `FROZEN_ASSET_MISSING`，不会用合成数据或重新训练替代 |
| 可选 `optional` | singlecell_neighbors, biomed_rag, llm_biomed_tables, agent_tools | 从 PBMC3k、BEIR、Qwen 原始仓库下载；模型下载必须单独明确同意 |

未就绪的场景不会启动，也不会被悄悄删掉：`report.json` 的 `selection.excluded` 列出每个未运行场景及原因。

## 场景说明

**药物候选批量评分（drug_screen）** 把 2048 位 Morgan 指纹交给冻结的小型回归 MLP，预测 RDKit FreeWilson 示例中 1,017 个化合物的 Act 值，再观察候选排序和 RMSE。它展示“分数看似更好也可能伴随底层数值异常”这一反直觉现象；不是完整 ChEMBL，也不是临床验证的药效预测。

**药物性质模型微调（drug_finetune）** 从同一份冻结初始化出发做 8 步 AdamW 更新，保存 checkpoint，再评估。它检查训练环节的数值是否进入参数，不冒充完整模型研发。

**分子相似性检索（molecule_neighbors）** 在 4,991 个有效 NCI 分子中用 Tanimoto 相似度为固定查询寻找 top-10 近邻；没有药效标签。它观察细微数值变化能否改变候选名单，即使程序从未报错。

**乳腺细胞学特征分类（cytology）** 569 条样本、30 个细胞学特征的表格分类基线。它同时保留“底层检查有变化、最终类别仍稳定”的阴性结果；不是切片分析，不提供临床诊断。

**视觉特征编码与缓存（ocr_cache）** 处理 1,797 张 8×8 手写数字，流程为 GPU 编码 → 主机缓存 → 同一 GPU 的 BF16 分类头。它检查设备计算、主机缓存和再次加载组成的完整链路，并展示底层异常与最终答案之间可能隔着多个环节。

**基因组序列分类（genomics_splice）** 3,190 条 60 碱基序列（UCI splice），固定编码 240 维，按重复序列分组划分，区分两类剪接边界及其他序列。

**医疗影像分类（medical_ultrasound）** BreastMNIST 780 张 28×28 图像，保留官方 546/78/156 划分的小型基线；教学与测试工作负载，不是临床超声诊断。

**材料性能预测（materials_screen）** 使用 21,263 条材料记录和 81 个特征预测超导临界温度，按化学式分组划分。它专门展示仍为有限数值的巨大偏差，说明只检查 NaN/Inf 不够。

**单细胞图谱分析（singlecell_neighbors）** 检查 PBMC3k 2,638 个细胞 × 1,838 个特征的近邻关系稳定性。它关注局部邻居是否变化；这种变化本身不等于细胞类型结论改变。

**生物医学知识检索（biomed_rag）** 使用 SciFact 5,183 篇文档、300 个查询、官方相关性标注和 512 维表示，只测 RAG 的检索环节。它同时比较汇总 Recall 与具体证据列表，避免汇总指标掩盖个别检索结果的变化。

**大语言模型结构化信息理解（llm_biomed_tables）与智能体决策（agent_tools）** 各有 32 个冻结任务，使用同一官方 Qwen3.5-4B 固定 revision，以 BF16、关闭 thinking、eager attention、无 KV cache 做受约束的 A/B/C/D 选择；不执行外部工具，需要 `transformers`（`pip install '.[llm]'`）。工具同时观察选择、分数和非有限值；选项评分不能包装成可靠医学助手或生产 Agent 的性能结论；任何选项出现 NaN/Inf 的题目不产生有效选择。

## 资产体积

| 资产 | 数据 MB | 模型 MB |
|---|---:|---:|
| 随包八场景合计 | 约 6.1 | 约 9.1 |
| genomics_splice / medical_ultrasound / materials_screen | 0.15 / 0.73 / 4.41 | 0.95 / 1.98 / 0.65 |
| singlecell（PBMC3k 原文件） | 24.7 | — |
| literature（SciFact 原 ZIP） | 2.8 | — |
| language（Qwen3.5-4B 官方文件） | 0.35（随包输入） | 9342.8 |

MB = 1,000,000 字节；可选项列出原始下载体积，不等于显存需求。PBMC3k / SciFact 下载后在本地预处理，处理版本与输入摘要随报告保存。

## 怎样把它做成自己的项目

先选一个真正感兴趣的场景，保留环境、配置、`report.json` 和 `outputs.npz`；再改变一个变量（batch、精度、软件版本），解释观察到了什么、有哪些证据、结论覆盖到哪里。完成后可以据实描述："构建 BF16/FP32 工作负载的 GPU 数值可靠性评估流程，完成某场景的配置对照与精确探针分析"。只运行一次命令不等于完成药物研发、临床系统或大型 Agent 平台。

场景指标是可解释的应用观察，数值探针才负责硬件异常判定。一个场景的最终指标不变不代表对应计算路径正确；指标改善、恶化或保持不变都必须结合独立探针阅读。逐场景文件体积见 [DATA_SIZES](DATA_SIZES.md)，源码入口见 [OPEN_SOURCE_BOUNDARY](OPEN_SOURCE_BOUNDARY.md)。
