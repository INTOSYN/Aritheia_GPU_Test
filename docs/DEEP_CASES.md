# 用户确认的签名深测

普通场景、数值对比和基础探针公开源码，不依赖深测扩展。异常或监控告警只是跟进线索，不自动证明显卡损坏。只有原报告为 `ATTENTION_REQUIRED`，维护者才可分配与该报告绑定的深测 case。深测是与原自然任务分开的新运行。

有两种 case：`case.v1` 只包含签名数据，使用公开的固定矩阵对比实现；`case.v2` 另外描述一个经签名指定的编译扩展。v2 的下载、执行、结果上传必须分别确认。拒绝深测不影响普通检测；没有匹配当前系统与 Python ABI 的编译包时，深测不可用，不表示显卡异常。

## 使用

维护者须先配置正式发行公钥并为该报告分配 case。没有分配时返回空列表，未配置可信公钥时拒绝运行。

```bash
# 仅获取原报告获分配的 case；使用本机回执鉴权
computeproof deep fetch runs/original

# v2：先预览下载地址、大小、文件哈希、平台和原生代码风险
computeproof deep preview-download runs/original --case runs/original/deep-cases/CASE.case.json

# v2：只下载这份编译包，不执行、不上传；使用上一步的摘要
computeproof deep download runs/original --case runs/original/deep-cases/CASE.case.json \
  --confirm DOWNLOAD_PREVIEW_SHA256

# 查看目的、目标 GPU、预算、到期时间和执行确认摘要
computeproof deep preview runs/original --case runs/original/deep-cases/CASE.case.json

# 只授权在原 GPU 上运行；使用上一步 SHA，新输出目录不可覆盖
computeproof deep run runs/original --case runs/original/deep-cases/CASE.case.json \
  --device cuda:0 --out runs/deep-001 --confirm PREVIEW_SHA256

# 结果先留本地；需要贡献时另行预览和确认
computeproof deep preview-result runs/original --case runs/original/deep-cases/CASE.case.json --out runs/deep-001
computeproof deep upload-result runs/original --case runs/original/deep-cases/CASE.case.json \
  --out runs/deep-001 --confirm RESULT_SHA256

# 撤回原报告时，同时撤回其在线深测结果
computeproof research withdraw runs/original
```

v1 数据 case 跳过 `preview-download` 和 `download`。两个预览的 SHA-256 不同，不能把下载许可当成执行许可。下载过同一份文件也不代替对该 case 的确认。

GPU 序号可能变化，因此通过安装范围 HMAC 绑定原卡；不能确认身份则不运行。两张相同型号不是同一张卡的证明。公开调度流程不调整硬件设置，不运行 pip、安装脚本、服务器指定命令或任意模块名。

**v2 会执行可信发行者提供的原生代码，拥有当前用户进程的权限。** 签名和哈希证明发行者与文件身份，不证明代码无害；独立工作进程和超时控制不是安全沙箱。用户需要信任发行者对该二进制的审核。发布者必须确保实现遵守显示的目的、参数和不改硬件设置要求。

## 限制与可查看内容

case 外层与参考包分别验签。限定三类目的、最多四个分配 case、最多 30 天有效期；单次超时 30–1800 秒、声明探针内存预算 16–2048 MiB、每探针最多 32 次。内存预算是检测后端必须遵守的分配前估算，不是操作系统强制的内存限制，也不是 CUDA 上下文、PyTorch 缓存及所有进程合计显存的硬上限。

v2 外层签名覆盖编译包 URL、压缩大小、解压后大小、SHA-256、全部文件清单以及 OS/架构/CPython ABI。单包下载上限 128 MiB、解压后上限 256 MiB、最多 64 个文件。清单只允许一个固定名 `agrel_detector_core` 编译扩展和 `.json/.bin/.fatbin/.cubin/.npz` 资源；不接受 Python 源码、安装脚本、额外共享库、嵌套路径、符号链接或清单之外的文件。它不是可以交给 pip 的 wheel。

文件保存到原运行目录的 `deep-binaries/<archive-sha256>/`，按哈希区分且不会自动替换已存在内容。缓存被改动时拒绝使用，不能静默重新下载掩盖修改。运行前，独立工作进程再次验证报告绑定、外层签名、两个确认记录、平台、文件清单和所有哈希，再从指定绝对路径加载扩展；不会从 Python 搜索路径寻找同名模块。普通签名 pack 即使写 `private_core_v1`，也只能得到 `EXPLICIT_DEEP_CONSENT_REQUIRED`，不能触发已安装的同名扩展。

下载许可保存到 `deep-cases/<case-sha256>.download-consent.json`，运行许可保存到新运行目录的 `consent.json`；结果与原报告、case 摘要、随机 attempt ID 绑定。中断保留已观察到的失败事件，但不冒称整套完成。case 到期后不能发起新运行；已在有效期内产生的结果仍可单独预览和确认上传。不自动上传原始张量、NPZ、stdout/stderr 或运行目录。

可选上传最多 512 KiB 未压缩白名单 JSON：case/原报告摘要、attempt ID、完成状态、已观察到的合同失败事件、逐探针输出/参考摘要与比较结果。不支持结果结构会拒绝而非偷偷丢弃。接收后的记录仍为 UNVERIFIED_CLIENT_REPORT；签名证明发行来源，不证明用户硬件或执行过程真实。

服务端可为不同报告分配不同的经审核 case，但仅获取 case 元数据不会下载或执行新二进制。生产使用仍需配置可信发行公钥、正式 HTTPS 下载位置，并对每个目标平台完成构建和实卡验证。软件协议测试不代替这些验证；当前源码中的构建辅助程序也不代表已有经验证的私有深测实现。

原始完整残差或特定任务私有数据不在此自动收集功能内，需要独立的研究协议与逐项授权。
