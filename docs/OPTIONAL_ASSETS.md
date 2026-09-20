# 轻量安装与按需资产

`pip install` 得到程序、冻结精确参考和八个内置场景的冻结数据/模型（体积见 [DATA_SIZES](DATA_SIZES.md)）。四个可选场景的大资产独立放在发布地址，不进入 Git 源码历史。

## 测试前选择

推荐新用户使用 `guided`：交互模式显示 10 秒可取消倒计时，后台下载三个已校验包，八个内置场景先计算，随后同次会话继续四个就绪场景。两个语言场景共用一次下载。已有资产直接复用，不重复运行内置场景。无人值守仍需 `--accept-download`；没有正式地址则不启动下载。缺依赖不自动修改用户 Python 环境，先安装 `.[llm]`；一键安装脚本在自己的环境内包含该依赖组。

交互终端运行 `check` 时，先列出本次内置场景，再逐项询问是否加入四个可选场景（默认否）。选中且资产缺失时，显示资产包名称、实际体积、许可与"SHA-256 已钉住"，再单独确认下载。拒绝下载时该场景记为 `DOWNLOAD_DECLINED`，不会启动缺数据的检测，也不会偷偷删除已选场景。

脚本默认运行八个内置场景，不等待交互；也可以显式选择：

```bash
# 只运行内置场景
python -m computeproof check --large-scenarios skip --out runs/light
# 只运行一个可选场景；仅在资产缺失且授权后下载
python -m computeproof check --scenarios biomed_rag --accept-download \
    --asset-release-url https://github.com/OWNER/REPO/releases/download/TAG --out runs/retrieval
# 全部十二个场景
python -m computeproof check --large-scenarios include --accept-download \
    --asset-release-url https://github.com/OWNER/REPO/releases/download/TAG --out runs/full
# 只下载，不检测
python -m computeproof assets download language --asset-release-url https://... --accept-download
```

`--accept-download` 只授权所选场景缺失的资产，不授权任何上传。发布地址也可写入配置 `asset_release_url` 或环境变量 `AGREL_ASSET_RELEASE_URL`；只接受 HTTPS。OWNER/REPO/TAG 是待配置的真实地址，源码不包含虚构或未经核验的公共地址。

## 下载与复用

仅使用 Python 标准库。ZIP 先写入资产根目录内的临时目录，核对声明大小、整包 SHA-256、成员列表、每个成员的大小与 SHA-256，全部通过后才移入 `~/.aritheia/assets`（或 `ARITHEIA_ASSETS`）。失败清理临时目录；已存在且内容相同的文件跳过，内容不同的文件绝不覆盖（需另选资产根）。成员路径只允许 `data/prepared/...` 与 `models/...`，拒绝 `..` 与绝对路径。两个语言场景只下载一次共享包。

**清单未钉住或没有发布地址时拒绝下载。** 随包 `assets/optional_assets.json` 已钉住三个包（singlecell 35.2 MB、literature 10.5 MB、language 214.8 MB）的 ZIP 与逐成员 SHA-256；`release_url` 在维护者上传到正式 Release 前为 `null`，用户此时看到 `ASSET_RELEASE_URL_MISSING`，不会发起任何网络请求。literature 包只含运行所需的 `data.npz` + `metadata.json`，不含重做准备才需要的 LSA 变换文件。

## 从 ZIP 离线安装

显式运行以下命令，与在线下载使用相同的整包、成员和逐文件摘要核验。不会联网，也不会覆盖不同内容的已有资产。

```bash
python -m computeproof assets install language /path/to/language.asset.zip
python -m computeproof assets install singlecell /path/to/singlecell.asset.zip
python -m computeproof assets install literature /path/to/literature.asset.zip
```

只运行 `check --probes-only --scenarios all` 不需要这些包。语言包含 Apache 2.0 许可证全文；两个语言场景共用这份模型。显式请求的应用缺少包时，整次应用计划为未完成，已完成的探针结果仍可查看。
