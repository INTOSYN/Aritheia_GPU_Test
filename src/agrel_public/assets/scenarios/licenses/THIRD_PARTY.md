# 第三方材料归属

本工具新编写代码采用 GNU GPL v3 或更高版本（见包根目录 LICENSE）；数据、依赖软件及下载模型按各自来源许可，不由 GPL 覆盖。正式公开分发前，维护者应复查原始数据许可版本及任何派生物义务，不应删去此说明。

RDKit示例文件取自本地官方RDKit 2025.09.4安装包，准备记录与 SHA-256 位于 assets/scenarios/data/prepared 下各 metadata.json，安装包清单为 assets/scenarios/ASSET_MANIFEST.json。RDKit软件BSD许可不自动取代原始化学数据库的数据权利。本包药物场景使用 RDKit FreeWilson 示例；旧文本中 CHEMBL2321810 的描述不适用于该冻结场景。ChEMBL 派生的其他记录仍应按各自来源署名。ChEMBL 数据，请遵守ChEMBL的CC BY-SA 3.0 Unported数据条款（https://chembl.gitbook.io/chembl-interface-documentation/about）及原始出处；本包保留SMILES、Name、Act，不伪造新实验测量。NCI first_5K 的处理数据保留原始标识与来源；本公开包提供冻结 NPZ，不声称附有完整上游原文件，重新分发需遵守其来源条款。

scikit-learn数据接口/软件：BSD-3-Clause，原软件COPYING已附。Breast Cancer Wisconsin Diagnostic原始数据应署名Wolberg、Street、Mangasarian及UCI仓库；UCI数据页提供CC BY4.0。Digits原始光学数字识别数据署名E. Alpaydin、C. Kaynak及UCI；保留官方数据说明文件。

以下提供经过处理的冻结数据；前三种为可选包，UCI splice、superconductivity 与 BreastMNIST 随基础包发布：PBMC来自10x健康供者数据，依10x公开数据CC BY4.0和Scanpy处理版本归属；BEIR SciFact数据卡为CC BY-SA4.0；UCI splice和superconductivity仓库提供CC BY4.0；BreastMNIST依MedMNIST/BUSI的数据归属和CC BY4.0。完整论文/原作者说明以数据卡为准。使用不等于临床许可。

Qwen3.5-4B 下载模型：Apache-2.0，直接从 Qwen 官方仓库取得固定 revision 的文件，包含 LICENSE；逐文件摘要见可选资产清单。模型版权不由本工具重新授权。未随包提供任何字体文件。

新编写的32条读表和32条工具路由模板为本工具fixture；底层Act记录仍保留其数据来源义务。处理NPZ是本工具生成，不能因此抹除底层数据的许可。基线权重仅供研究，正式商业/公共发布前审核训练数据相应条款。
