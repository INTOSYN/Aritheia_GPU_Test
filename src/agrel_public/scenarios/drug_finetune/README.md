# 药物性质模型微调

场景 ID：`drug_finetune`。本目录中的 `SPEC` 是注册表的唯一参数来源，`execute` 是实际执行入口。

## 独立运行

```bash
python -m agrel_public.scenarios.drug_finetune --device cuda:0 --out runs/drug_finetune-new
```

执行两个冻结配置。共享模型、数据校验、计算、指标与输出实现在同层公开的 `engine.py`、`assets.py`、`llm.py`；不调用受保护检测核心完成场景计算。此入口不下载、不上传、不改频率。缺失资产可用 README 的 Release 资产安装命令补齐，或运行 guided 自动准备。

## 证据边界

完成与指标是任务观察，不是精确算术证明。结果保存独立内容/决策指纹、数据和模型绑定；跨设备/版本哈希不同不直接判硬件故障。不得把后来独立探针的异常附会为本次场景调用已出错。
