# PaperScout

PaperScout 是一个面向 AI 研究与技术调研的开源论文研究 Agent。它不仅搜索和总结论文，还会围绕研究任务、数据集、指标、模型规模、训练预算与实验条件判断论文是否真正可比，并将结论、原始证据和引用关系保存在可追溯的研究状态中。

项目的目标不是替代论文阅读，而是减少检索、筛选、条件核对和研究决策中的重复工作，尤其避免把“实验条件不同”误判为“论文结论冲突”。

## 项目亮点

- **对话式研究流程**：先澄清简称、领域、时间范围、算力、数据集和代码要求，任务明确后自动开始研究。
- **arXiv 优先检索**：面向 AI 领域，支持中文问题转英文检索、相关度／时间／引用量排序，以及 1–20 篇论文数量控制。
- **证据驱动总结**：主要发现由模型综合生成，每条结论保留证据编号；缺少证据时明确降级，不伪造确定性答案。
- **五维可比性分析**：检查任务、数据集与划分、指标定义、模型规模与训练预算、实验条件。
- **可靠冲突门禁**：只有论文条件可比时才允许判定支持或冲突；否则标记为“条件不一致”或“证据不足”。
- **研究决策报告**：输出推荐阅读、方法与实验对比、适用边界、局限性、初步复现准备度和下一步建议。
- **项目知识库**：支持 Markdown、PDF、PPTX 和 TXT 批量异步入库、删除、导出、报告归档和项目内持续检索。
- **中英文界面与报告**：界面一次只显示一种语言，可在右上角切换；报告直接渲染为笔记并支持导出 PDF。
- **模型职责拆分**：内置模型负责需求理解，论文事实抽取和综合分析可使用独立 OpenAI-compatible 研究模型。

## 工作流程

```text
用户对话与约束澄清
        ↓
中文检索词解释与 arXiv 搜索
        ↓
候选论文排序与证据提取
        ↓
结构化事实抽取
        ↓
五维可比性门禁
        ↓
支持／冲突／不可比较判断
        ↓
研究决策报告与项目归档
```

外部 arXiv 论文只缓存题录、摘要和来源链接，不会自动下载 PDF，也不会自动进入用户知识库。只有用户主动收藏的论文、上传资料和生成报告才会进入项目空间。

## 快速开始

环境要求：Python 3.11+。

```bash
git clone https://github.com/exuusiai/paperscout.git
cd paperscout
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
paperscout serve --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/` 即可使用 Web 界面。

命令行示例：

```bash
paperscout health
paperscout ingest papers/example.pdf --corpus data/corpus.sqlite
paperscout search "retrieval hallucination" --corpus data/corpus.sqlite
paperscout ask "哪些方法可以降低 RAG 幻觉？" --corpus data/corpus.sqlite
```

## 模型配置

默认模型使用本机 OpenAI-compatible Chat Completions 接口：

```bash
PAPERSCOUT_MODEL_BASE_URL=http://127.0.0.1:8001/v1
PAPERSCOUT_MODEL_NAME=Qwen/Qwen3-8B
PAPERSCOUT_MODEL_API_KEY=EMPTY
PAPERSCOUT_USE_MODEL_REASONING=true
```

可以为论文分析单独配置研究模型：

```bash
PAPERSCOUT_RESEARCH_MODEL_BASE_URL=https://example.com/v1
PAPERSCOUT_RESEARCH_MODEL_NAME=your-research-model
PAPERSCOUT_RESEARCH_MODEL_API_KEY=your-api-key
```

内置模型端点严格限制为 loopback 地址。独立研究模型必须显式配置为以 `/v1` 结尾的 HTTPS OpenAI-compatible 接口，调用路径为 `/chat/completions`；项目不使用 `/v1/responses`。不要把 API Key 写入代码、提交到 Git 或输出到日志。

## 数据与检索

本地检索默认使用 SQLite FTS5。安装可选依赖后可启用 BGE-M3、FAISS 和重排器：

```bash
python -m pip install -e '.[retrieval]'
paperscout index --corpus data/corpus.sqlite
PAPERSCOUT_RETRIEVAL_MODE=semantic paperscout ask "..." --corpus data/corpus.sqlite
```

下载并导入 SciFact、Qasper 数据：

```bash
python scripts/download_datasets.py --dataset scifact --split all
python scripts/download_datasets.py --dataset qasper --split train
paperscout ingest-jsonl data/raw/scifact/corpus.jsonl --corpus data/scifact.sqlite
```

## 评测

项目支持 Recall@K、Evidence Recall@K、MRR、消融实验，以及 Qasper 回答 F1、答案类型、证据 F1、延迟和 Token 用量评测。

```bash
paperscout evaluate evals/custom_topics/queries.jsonl --corpus data/corpus.sqlite
paperscout benchmark evals/custom_topics/queries.jsonl --corpus data/corpus.sqlite
paperscout ablate evals/custom_topics/queries.jsonl --corpus data/corpus.sqlite
python -m pytest -q
```

当前 Qasper 检索实验中，`top-k=10` 的证据召回率为 `0.6653`，`top-k=5` 为 `0.4849`。跨论文冲突评测与 SciFact 的 claim-evidence 标签严格分离；现有模型试标数据只能用于误报诊断，不能作为正式人工 ground truth 或冲突召回率依据。

## 当前边界

- arXiv 路径目前主要依赖题录和摘要，尚未完整恢复 PDF/TeX 中的表格、公式、图表和附录。
- 复现准备度对代码仓库、License、硬件和依赖健康度采用保守策略；未知信息会被扣分，不会假装已经验证。
- 知识库当前采用单机线程池和 SQLite，任务尚不能在进程崩溃后自动恢复。
- 项目隔离目前是存储边界，不等同于完整的多用户认证和授权。
- PII 处理、团队审计、人工冲突复核和研究主题持续监控仍在规划与开发中。

## 文档

- [系统架构](docs/architecture.md)
- [评测方法](docs/evaluation.md)
- [安全设计](docs/security.md)
- [已知失败案例](docs/failure_cases.md)
- [SciFact 检索对比](docs/scifact-retrieval-comparison.md)
- [开发记录](docs/development-log.md)

## 开源定位

PaperScout 适合作为 Agentic RAG、科研智能体、证据检索和可信生成方向的实验平台。项目强调可验证的工程闭环：检索结果可追踪、模型输出有证据、论文比较有条件门禁、失败和未知状态显式呈现。

欢迎通过 Issue 提交检索失败案例、可比性误判、公开评测数据和功能建议。
