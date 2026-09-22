# GrantScout → 立项本子撰写平台 重构设计(定稿)

状态:已评审定稿。决策记录见文末;需求覆盖矩阵见第 8 节。

## 1. 定位

从"论文调研 agent"升级为"立项本子撰写平台"。产品分两种写作模式与一个平台层:

- **整稿生成(Phase 1,先行)**:按模板生成 20000 字级项目方案的整稿骨架
  (需求分析、国内外研究现状、系统设计、技术方案、指标分析),每节带证据引用,
  用户在工作台中逐节确认。复用现有检索栈(arXiv、SQLite FTS5、BGE-M3、重排器)。
- **Copilot 补全(Phase 2)**:编辑器内行内补全(前缀/查找/指令)、/命令、@ 过滤器、
  连续补全、引用跳转;前端形态为 Web 编辑器 → VSCode → WPS。
- **平台层(Phase 2 末–Phase 3)**:批量入库管线、多用户隔离、可观测、打包部署。

## 2. 已确认的决策

| 决策点 | 结论 | 理由摘要 |
|---|---|---|
| HTTP 框架 | **FastAPI 保留**,领域逻辑框架无关 | 70 个测试是重构安全网;1–3s 延迟预算下 litestar 性能优势不生效;pydantic v2 生态一致。/help 的 API 文档用 FastAPI 原生 /docs(Swagger)与 /redoc |
| 存储规模 | **T1 档(≤5千文档,≤5人):SQLite FTS5 + FAISS** | 百万级 chunk 下 bm25 毫秒级;向量 8–20GB 用 FAISS(HNSW)可扛;`SemanticIndex.search` 已是"SQL 过滤 → paper_id allowlist → 向量检索"的混合架构,@ 过滤器天然适配。接口按可替换设计,T2(Typesense/milvus-lite)、T3(milvus+pg)只换实现 |
| ETL | **不用 Airbyte**(评估后否决) | Airbyte 解决外部数据源同步;我们是文件→索引的进程内转换,用入库工作队列(线程池 + 状态表 + 启动重扫)等价实现,零部署负担。出现外部 DB 同步需求时再评估 |
| 延迟 | 行内补全 1–3s | L5 直接 vLLM(prefix caching 常开);L2 前缀/照抄补全零 LLM、亚秒级;不做极致优化 |
| /web 搜索 | provider 接口;jina.ai 为首个已接 provider;SearXNG 自托管为离线兜底;Perplexity 可选 | SearXNG 零 key 零配额,需在 settings.yml 开 `formats: [json]`;Bing API 已退役不考虑 |
| 部署环境 | 全部组件自托管于实验室线上机器 | vLLM、TEI(嵌入服务)、SearXNG、langfuse、prometheus 均可自托管,无云依赖 |
| 测试语料 | 私域=实验室论文;公域=arXiv 批量抓取(网络仿真/星地网络方向,500–1000 篇) | 用户例子(EMANE、StarryNet、ns-3、网络数字孪生)指示领域 |
| 模板 | 框架先行,内置五节占位模板;真实模板到位后填 TemplateSpec 数据即固化 | 需要用户提供:节结构、字数范围、写法要点/评审关注点、范文 |

## 3. Phase 1:整稿生成流水线(本阶段交付)

### 3.1 数据模型(新增 `src/grantscout/proposal/schemas.py`)

```text
ProposalTemplate ── sections: list[SectionSpec]   # key/标题/写法要点/字数上下限/是否需引用
IdeaBrief         # 选题:主题、目标、创新点、周期、约束、待澄清问题
LiteratureDossier # 文献卷宗 = 复用 ResearchState + GapStatement[] + ScienceQuestion[]
Outline           # 分节要点 + 字数预算 + 证据支撑,approved 人工确认位
SectionDraft      # 单节草稿:版本、markdown 正文、DraftCitation[]、source(agent/material)
ProposalReview    # 自检结果:引用计数、缺证据节、字数预算告警
ProposalDocument  # 顶层聚合,持久化到 runs/{id}.proposal.json + .md
```

### 3.2 流水线七阶段(`proposal/pipeline.py`,每步记录 ToolCall 可回放)

1. **澄清** → IdeaBrief(确定性抽取 + 待澄清问题;对话式澄清后续接入 /api/chat)
2. **文献卷宗** → 直接运行现有调研循环 `GrantScoutAgent.run()`,产出 Dossier
3. **研究空白凝练** → 模型从证据中提炼 2–4 条 GapStatement,每条必须挂证据 id
   (校验 id 合法性 + 文本重叠度,复用 evidence.py 的做法);无模型时降级为
   limitations 事实改写;再无 → 置信度 0 的占位 + 警告
4. **科学问题** → 从 gaps 转写,同样挂证据
5. **大纲** → 模板骨架 + 每节要点与字数预算(模型生成或证据回填),approved=False
   等人工确认
6. **分节撰写** → 每节独立:检索证据(节标题+选题为 query,top 8)→ 编号资料表 →
   模型写 markdown,引用标记【文献n】,校验 n 合法、剔除非法标记;无模型时降级为
   素材整理稿(source=material);查不到资料的节输出【待补充】占位
7. **自检 + 导出** → 引用核对(引用必须存在于卷宗)、缺证据节清单、字数预算对照;
   渲染 Markdown(全局统一重编号参考文献),持久化 JSON+MD

### 3.3 反幻觉硬约束(代码层,非 prompt 约定)

- 引用标记只允许指向本轮提供的资料表;非法标记剔除并计入告警
- 无证据的论断输出【待补充】占位,不生成貌似合理的引用
- 研究基础/用户成果节只接受用户上传材料,agent 拒绝生成(Phase 2 落地该节类型)
- 降级路径显式:material 稿、占位 gap、警告列表,绝不伪装确定性

### 3.4 模板系统(`proposal/templates.py` + `proposal/real_templates.py`)

注册表机制 + 六个已注册模板。官方原文归档于 `templates/`:

| 模板 id | 来源 | 章节 | 说明 |
|---|---|---|---|
| `lasa-key-project-2023` | 拉萨市重点科技计划申报书 | 10 可生成节 + 3 表格栏 | 叙述节限 500–1000 字;预算/人员/伦理表为人工栏目 |
| `most-innovation-2015` | 科技部创新方法工作专项申请书 | 8 可生成节 + 3 表格栏 | 课题/预算/人员表人工填写 |
| `jiangsu-erc-2023` | 江苏省企业工程技术研究中心申报书 | 6 节 | 工程中心/平台类,建设期 3 年 |
| `guangxi-collab-2025` | 广西"协同"行动计划申报书 | 11 节 | 含摘要/总体目标等严格字数上限节 |
| `universal-plan-v1` | 四份官方模板归纳的通用骨架 | 14 节 | 无指定申报格式时使用 |
| `project-plan-v1` | 初版占位模板 | 5 节 | 保留兼容 |

固化自真实模板的机制(来源 `templates/README.md` 使用建议):

- **人工栏目(generatable=False)**:预算表、人员表、伦理表等表格/签章栏,管线跳过,
  渲染时标注"由人工填写",不伪装成可生成内容
- **考核指标六字段结构化(structured_indicators=True)**:指标名称、目标值、测试条件、
  测试方法、验收材料、来源依据;指标表中的引用(来源依据列)同样登记进引用审计
- **先锁目录再逐章生成**:大纲确认门(approved)对应"锁目录";分节生成对应"逐章";
  自检阶段执行引用/指标/术语一致性检查
- **公私域标记**:"研究基础/单位情况"类节在 guidance 中硬性规定只接受用户上传的
  私域材料,agent 拒绝生成
- 模板可运行时注册(`register_template`/`register_real_templates`),新模板是纯数据

## 4. Phase 2:Copilot 补全(需求已全部并入,按依赖排序)

**状态:后端补全核心已实现(`src/grantscout/completion/`),含 /api/completion 端点。**
已落地:L0 缓存(TTL+LRU+会话过期标记)、L1 @过滤器解析(filename/time/author 生效,
tag/domain/language/impact/conference 解析但提示待元数据层)、L2 句子级 trigram 照抄
补全(`retrieval/sentences.py`,多长度片段匹配,零 LLM)、L2 词法 + L3 语义查找补全
(CJK 检索自动经 trigram 兜底)、L5 模型续写(本地图端点关闭思考模式,引用标记解析,
非法标记剔除)、/summarize /refine /find 命令(summarize 无模型时抽取式降级)、
全部候选带证据引用。真实模型实测:照抄补全亚秒级,LLM 续写约 1.1s(关闭思考后)。

**已补齐(2026-09-22):**元数据层(scope/venue/tags/cited_by_count)入库、
@{tag}/@{impact}/@{conference} 生效、公私域路由硬门禁(私域证据仅允许本地回环模型)、
引用带 scope 标注与原文跳转 URL。前端编辑器原型已上线 `/playground`(句读触发、
1s 防抖、AbortController 取消、候选采纳、引用跳转/预览)。源文件托管内置:
`/files/{path}` 受控静态服务 + `/viewer?file=&page=` 原生 PDF 页码定位,
配置 `GRANTSCOUT_FILE_SERVER_BASE_URL` 可切换外部 pdf.js。
PII 清洗(`privacy/pii.py`):规则层(手机号/身份证/邮箱/自定义敏感词)常开,
Presidio+spaCy zh NER 可选(缺失自动降级),私域文档入库即脱敏、原文保留。
chat 项目记忆:按项目持久化记忆卡(模型提炼,上限 100 条),
`/api/chat` 自动注入 + `GET/DELETE /api/projects/{id}/memory` 管理。
ThinkLab 8 篇论文作为 private 测试语料(`data/library/thinklab/`,gitignore 隔离)。
待做:VSCode/WPS 扩展、连续补全的前后端协同打磨、领域/语言元数据来源。

### 4.1 补全核心(L0–L5,决定 QPS 的三件事:触发逻辑、批处理/取消、缓存)

- **L0 缓存**:前缀哈希缓存 + 连续补全加速(接受后缓存命中资料与解码前缀);request-id
  过期取消(新击键 abort 上游 vLLM 请求)
- **L1 意图/过滤**:A 模板续写 / B 查找补全 / C 照抄补全 / D 指令;@ 过滤器解析
- **L2 词法**:句子级索引(入库时切句存 offset);Type C 照抄 = 短语查询最长前缀延续,
  零 LLM;Type B 前缀 = 搜索引擎 + 前缀匹配
- **L3 向量**:BGE-M3 + FAISS(服务化时换 TEI);Type B 查找补全走这里
- **L4 重排**:bge-reranker-v2-m3 + 连贯性打分
- **L5 生成**:vLLM(prefix caching);补全模型独立配置(可换小模型);公私域路由硬门禁
- **前端触发逻辑(核心)**:停顿 ≥1s 或句读触发;词边界触发(避免"论文集|合"式断裂
  token);批处理与任务取消
- **连续补全**:接受后锁定类型继续补全,前端 + L0 缓存协同

### 4.2 命令与过滤器

- /summarize(LLM+prompt,不引入 BART——中文摘要小模型需另评,先用现有模型)、
  /find(复用 RAG)、/chat(RAG as a service + 项目记忆,参考 LlamaIndex/LangGraph
  的记忆管理)、/web(jina.ai 已接;provider 接口 + SearXNG 兜底)、/arxiv(现有零 LLM
  实现 + GB/T 7714 与 BibTeX 引用格式渲染;具体格式清单待确认)、/refine(prompt)、
  /help(FastAPI /docs + ReDoc,视频与前端文档最后写)
- @ 全量过滤器:filename/tag/time/author/domain/language/impact/conference/unset;
  tag 需前端动态增删;元数据来源:入库时 OpenAlex 抓 venue 与 cited_by_count
  (impact 档位用引用数近似,精确 JCR 分区后续再议)

### 4.3 引用呈现(前端重点)

- chat 与 B/C 类补全全部带引用;引用 = 证据 id → 页码/行号元数据 → 原文跳转链接
  (源文件托管由用户自行完成);不做图像级 RAG
- 私域来源显式标注【私域:文档名 §页码】,区分成果来源

### 4.4 WPS / 文档格式

- **docx 解析与导出提前到 Phase 1 末**(本子交付物即 docx);老 .doc 用 LibreOffice
  headless 转换,标注为已知风险
- wpsjs 插件(段落样式列表注入 prompt、表格生成、图注/标注交叉引用——低优先级)Phase 2 末起
- xlsx 解析(openpyxl)Phase 2

### 4.5 Chat 模式与记忆

- 按项目持久化记忆:会话摘要 + 事实卡,存 knowledge.sqlite;一个文档属一个项目,
  一个项目多文档;基于项目文档范围讨论(复用 @ 过滤与 corpus_path)

## 5. 需求模块(入库管线)

- **格式**:md/txt/pdf/pptx(现有)+ docx(Phase 1 末)+ xlsx(Phase 2, openpyxl)
- **批量异步**:上传即建任务 → 工作队列线程解析 → 热入库(写库即查,FTS 增量同步);
  任务状态表 + 启动重扫(修复进程重启卡死);前端进度 UI 先用现有任务轮询接口 +
  工作台进度条,不引重型组件
- **易用性**:MVP 用 Web 拖拽上传(现有端点加前端页);VSCode 右键上传、alist 对接
  Phase 3;上传空间按用户隔离(Phase 2 与多用户 token 认证一起做)
- **自定义解析**:扩展现有 `register_parser` 注册表为配置驱动(配置文件声明后缀 →
  解析器/预处理脚本),函数接口保留
- **PII**:微软 Presidio 框架 + 中文 NER 引擎(spacy zh 或 HanLP)+ 中文规则
  (身份证/手机号/邮箱);私域入库时清洗,原文留私有区,索引存脱敏文本

## 6. 优化与评测

- 小模型分工:补全/摘要可配置独立小模型(research-model 拆分机制已有,加
  completion-model 档);embedder/reranker 微调脚本接入评测闭环
- 评测:ragas 本地方案 + API 评测可选(工作量小);北极星指标 = 补全采纳率 +
  引用有效率;整稿生成用 rubric 模拟评审(Phase 2 末)
- 推理加速:vLLM(补全)+ TEI(嵌入),成熟方案不自研

## 7. 平台与运维

- 打包:pip 包(现有 pyproject 完善)+ docker compose + k8s(Helm)
- 可观测:langfuse(LLM trace)+ prometheus(指标);补全审计表(Phase 2)作为
  两者的本地兜底;报警用开源组件(alertmanager)不自制
- 配置可视化:嵌入模型/重排器/向量库/LLM 端点全部走 Settings(已有),加 Web 配置页
- 多用户:简单 token 认证 + 项目归属,Phase 2

## 8. 需求覆盖矩阵

| 需求 | 阶段 | 落点 |
|---|---|---|
| md/pdf/pptx 解析入库 | 已有 | retrieval/parser.py |
| docx 解析与导出 | P1 末 | parser + 新 exporter(python-docx) |
| xlsx 解析 | P2 | parser 注册表 + openpyxl |
| 批量异步上传 + 热入库 | P1 末/P2 | knowledge.py 队列 + 状态表 + 启动重扫 |
| 上传进度 UI | P2 | 工作台 + 现有任务轮询接口 |
| 拖拽上传前端 | P2 | 工作台;alist/VSCode 右键 P3 |
| 自定义上传逻辑 | P2 | register_parser 配置化 |
| 用户空间隔离 | P2 | token 认证 + 项目归属 |
| PII 清洗 | P2 | Presidio + 中文 NER + 规则 |
| Chat 项目记忆 | P2 | knowledge.sqlite 记忆表 + /chat |
| L0–L5 补全管线 | P2 | 新 completion/ 模块 |
| 触发逻辑/批处理/取消/缓存 | P2 | 前端 + L0;决定 QPS |
| / 系列命令 | P2 | /summarize /find /chat /web /arxiv /refine /help |
| @ 全量过滤器 + tag 管理 | P2 | query_interpreter 扩展 + 元数据层 |
| 连续补全 | P2 | 前端锁定类型 + L0 缓存 |
| 引用标注 + 页码跳转 | P2 | DraftCitation → 证据元数据;前端呈现 |
| WPS/wpsjs(样式/表格/图注) | P2 末–P3 | wpsjs 插件;图注交叉引用低优先级 |
| 图片检索(已有 demo) | P3 | 独立多模态索引管线接入;自动插图列为研究项 |
| 小模型分工 / 微调 / ragas | P2 末–P3 | completion-model 配置 / 评测 harness |
| langfuse + prometheus + 报警 | P3 | 自托管;审计表兜底 |
| pip / compose / k8s / 配置 UI | P2 末–P3 | 打包与配置页 |
| 整稿生成流水线 | **P1(本阶段)** | proposal/ 模块 |

## 9. 决策记录

1. D1 框架:FastAPI 保留(用户确认);litestar 相关诉求(API 文档自动生成)由 FastAPI
   原生 /docs 满足
2. D2 存储:T1 档 SQLite+FAISS 起步(用户确认);milvus/pg/typesense 列为 T2/T3 演进,
   接口已按 allowlist 混合检索形态设计
3. D3 Airbyte 否决,进程内队列等价实现(见第 2 节)
4. D4 整稿生成优先,Copilot 后置(用户确认)
5. D5 延迟预算 1–3s(用户确认),L5 用现有 vLLM
6. D6 /arxiv 保持零 LLM 网页解析,引用格式清单待用户确认
7. D7 全部组件自托管于实验室线上环境,不引入云依赖
