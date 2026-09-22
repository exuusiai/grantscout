"""Real proposal templates extracted from the official PDFs in templates/.

Source documents (archived under templates/):
- lasa-key-project-2023.pdf  拉萨市重点科技计划项目(课题)申报书, 拉萨市科技局 2023
- most-innovation-2015.pdf   科技部创新方法工作专项项目申请书, 国科发资〔2015〕86 号
- jiangsu-erc-2023.pdf       江苏省企业工程技术研究中心项目申报书, 连云港市科技局 2023
- guangxi-collab-2025.pdf    广西"协同"行动计划项目申报书及可行性报告提纲, 2025 年制
- README.md                  模版清单与使用建议(含考核指标六字段、先锁目录再逐章生成)

Word limits follow the source documents (限 X 字以内 -> max_words). Assessment
indicator sections use the six-field structure from README.md: 指标名称、目标值、
测试条件、测试方法、验收材料、来源依据.
"""

from grantscout.proposal.schemas import ProposalTemplate, SectionSpec
from grantscout.proposal.templates import register_template

LASA_KEY_PROJECT_2023 = ProposalTemplate(
    id="lasa-key-project-2023",
    name="拉萨市重点科技计划项目(课题)申报书",
    description="适用拉萨市重点科技计划;叙述性章节加表格栏,表格栏由人工填写。",
    sections=[
        SectionSpec(
            key="sota-ip-standards",
            title="国内外现有技术、知识产权和技术标准现状及预期分析",
            guidance=(
                "项目概述第 1 节。梳理国内外现有技术格局、知识产权态势与技术标准现状,并给出"
                "预期演进分析;每条现状论断必须引用文献证据,收束到本项目切入点。限 1000 字。"
            ),
            min_words=400,
            max_words=1000,
        ),
        SectionSpec(
            key="research-foundation",
            title="申请单位及主要参与单位研究基础",
            guidance=(
                "项目概述第 2 节。只允许依据用户上传的私域材料(已有研发经历、科技成果、科研"
                "条件与队伍现状)组织;不得生成任何未在上传材料中出现的单位成果。限 1000 字。"
            ),
            min_words=300,
            max_words=1000,
        ),
        SectionSpec(
            key="goals-tasks-analysis",
            title="项目确定的目标与任务需求分析",
            guidance="目标与任务第 1 节。需求驱动:从现状空白推出目标与任务边界,指标明确可考核。限 1000 字。",
            min_words=400,
            max_words=1000,
        ),
        SectionSpec(
            key="content-decomposition",
            title="研究内容及任务分解",
            guidance=(
                "目标与任务第 2 节。写清要解决的主要技术难点和问题、技术方案和创新点;任务"
                "分解到可考核的颗粒度。限 1000 字。"
            ),
            min_words=400,
            max_words=1000,
        ),
        SectionSpec(
            key="technical-indicators",
            title="主要技术指标",
            guidance=(
                "知识产权、技术标准、新技术、新产品、新装置、论文专著等数量、指标及其水平;"
                "按六字段结构化填写,指标须量化可考核。限 1000 字。"
            ),
            min_words=200,
            max_words=1000,
            structured_indicators=True,
        ),
        SectionSpec(
            key="economic-indicators",
            title="主要经济指标",
            guidance="技术及产品应用所形成的市场规模、效益等;测算依据须可验证。限 1000 字。",
            min_words=200,
            max_words=1000,
        ),
        SectionSpec(
            key="demonstration-scale",
            title="示范基地、中试线、生产线及其规模",
            guidance="项目实施中形成的示范基地/中试线/生产线及其规模。限 500 字。",
            min_words=100,
            max_words=500,
            requires_citations=False,
        ),
        SectionSpec(
            key="team-building",
            title="人才队伍建设",
            guidance="项目期内人才引进、培养与团队建设安排。限 500 字。",
            min_words=100,
            max_words=500,
            requires_citations=False,
        ),
        SectionSpec(
            key="other-indicators",
            title="其他应考核的指标",
            guidance="上述未覆盖但需考核的指标,无则填无。限 500 字。",
            min_words=0,
            max_words=500,
            requires_citations=False,
        ),
        SectionSpec(
            key="annual-plan",
            title="项目年度计划及年度目标",
            guidance="按年度写计划与年度目标,每年度 500 字左右;年度目标须与考核指标衔接。",
            min_words=400,
            max_words=1000,
            requires_citations=False,
        ),
        SectionSpec(
            key="budget-form",
            title="总投资预算、经费来源与各项任务经费分配表(表格)",
            guidance="表格栏:预算科目、专项经费、自筹经费、年度资金安排,由人工填写。",
            generatable=False,
        ),
        SectionSpec(
            key="personnel-form",
            title="项目主要研究人员(表格)",
            guidance="表格栏:负责人与研究人员姓名、分工、人月数,由人工填写。",
            generatable=False,
        ),
        SectionSpec(
            key="ethics-form",
            title="科技伦理审查意见表",
            guidance="表格栏:涉及人体/动物实验、人工智能、遗传基因等须先伦理审查,由人工办理。",
            generatable=False,
        ),
    ],
)

MOST_INNOVATION_2015 = ProposalTemplate(
    id="most-innovation-2015",
    name="科技部创新方法工作专项项目申请书",
    description="适用国家级专项、多课题联合申报;章节完整,适合长篇生成。",
    sections=[
        SectionSpec(
            key="importance-urgency",
            title="重要性和紧迫性",
            guidance="从国家/行业需求出发论证重要性与紧迫性,与指南方向衔接;背景论断须有依据。",
            min_words=800,
            max_words=1500,
        ),
        SectionSpec(
            key="sota-trends",
            title="国内外现状及趋势",
            guidance=(
                "国内外技术与应用现状、发展趋势;每条论断引用文献证据,收束到本项目必要性。"
            ),
            min_words=800,
            max_words=2000,
        ),
        SectionSpec(
            key="goals-outcomes",
            title="预期目标及预期成果",
            guidance="总体目标、预期成果形式与量化水平;与考核指标一致。",
            min_words=600,
            max_words=1500,
        ),
        SectionSpec(
            key="content-topics",
            title="主要工作内容及课题设置方案",
            guidance=(
                "详细阐述工作内容,包括创新点、任务重点与难点、技术路线、工作方案和措施;"
                "如下设课题,说明课题设置思路、各课题间有机联系及与预期目标的关系。"
            ),
            min_words=1200,
            max_words=2500,
        ),
        SectionSpec(
            key="assessment-indicators",
            title="考核指标",
            guidance=(
                "指标需量化、可考核;如有企业应用示范任务,需说明知识产权、新技术、新产品、"
                "人才队伍建设、预期经济与社会效益等量化指标。按六字段结构化填写。"
            ),
            min_words=400,
            max_words=1200,
            structured_indicators=True,
        ),
        SectionSpec(
            key="mechanism-sharing",
            title="运行机制及共享方案",
            guidance="组织运行机制、成果开放共享与宣传推广安排。",
            min_words=500,
            max_words=1200,
            requires_citations=False,
        ),
        SectionSpec(
            key="foundation-conditions",
            title="现有工作基础和条件",
            guidance=(
                "牵头与参与单位基本情况、研发经历与在研项目、队伍现状、已有应用经历;"
                "只能依据用户上传的私域材料撰写。"
            ),
            min_words=600,
            max_words=1500,
        ),
        SectionSpec(
            key="annual-plan",
            title="年度计划及考核指标",
            guidance="按第一/二/三年度写工作目标与内容、对应考核指标,与总考核指标一致。",
            min_words=400,
            max_words=1000,
            requires_citations=False,
        ),
        SectionSpec(
            key="topics-form",
            title="课题设置情况(表格)",
            guidance="表格栏:课题名称、目标、内容、承担单位、负责人、经费,由人工填写。",
            generatable=False,
        ),
        SectionSpec(
            key="budget-form",
            title="经费支出预算(表格)",
            guidance="表格栏:直接费用(设备/材料/测试/差旅等)与间接费用,含测算说明,由人工填写。",
            generatable=False,
        ),
        SectionSpec(
            key="personnel-form",
            title="项目参与人员(表格)",
            guidance="表格栏:人员明细、职称代码、人月数、人员分类,由人工填写。",
            generatable=False,
        ),
    ],
)

JIANGSU_ERC_2023 = ProposalTemplate(
    id="jiangsu-erc-2023",
    name="江苏省企业工程技术研究中心项目申报书",
    description="适用工程化、平台建设与企业研发中心类项目;建设期 3 年。",
    sections=[
        SectionSpec(
            key="industry-requirements",
            title="行业需求分析",
            guidance=(
                "所涉技术领域国内外发展现状与今后发展趋势、江苏产业现有优势和主要问题;"
                "项目组建对相关产业发展、企业创新的作用与意义。现状论断须引用证据。"
            ),
            min_words=800,
            max_words=1800,
        ),
        SectionSpec(
            key="implementation-foundation",
            title="项目实施基础",
            guidance=(
                "申报单位基本情况(主营业务、行业优势、上年度销售收入、R&D 投入占比)、现有"
                "研发基础条件(场所/中试基地/仪器装备/市级工程中心/人才团队)、近 2 年承担的"
                "科技项目与知识产权。只能依据用户上传的私域材料撰写。"
            ),
            min_words=600,
            max_words=1500,
        ),
        SectionSpec(
            key="goals-tasks",
            title="项目主要目标和建设任务",
            guidance=(
                "总体目标和定位;凝练 2-3 个主要研发方向;组织功能架构(框图另附);建设地点;"
                "硬件建设、研究开发、领军人才引进与培养、管理体制与运行体制四类任务(建设期 3 年);"
                "涉及产学研共建的需说明合作基础与预期成效。"
            ),
            min_words=1000,
            max_words=2000,
        ),
        SectionSpec(
            key="indicators-summary",
            title="主要任务与具体考核指标简述",
            guidance="依据前述任务量化考核指标,限 500 字;按六字段结构化填写。",
            min_words=300,
            max_words=500,
            structured_indicators=True,
        ),
        SectionSpec(
            key="implementation-plan",
            title="项目实施计划",
            guidance=(
                "项目投资规模及建设资金来源与构成比例;建设经费支出预算及仪器设备添置清单;"
                "项目组建的计划进度与阶段性考核指标;负责人及主要技术人员清单。清单类内容由人工填写。"
            ),
            min_words=600,
            max_words=1500,
            requires_citations=False,
        ),
        SectionSpec(
            key="direction-content",
            title="工程中心主要研究方向和建设内容",
            guidance="基本信息表栏目,限 1000 字;概括研发方向与建设内容,与正文一致。",
            min_words=400,
            max_words=1000,
        ),
    ],
)

GUANGXI_COLLAB_2025 = ProposalTemplate(
    id="guangxi-collab-2025",
    name="广西'协同'行动计划项目申报书",
    description="适用广西区域创新能力提升计划;以非高新区可行性报告提纲为主体。",
    sections=[
        SectionSpec(
            key="summary",
            title="项目摘要",
            guidance="描述项目要解决的关键问题、主要目标、任务、标志性成果。限 300 字。",
            min_words=150,
            max_words=300,
            requires_citations=False,
        ),
        SectionSpec(
            key="overall-goal",
            title="项目总体目标",
            guidance="限 500 字;总体目标须可分解、可考核。",
            min_words=200,
            max_words=500,
            requires_citations=False,
        ),
        SectionSpec(
            key="main-content",
            title="项目主要内容",
            guidance="限 800 字;研究开发内容与建设内容。",
            min_words=300,
            max_words=800,
        ),
        SectionSpec(
            key="key-technical-problems",
            title="研究解决的关键技术问题",
            guidance="限 800 字;凝练关键技术问题,每条对应研究内容,问题提炼须有文献依据。",
            min_words=300,
            max_words=800,
        ),
        SectionSpec(
            key="background-necessity",
            title="项目技术需求背景、立项依据与必要性",
            guidance=(
                "技术需求来源与列入的规划计划;应用前景;对全区(全市或本单位)产业发展的意义;"
                "国内外同类产品或技术研究应用的情况(现状论断必须引用文献证据)。"
            ),
            min_words=600,
            max_words=1500,
        ),
        SectionSpec(
            key="research-content-goals",
            title="项目研究内容与目标",
            guidance=(
                "研究目标;研究的主要内容;研究方法;技术路线;拟解决的关键技术问题;"
                "围绕基础前沿、共性关键技术或应用层面简述主要预期创新点和先进性。"
            ),
            min_words=800,
            max_words=1800,
        ),
        SectionSpec(
            key="feasibility",
            title="开展项目研究的可行性",
            guidance=(
                "研究基础(团队/设备/平台/场地/前期成果)、核心关键技术研发与合作情况、"
                "产业化基础、主要研发人员情况。只能依据用户上传的私域材料撰写。"
            ),
            min_words=500,
            max_words=1200,
        ),
        SectionSpec(
            key="outcomes-indicators",
            title="成果预期的技术指标及国内外的水平",
            guidance=(
                "成果形式(论文、专利、样机、新技术、新产品等);技术指标及其国内外水平;"
                "按六字段结构化填写,指标须可考核。"
            ),
            min_words=300,
            max_words=1000,
            structured_indicators=True,
        ),
        SectionSpec(
            key="transformation-benefits",
            title="成果转化应用、产业化计划与经济社会效益",
            guidance=(
                "成果转化应用及产业化计划;预期经济社会效益指标(年产值、利润、税收等);"
                "直接经济效益需列出具体、科学、规范的测算依据,此为财务和技术专家重点评审内容。"
            ),
            min_words=400,
            max_words=1000,
        ),
        SectionSpec(
            key="unit-description",
            title="牵头承担单位情况说明",
            guidance=(
                "上一年研发投入、自立项科研活动、税前加计扣除、近 5 年项目与获奖、近 3 年"
                "产业化转化。只能依据用户上传的私域材料撰写。"
            ),
            min_words=400,
            max_words=1000,
            requires_citations=False,
        ),
        SectionSpec(
            key="budget-calculation",
            title="实施期限与项目经费测算",
            guidance=(
                "概况(期限、总投资、科研经费、申请财政经费);按科目写测算说明(设备费/业务费/"
                "劳务费/间接费用与绩效支出),含测算方法和依据。金额表格由人工填写。"
            ),
            min_words=400,
            max_words=1000,
            requires_citations=False,
        ),
    ],
)

UNIVERSAL_PLAN_V1 = ProposalTemplate(
    id="universal-plan-v1",
    name="通用项目方案骨架",
    description=(
        "由四份官方模版归纳的通用 14 节骨架(见 templates/README.md),适合尚无"
        "指定申报格式的场景;正式申报时切换到对应官方模板。"
    ),
    sections=[
        SectionSpec(key="overview", title="项目概述与摘要", min_words=400, max_words=1200, requires_citations=False),
        SectionSpec(
            key="background-necessity",
            title="需求背景、任务来源与立项必要性",
            guidance="需求来源、任务来源(规划/计划/委托)、立项必要性;需求论断须有依据。",
            min_words=800,
            max_words=1800,
        ),
        SectionSpec(
            key="sota",
            title="国内外研究与技术发展现状",
            guidance="分国内/国外梳理代表性工作与趋势,每条论断引用文献证据,收束到研究空白。",
            min_words=1200,
            max_words=2500,
        ),
        SectionSpec(
            key="goals-indicators",
            title="总体目标、任务边界与量化指标",
            guidance="总体目标可分解,任务边界清晰,量化指标可考核。",
            min_words=600,
            max_words=1500,
        ),
        SectionSpec(
            key="content-decomposition",
            title="研究内容与任务分解",
            guidance="研究内容分解为可考核任务,标注任务间依赖与交付物。",
            min_words=800,
            max_words=2000,
        ),
        SectionSpec(
            key="system-design",
            title="总体架构或系统设计",
            guidance="总体架构、模块划分、数据流与接口设计;架构须支撑需求与指标。",
            min_words=800,
            max_words=1800,
        ),
        SectionSpec(
            key="technical-route",
            title="技术路线、关键技术与实施方案",
            guidance="技术路线图(文字描述)、关键技术选型及依据(须引用证据)、实施路径。",
            min_words=1000,
            max_words=2200,
        ),
        SectionSpec(
            key="innovation-ip",
            title="创新点、先进性与知识产权分析",
            guidance="逐条列出创新点并说明先进性;与现状空白对应;知识产权布局分析。",
            min_words=500,
            max_words=1200,
        ),
        SectionSpec(
            key="indicators-testing",
            title="指标分解、测试方法和验收依据",
            guidance="指标逐条分解并按六字段结构化;测试条件与方法须可执行。",
            min_words=600,
            max_words=1500,
            structured_indicators=True,
        ),
        SectionSpec(
            key="foundation-team",
            title="研究基础、团队与条件保障",
            guidance="只能依据用户上传的私域材料撰写:前期成果、团队、平台与保障条件。",
            min_words=500,
            max_words=1500,
            requires_citations=False,
        ),
        SectionSpec(
            key="schedule-milestones",
            title="进度计划、里程碑与年度考核指标",
            guidance="按年度/阶段列任务、完成标志与里程碑;里程碑至少 1 个。",
            min_words=400,
            max_words=1000,
            requires_citations=False,
        ),
        SectionSpec(
            key="outcomes-benefits",
            title="预期成果、应用示范和效益分析",
            guidance="成果形式与数量、应用示范、经济社会效益;效益须给测算依据。",
            min_words=500,
            max_words=1500,
        ),
        SectionSpec(
            key="risks",
            title="风险分析与应对措施",
            guidance="技术、进度、经费等风险及应对;风险识别须与方案对应。",
            min_words=300,
            max_words=1000,
            requires_citations=False,
        ),
        SectionSpec(
            key="budget",
            title="经费预算及说明",
            guidance="预算结构与测算说明;金额表格由人工填写。",
            min_words=300,
            max_words=800,
            requires_citations=False,
        ),
    ],
)


def register_real_templates() -> None:
    for template in (
        LASA_KEY_PROJECT_2023,
        MOST_INNOVATION_2015,
        JIANGSU_ERC_2023,
        GUANGXI_COLLAB_2025,
        UNIVERSAL_PLAN_V1,
    ):
        register_template(template)
