"""Registry of proposal templates.

Built-in templates carry placeholder guidance. When the real proposal
templates arrive, fill in the ``guidance`` and word budgets of a
``SectionSpec`` — no code changes are needed.
"""

from grantscout.proposal.schemas import ProposalTemplate, SectionSpec

_PROJECT_PLAN_V1 = ProposalTemplate(
    id="project-plan-v1",
    name="项目方案模板(通用)",
    description=(
        "基于需求分析、研究现状、系统设计、技术方案、指标分析五段式的项目方案。"
        "guidance 为占位内容,等待真实模板固化后替换。"
    ),
    sections=[
        SectionSpec(
            key="requirements",
            title="需求分析",
            guidance=(
                "占位:说明业务背景、痛点、使用对象与需求条目。评审关注需求是否有依据、"
                "是否与现状分析衔接。【待真实模板固化后替换】"
            ),
            min_words=2000,
            max_words=3000,
        ),
        SectionSpec(
            key="state-of-the-art",
            title="国内外研究现状",
            guidance=(
                "占位:分国内/国外梳理代表工作,每条论断需引用文献证据,并收束到研究空白。"
                "评审关注覆盖面与引用真实性。【待真实模板固化后替换】"
            ),
            min_words=4000,
            max_words=5000,
        ),
        SectionSpec(
            key="system-design",
            title="系统设计",
            guidance=(
                "占位:总体架构、模块划分、数据流与接口设计。评审关注架构是否支撑需求。"
                "【待真实模板固化后替换】"
            ),
            min_words=3000,
            max_words=4000,
        ),
        SectionSpec(
            key="technical-approach",
            title="技术方案",
            guidance=(
                "占位:关键技术路线、算法选型与依据、实现路径。技术选型需引用证据支撑。"
                "【待真实模板固化后替换】"
            ),
            min_words=4000,
            max_words=5000,
        ),
        SectionSpec(
            key="metrics",
            title="指标分析",
            guidance=(
                "占位:性能/功能指标定义、测算依据、与现状对比。指标需可验证。"
                "【待真实模板固化后替换】"
            ),
            min_words=2000,
            max_words=3000,
        ),
    ],
)

_TEMPLATES: dict[str, ProposalTemplate] = {
    template.id: template for template in (_PROJECT_PLAN_V1,)
}


def list_templates() -> list[ProposalTemplate]:
    return list(_TEMPLATES.values())


def get_template(template_id: str) -> ProposalTemplate:
    template = _TEMPLATES.get(template_id)
    if template is None:
        known = ", ".join(sorted(_TEMPLATES)) or "(none)"
        raise KeyError(f"Unknown proposal template: {template_id}; known templates: {known}")
    return template


def register_template(template: ProposalTemplate) -> None:
    """Register or replace a template at runtime (used by config-driven setups)."""
    _TEMPLATES[template.id] = template
