"""Web/学术搜索:OpenAlex 确定性检索 + 可选外部 LLM 综合。

设计约束:
- 检索源是 OpenAlex(公开学术索引,确定性解析,零幻觉);
- 外部 LLM(如 DeepSeek)只做结果综合,且**仅接收检索到的公开元数据**,
  私域语料内容绝不发往外部接口;
- 外部 API 不可用或未配置时降级为纯列表呈现。
"""

import logging

import httpx

from grantscout.config import Settings

logger = logging.getLogger(__name__)

OPENALEX_URL = "https://api.openalex.org/works"


def search_openalex(query: str, max_results: int = 5, timeout: float = 15.0) -> list[dict]:
    """Deterministic OpenAlex title search; raises httpx.HTTPError on failure."""
    clean = query.strip()[:200]
    with httpx.Client() as client:
        response = client.get(
            OPENALEX_URL,
            params={
                "filter": f"title.search:{clean}",
                "sort": "cited_by_count:desc",
                "per-page": max_results,
                "mailto": "grantscout@localhost",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        works = response.json().get("results", [])
    sources = []
    for work in works:
        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}
        sources.append(
            {
                "title": work.get("display_name") or "",
                "year": work.get("publication_year"),
                "venue": source.get("display_name"),
                "cited_by_count": work.get("cited_by_count") or 0,
                "doi": work.get("doi"),
                "url": primary.get("landing_page_url") or work.get("id"),
            }
        )
    return sources


def web_answer(query: str, settings: Settings, max_results: int = 5) -> dict:
    """检索 + 可选外部模型综合。返回 {answer, sources, warnings}。

    answer 为 None 表示外部模型不可用/未配置,前端呈现纯来源列表。
    """
    warnings: list[str] = []
    try:
        sources = search_openalex(query, max_results=max_results)
    except httpx.HTTPError as error:
        logger.warning("OpenAlex search failed: %s", error)
        return {
            "answer": None,
            "sources": [],
            "warnings": [f"OpenAlex 检索失败:{error}"],
        }
    if not sources:
        return {"answer": None, "sources": [], "warnings": ["OpenAlex 未返回结果。"]}

    if not (settings.web_api_base_url and settings.web_api_key and settings.web_api_model):
        warnings.append("外部模型未配置(GRANTSCOUT_WEB_API_*),仅呈现检索结果。")
        return {"answer": None, "sources": sources, "warnings": warnings}

    from grantscout.models.llm import ModelClientError, OpenAICompatibleClient, strip_thinking

    client = OpenAICompatibleClient(
        base_url=settings.web_api_base_url,
        api_key=settings.web_api_key,
        model=settings.web_api_model,
        timeout_seconds=60.0,
        allow_research_endpoint=True,
    )
    listing = "\n".join(
        f"【{index}】{item['title']}({item['venue'] or 'venue 未知'},{item['year'] or '年份未知'},"
        f"被引 {item['cited_by_count']})"
        for index, item in enumerate(sources, start=1)
    )
    try:
        # 推理型模型(DeepSeek 等)的思考段可能占用大量 token,放宽限额。
        response = client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是学术搜索助手。只依据提供的检索结果,用中文总结该主题的研究现状与"
                        "代表性工作(不超过 6 句),并在引用的条目处标注【n】。检索结果之外的信息"
                        "一律不写。直接输出总结正文。"
                    ),
                },
                {"role": "user", "content": f"搜索词:{query}\n\n检索结果:\n{listing}"},
            ],
            max_tokens=2048,
            temperature=0.2,
        )
        # 推理型端点可能把正文放在 reasoning_content,content 为空;两者都留。
        content = response.content or ""
        usage = response.usage or {}
        logger.debug("web synth usage: %s", usage)
        answer = strip_thinking(content).strip() or None
    except ModelClientError as error:
        logger.warning("External web model failed: %s", error)
        warnings.append(f"外部模型综合失败,已降级为纯列表:{error}")
        answer = None
    return {"answer": answer, "sources": sources, "warnings": warnings}
