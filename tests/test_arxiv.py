from io import BytesIO
from urllib.error import HTTPError

from paperscout.retrieval.arxiv import build_arxiv_query, search_arxiv


ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2402.03300v3</id>
    <published>2024-02-05T18:42:34Z</published>
    <title>DeepSeekMath: Pushing the Limits of Mathematical Reasoning</title>
    <summary>We introduce Group Relative Policy Optimization (GRPO).</summary>
    <author><name>Author One</name></author>
  </entry>
</feed>"""


class Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_arxiv_query_extracts_english_terms_from_chinese_question() -> None:
    assert build_arxiv_query("查找一下GRPO相关的文章？") == "GRPO"


def test_arxiv_query_translates_common_chinese_ai_topics() -> None:
    assert build_arxiv_query("查找世界模型相关论文") == "world model"
    assert build_arxiv_query("多模态大语言模型") == "large language model multimodal"


def test_arxiv_atom_results_become_parsed_papers(monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response(ATOM))

    results = search_arxiv("GRPO", max_results=1)

    assert len(results) == 1
    assert results[0].paper.id == "arxiv-2402-03300v3"
    assert results[0].paper.year == 2024
    assert results[0].paper.source_path == "https://arxiv.org/abs/2402.03300v3"
    assert "Group Relative Policy Optimization" in results[0].evidence_items[0].text


def test_arxiv_uses_cached_results_after_rate_limit(tmp_path, monkeypatch) -> None:
    responses = iter([Response(ATOM), HTTPError("url", 429, "rate limited", {}, None)])

    def urlopen(request, timeout):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    first = search_arxiv("GRPO", max_results=1, cache_dir=tmp_path, max_retries=0)
    cached = search_arxiv("GRPO", max_results=1, cache_dir=tmp_path, max_retries=0)

    assert cached == first
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_arxiv_cache_is_used_without_network_request(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response(ATOM))
    search_arxiv("GRPO", max_results=1, cache_dir=tmp_path)

    def fail_if_called(request, timeout):
        raise AssertionError("network should not be called for a cached query")

    monkeypatch.setattr("urllib.request.urlopen", fail_if_called)
    assert search_arxiv("GRPO", max_results=1, cache_dir=tmp_path)
