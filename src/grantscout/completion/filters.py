"""Parsing for the ``@{...}`` context-scoping filters typed inline by users.

Supported keys (values after ":"):

- ``@{filename:xxx}``  restrict to one document (title or source path match)
- ``@{time:2022-2024}`` or ``@{time:recent3}`` restrict by publication year
- ``@{author:xxx}``    restrict to one author
- ``@{tag:xxx}``       restrict to documents carrying a tag
- ``@{impact:>5}``     restrict to documents with cited_by_count >= 5
- ``@{conference:CVPR,ICCV}`` restrict to papers published at those venues
- ``@{unset}``         clear every active filter

Keys that need data not yet ingested (domain / language) parse successfully
but produce a warning instead of a filter, so the frontend can show what
actually took effect.
"""

import re
from dataclasses import dataclass, field

_FILTER_PATTERN = re.compile(r"@\{([^{}]+)\}")
_SUPPORTED_WITH_VALUE = {"filename", "time", "author", "tag", "impact", "conference"}
_METADATA_PENDING = {"domain", "language"}


@dataclass
class FilterSet:
    filename: list[str] = field(default_factory=list)
    time_from: int | None = None
    time_to: int | None = None
    author: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    impact_min: int | None = None
    conferences: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.filename
            or self.author
            or self.tags
            or self.conferences
            or self.time_from is not None
            or self.time_to is not None
            or self.impact_min is not None
        )


@dataclass
class ParsedPrefix:
    clean_text: str
    filters: FilterSet
    warnings: list[str] = field(default_factory=list)


def parse_filters(prefix: str) -> ParsedPrefix:
    """Strip ``@{...}`` filters out of the prefix and interpret them."""
    filters = FilterSet()
    warnings: list[str] = []

    def consume(match: re.Match[str]) -> str:
        raw = match.group(1).strip()
        key, _, value = raw.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "unset":
            filters.filename.clear()
            filters.author.clear()
            filters.tags.clear()
            filters.conferences.clear()
            filters.time_from = None
            filters.time_to = None
            filters.impact_min = None
            return ""
        if key in _SUPPORTED_WITH_VALUE:
            if not value:
                warnings.append(f"@{{{key}}} 缺少取值,已忽略。")
                return ""
            if key == "filename":
                filters.filename.append(value)
            elif key == "author":
                filters.author.append(value)
            elif key == "tag":
                filters.tags.append(value)
            elif key == "conference":
                filters.conferences.extend(item.strip() for item in value.split(",") if item.strip())
            elif key == "time":
                _apply_time(filters, value, warnings)
            elif key == "impact":
                digits = "".join(char for char in value if char.isdigit())
                if not digits:
                    warnings.append(f"@{{impact:{value}}} 无法解析数值,已忽略。")
                else:
                    filters.impact_min = int(digits)
            return ""
        if key in _METADATA_PENDING:
            warnings.append(f"@{{{key}}} 需要元数据层支持,当前版本未生效。")
            return ""
        warnings.append(f"未知的过滤器 @{{{key}}},已忽略。")
        return ""

    clean = _FILTER_PATTERN.sub(consume, prefix)
    clean = re.sub(r"[ \t]+", " ", clean).strip()
    return ParsedPrefix(clean_text=clean, filters=filters, warnings=warnings)


def _apply_time(filters: FilterSet, value: str, warnings: list[str]) -> None:
    if value.lower().startswith("recent"):
        digits = "".join(char for char in value if char.isdigit())
        years = int(digits) if digits else 3
        import datetime

        current = datetime.datetime.now(datetime.timezone.utc).year
        filters.time_from = current - years
        return
    parts = value.replace("–", "-").split("-")
    try:
        if len(parts) == 2:
            filters.time_from = int(parts[0])
            filters.time_to = int(parts[1])
        else:
            filters.time_from = int(parts[0])
    except (ValueError, IndexError):
        warnings.append(f"@{{time:{value}}} 无法解析年份区间,已忽略。")
