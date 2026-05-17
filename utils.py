"""Utility helpers for the Streamlit app."""

from typing import List


def get_file_icon(ext: str) -> str:
    return {
        "xlsx": "📊",
        "xls": "📊",
        "csv": "📋",
        "pdf": "📕",
        "docx": "📝",
        "doc": "📝",
    }.get(ext.lower(), "📄")


def format_sources(sources: List[str]) -> str:
    if not sources:
        return ""
    return "**Sources:** " + " · ".join(f"`{s}`" for s in sources)
