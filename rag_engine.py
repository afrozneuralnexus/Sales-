"""
RAGEngine – retrieval-augmented generation using the Grok (xAI) API.
"""

from __future__ import annotations

import json
from typing import Dict, List

import requests

from src.vector_store import VectorStore

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-3-mini"

SYSTEM_PROMPT = """You are an expert Sales Data Analyst assistant. You analyze sales data and provide \
clear, actionable insights.

When answering:
- Be specific with numbers, percentages, and trends from the provided context
- Highlight key findings and anomalies
- Suggest actionable recommendations when relevant
- Format responses with clear structure (use bullet points, tables in markdown where helpful)
- If data is insufficient, say so clearly

Always base your answers on the retrieved context. Do not hallucinate figures."""


class RAGEngine:
    def __init__(self, api_key: str, vector_store: VectorStore, top_k: int = 5):
        self.api_key = api_key
        self.vs = vector_store
        self.top_k = top_k

    def query(self, question: str, chat_history: List[Dict] | None = None) -> Dict:
        # 1. Retrieve relevant chunks
        results = self.vs.similarity_search(question, k=self.top_k)
        if not results:
            return {"answer": "No relevant data found in the uploaded documents.", "sources": []}

        # 2. Build context
        context_parts = []
        sources = []
        for chunk, score in results:
            src = chunk.source
            if chunk.sheet:
                src += f" [{chunk.sheet}]"
            if chunk.page:
                src += f" [Page {chunk.page}]"
            context_parts.append(f"--- Source: {src} (relevance: {score:.2f}) ---\n{chunk.text}")
            if src not in sources:
                sources.append(src)

        context = "\n\n".join(context_parts)

        # 3. Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add condensed chat history (last 6 turns)
        if chat_history:
            for msg in chat_history[-6:]:
                if msg["role"] in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

        user_message = f"""Context from uploaded sales documents:

{context}

---
Question: {question}

Please provide a thorough analysis based on the context above."""

        messages.append({"role": "user", "content": user_message})

        # 4. Call Grok API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.2,
        }

        resp = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        answer = data["choices"][0]["message"]["content"]
        return {"answer": answer, "sources": sources}
