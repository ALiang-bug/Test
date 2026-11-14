import json
import logging
import random
from typing import List, Dict, Any
from dataclasses import dataclass
from ddgs import DDGS
from langchain_openai import ChatOpenAI
import numpy as np
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from prompt_template.external_knowledge.system_prompts import *
from prompt_template.external_knowledge.user_prompts import *
import time
import re
import os

_global_embedding_model = None
_global_embedding_lock = None

def get_global_embedding_model(
    embedding_model_path: str = "~/model_path/bge-base-en-v1.5/",
    embedding_device: str = 'cuda:1'
):
    embedding_model_path = os.path.expanduser(embedding_model_path)

    global _global_embedding_model, _global_embedding_lock
    if _global_embedding_model is None:
        import threading
        if _global_embedding_lock is None:
            _global_embedding_lock = threading.Lock()
        with _global_embedding_lock:
            if _global_embedding_model is None:
                # print(f"Loading BGE model globally on {embedding_device}...")
                model_kwargs = {'device': embedding_device}
                encode_kwargs = {'normalize_embeddings': True}
                _global_embedding_model = HuggingFaceEmbeddings(
                    model_name=embedding_model_path,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
                # print("BGE model loaded globally.")
    return _global_embedding_model

@dataclass
class SearchResult:
    title: str
    href: str
    body: str
    relevance_score: float = 0.0  # LLM score
    bge_score: float = 0.0  # BGE score
    is_filtered: bool = False
    filter_reason: str = ""


@dataclass
class KnowledgeResult:
    query: str
    search_results: List[SearchResult]
    filtered_results: List[SearchResult]
    retrieval_time: float = 0.0


class ExternalKnowledgeRetriever:
    def __init__(
        self,
        base_url: str = "http://localhost:19887/v1",
        api_key: str = "sk-123456",
        model: str = "Qwen3-VL-32B-Instruct-FP8",
        temperature: float = 0.0,
        proxy_url: str = "socks5h://127.0.0.1:7891",
        max_results: int = 10,
        timeout: int = 10,
        relevance_threshold: float = 0.85,
        embedding_model_path: str = "~/model_path/bge-base-en-v1.5/",
        embedding_device: str = 'cuda:1',
        max_pre_filtered: int = 10
    ):
        self.logger = logging.getLogger(__name__)
        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature
        )
        self.search_engines = ["yahoo", "wikipedia", "yandex"]
        self.ddgs = DDGS(proxy=proxy_url, timeout=timeout)
        self.max_results = max_results
        self.relevance_threshold = relevance_threshold
        embedding_model_path = os.path.expanduser(embedding_model_path)
        self.embedding_model = get_global_embedding_model(
            embedding_model_path=embedding_model_path,
            embedding_device=embedding_device
        )
        self.max_pre_filtered = max_pre_filtered

    def _invoke_llm(self, system_prompt: str, user_prompt: str, json_format: bool = False) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            if json_format:
                json_llm = self.llm.bind(response_format={"type": "json_object"})
                response = json_llm.invoke(messages)
            else:
                response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return "{}" if json_format else ""

    def generate_queries(self, text: str, debate_summary: str) -> List[str]:
    
        system_prompt = Generate_Queries_System.strip()

        generate_queries_prompt = Generate_Queries_User.format(news_text=text, debate_summary=debate_summary).strip()

        raw = self._invoke_llm(system_prompt, generate_queries_prompt)

        try:
            queries = json.loads(raw)
            if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
                raise ValueError
            queries = queries[:5]
        except Exception as e:
            self.logger.error(f"Failed to parse generated queries: {e}. Response was: {raw}")

            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                queries = json.loads(match.group(0))
            else:
                queries = []

        return queries
    
    def search_web(self, query: str) -> List[SearchResult]:
        try:
            results = self.ddgs.text(
                query=query,
                region="us-en",
                safesearch='moderate',
                timelimit=None,
                max_results=self.max_results,
                page=1,
                backend=", ".join(self.search_engines)
            )
            search_results = []
            if results:
                for result in results:
                    if isinstance(result, dict):
                        search_result = SearchResult(
                            title=result.get('title', ''),
                            href=result.get('href', ''),
                            body=result.get('body', ''),
                        )
                        search_results.append(search_result)
                    else:
                        self.logger.warning(f"Unexpected result format: {type(result)} - {result}")
            return search_results
        except Exception as e:
            self.logger.error(f"Search failed for query '{query}': {e}")
            return []

    def pre_filter_with_bge(
        self,
        search_results: List[SearchResult],
        text: str,
        debate_summary: str
    ) -> List[SearchResult]:

        if not search_results:
            return []

        # 1. reference text
        reference_text = (
            f"News Text: {text}\nDebate Summary: {debate_summary}"
            if debate_summary.strip()
            else f"News Text: {text}"
        )

        # 2. embedding
        ref_emb = self.embedding_model.embed_query(reference_text)

        # 3. search text
        passages = [f"Title: {r.title}\nContent: {r.body}" for r in search_results]
        doc_embs = self.embedding_model.embed_documents(passages)

        # 4. cosine
        similarities = cosine_similarity([ref_emb], doc_embs)[0]   # shape: (n,)

        for result, score in zip(search_results, similarities):
            result.bge_score = round(float(score), 2)

        sorted_results = sorted(
            search_results,
            key=lambda r: r.bge_score,
            reverse=True
        )

        return sorted_results[: self.max_pre_filtered]

    def filter_search_results(
        self,
        search_results: List[SearchResult],
        text: str,
        debate_summary: str
    ) -> List[SearchResult]:

        if not search_results:
            return []

        has_debate = debate_summary and debate_summary.strip()

        debate_context_prompt = debate_summary if has_debate else "Not provided. Focus on key factual claims in the news text only."

        results_text = ""
        items = []
        for i, result in enumerate(search_results):
            # results_text += f"""Result {i+1}:\nIndex: {i+1}\nTitle: {result.title}\nContent: {result.body}\nURL: {result.href}\n-----\n"""
            index = i + 1
            item = {
                "index": index,
                "title": result.title,
                "content": result.body,
                "url": result.href
            }
            items.append(json.dumps(item, ensure_ascii=False))

        results_text = "\n---\n".join(items)

        filter_prompt = Filter_Search_Results_User.format(
            text=text,
            debate_context_prompt=debate_context_prompt,
            results_text=results_text,
            len_results=len(search_results)
        )

        system_prompt = Filter_Search_Results_System

        try:
            filter_response = self._invoke_llm(system_prompt, filter_prompt, json_format=True)
            self.logger.debug(f"Filter response: {filter_response}")
            filter_decision = json.loads(filter_response)

            evaluations = filter_decision.get("evaluations", [])
            eval_map = {eval_item["index"]: eval_item for eval_item in evaluations}
            self.logger.debug(f"Evaluations: {eval_map}")

            threshold = self.relevance_threshold
            for i, result in enumerate(search_results):
                index = i + 1
                if index in eval_map:
                    score_str = eval_map[index].get("score", 0.0)
                    reason = eval_map[index].get("reason", "")
                    try:
                        result.relevance_score = round(float(score_str), 2)
                    except (ValueError, TypeError):
                        result.relevance_score = 0.00

                    result.filter_reason = reason
                    result.is_filtered = result.relevance_score < threshold

                    self.logger.debug(f"Result {index}: LLM_score={result.relevance_score}, BGE_score={result.bge_score}, filtered={result.is_filtered}")
                else:
                    result.relevance_score = 0.00
                    result.is_filtered = True
                    result.filter_reason = "No evaluation provided by LLM"
                    self.logger.debug(f"Result {index}: default filtered (no evaluation)")

            return search_results
        except Exception as e:
            self.logger.error(f"Filtering failed: {e}. Returning all results unfiltered.")
            return search_results


    def retrieve_knowledge(
        self,
        text: str,
        debate_summary: str
    ) -> KnowledgeResult:

        start_time = time.time()

        # Step 1: Generate Queries
        queries = self.generate_queries(text, debate_summary)
        # print(time.time() - start_time)

        # Step 2: Remove duplicates from search results
        all_search_results = []
        if queries:
            for query in queries:
                results = self.search_web(query)
                all_search_results.extend(results)
                time.sleep(random.randint(1, 5))

            seen_hrefs = set()
            unique_results = []
            for res in all_search_results:
                if res.href not in seen_hrefs:
                    unique_results.append(res)
                    seen_hrefs.add(res.href)
            all_search_results = unique_results
        # print(all_search_results)
        # print(type(all_search_results))
        # print(time.time() - start_time)

        # Step 3: BGE Score
        pre_filtered = self.pre_filter_with_bge(all_search_results, text, debate_summary)

        # print(pre_filtered)
        # print(time.time() - start_time)

        # Step 4: LLM Score
        evaluated = self.filter_search_results(pre_filtered, text, debate_summary)

        # print(evaluated)
        final_results = [r for r in evaluated if not r.is_filtered]
        # print(time.time() - start_time)

        # Step 5: sort
        sorted_results = sorted(
            final_results, 
            key=lambda r: (r.relevance_score, r.bge_score), 
            reverse=True
        )

        # print(time.time() - start_time)

        retrieval_time = time.time() - start_time

        knowledge_result = KnowledgeResult(
            query="; ".join(queries),
            search_results=evaluated,
            filtered_results=sorted_results,
            retrieval_time=retrieval_time
        )
        # print(retrieval_time)
        self.logger.info(f"Knowledge retrieval + completion done in {retrieval_time:.2f}s, "
                        f"got {len(sorted_results)} full texts.")

        return knowledge_result