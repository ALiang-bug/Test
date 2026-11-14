import json
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import base64
import logging
import re
import os
from typing import TypedDict, Dict, Any, List
from prompt_template.news_analyzer.system_prompts import *
from prompt_template.news_analyzer.user_prompts import *
from external_knowledge import ExternalKnowledgeRetriever, KnowledgeResult, SearchResult

class NewsState(TypedDict):
    text: str
    image: str  # base64
    image_analysis: str
    entity: List[str]
    domain_tags: List[str]
    domain_profile: str
    debate_log: str
    debate_summary: str
    label: str
    text_reason: str
    image_reason: str
    consistency_reason: str
    current_round: int
    query_list: List[str]
    external_knowledge: List[str]
    retrieved_knowledge_only: List[str]
    short_term_memories: List[str]
    long_term_memory: str
    captions: str


class NewsAnalyzer:
    def __init__(self, base_url: str = "http://localhost:19887/v1", api_key: str = "sk-123456", model: str = "Qwen3-VL-32B-Instruct-FP8", temperature: float = 0.0, n_rounds: int = 3, logger: logging.RootLogger = None):
        # init
        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature
        )
        self.n_rounds = n_rounds  # debate round
        embedding_model_path = os.path.expanduser("~/model_path/bge-base-en-v1.5/")
        self.retriever = ExternalKnowledgeRetriever(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            embedding_device='cuda:0',
            embedding_model_path=embedding_model_path
        )
        # LangGraph workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

        self.logger = logger

    def _invoke_llm_for_text(self, system_prompt: str, user_prompt: str, json_format: bool = False) -> str:
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            if json_format:
                json_llm = self.llm.bind(response_format={"type": "json_object"})
                resp = json_llm.invoke(prompt)
            else:
                resp = self.llm.invoke(prompt)
            return resp.content
        
        except Exception as e:
            # Catch any exception during LLM invocation or token access
            self.logger.info("Error, system prompt and user prompt:")
            # self.logger.info(resp.response_metadata['token_usage']['total_tokens'])
            self.logger.info(system_prompt)
            self.logger.info(user_prompt)
            raise

    def _invoke_llm_for_multimodal(self, system_prompt: str, user_prompt: str, image_base64: str = None, json_format: bool = False) -> str:
        content_parts = []
        if image_base64:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        content_parts.append({
            "type": "text",
            "text": user_prompt
        })
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts}
        ]
        try:
            if json_format:
                json_llm = self.llm.bind(response_format={"type": "json_object"})
                resp = json_llm.invoke(prompt)
            else:
                resp = self.llm.invoke(prompt)

            return resp.content

        except Exception as e:
            # print(system_prompt)
            print(user_prompt)
            # Catch any exception during LLM invocation or token access
            self.logger.info("Error, system prompt and user prompt:")
            # self.logger.info(resp.response_metadata['token_usage']['total_tokens'])
            self.logger.info(system_prompt)
            self.logger.info(user_prompt)
            raise

    def _build_workflow(self) -> StateGraph:
        # LangGraph
        workflow = StateGraph(NewsState)
        workflow.add_node("pseudo_node", self._pseudo_label)
        workflow.add_node("domain_profile", self._domain_profile)
        workflow.add_node("debate_node", self._debate_agents)
        workflow.add_node("summarizer_node", self._authenticity_summarizer)
        workflow.add_node("classify_node", self._classify_news_multi_agent)
        workflow.set_entry_point("pseudo_node")
        workflow.add_edge("pseudo_node", "domain_profile")
        workflow.add_edge("domain_profile", "debate_node")
        workflow.add_edge("debate_node", "summarizer_node")
        workflow.add_edge("summarizer_node", "classify_node")
        workflow.add_edge("classify_node", END)
        return workflow

    # base64
    @staticmethod
    def encode_image(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def format_filtered_results(last_num: int, filtered_results: List[SearchResult]) -> str:
        if not filtered_results:
            return "No external knowledge was retrieved in this round (rely solely on internal common-sense reasoning)."
    
        # formatted = ["=== EXTERNAL KNOWLEDGE PROVIDED THIS ROUND ===\n"]
        formatted = []
        for idx, result in enumerate(filtered_results[:5], 1):
            formatted.append(
                f"[{idx+last_num}] {result.title}\n"
                f"    URL: {result.href}\n"
                f"    Excerpt: {result.body[:300]}{'...' if len(result.body) > 300 else ''}\n"
            )
        # formatted.append("=" * 50)
        return "\n".join(formatted)

    # pseudo label
    def _pseudo_label(self, state: NewsState) -> NewsState:
        DOMAINS = ["Science", "Military", "Education", "Disaster", "Politics", "Health", "Finance", "Entertainment", "Society"]
        resp = self._invoke_llm_for_text(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=PSEUDO_LABEL_USER.format(text=state['text'], domains=', '.join(DOMAINS))
        )
        try:
            domains = json.loads(resp)
        except:
            domains = [d.strip() for d in resp.split(",")][:3]
        state['domain_tags'] = domains
        return state
    
    def format_domain_tags(self, domain_tags):
        if not domain_tags:
            return "general"
        elif len(domain_tags) == 1:
            return domain_tags[0]
        elif len(domain_tags) == 2:
            return f"{domain_tags[0]} and {domain_tags[1]}"
        else:
            return ", ".join(domain_tags[:-1]) + ", and " + domain_tags[-1]

    # Domain profile
    def _domain_profile(self, state: NewsState) -> NewsState:
        domain_tags = state['domain_tags']
        formatted_tags = self.format_domain_tags(domain_tags)
        resp = self._invoke_llm_for_text(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=DOMAIN_PROFILE_USER.format(formatted_tags=formatted_tags)
        )
        state['domain_profile'] = resp
        return state

    def _analyze_image(self, state: NewsState) -> NewsState:
        resp = self._invoke_llm_for_multimodal(
            system_prompt=IMAGE_ANALYSIS_SYSTEM,
            user_prompt=IMAGE_ANALYSIS_USER,
            image_base64=state['image']
        )
        resp = json.loads(resp)
        state["image_analysis"] = resp["image_analysis"]
        state["entity"] = resp["entity"]
        return state

    # short term memory
    def _summarize_short_term_memory(self, debate_content: str) -> str:
        if not debate_content.strip():
            return ""
        user_prompt = UPDATE_SHORT_TERM_MEMORY.format(debate_content=debate_content)
        resp = self._invoke_llm_for_text(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt
        )
        return resp

    # long term memory
    def _update_long_term_memory(self, current_ltm: str, short_term_memory: str) -> str:
        if not short_term_memory.strip():
            return current_ltm
        combine_prompt = UPDATE_LONG_TERM_MEMORY.format(
            current_ltm=current_ltm or "No debate history yet.",
            short_term_memory=short_term_memory
        ).strip()
        resp = self._invoke_llm_for_text(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=combine_prompt
        )
        return resp

    def _route_verification(self, news_text: str, debate_summary: str) -> str:
        """Invoke LLM to route verification: SEARCH or INTERNAL."""
        router_system_prompt = ROUTER_SYSTEM_PROMPT.strip()
        router_user_prompt = ROUTER_USER_PROMPT.format(news_text=news_text, debate_summary=debate_summary).strip()
        resp = self._invoke_llm_for_text(
            system_prompt=router_system_prompt,
            user_prompt=router_user_prompt
        )
        return resp.strip().upper()  # Normalize to SEARCH or INTERNAL

    # Debate
    def _debate_agents(self, state: NewsState) -> NewsState:
        reverse_image = state.get('captions', '')
        reverse_image = reverse_image if reverse_image else "There are no results from reverse image search, you should rely on internal knowledge."

        domain_profile = state['domain_profile']

        state.setdefault("short_term_memories", [])
        state.setdefault("long_term_memory", "")

        # debate_rounds: List[str] = []
        debate_log: List[str] = []
        query_list: List[str] = []
        external_knowledge_list: List[str] = []
        retrieved_knowledge_only: List[str] = []

        prev_round_summary = ""
        last_num = 0
        for round_idx in range(1, self.n_rounds + 1):
            state["current_round"] = round_idx

            # Route Agent
            decision = self._route_verification(state["text"], prev_round_summary)
            if decision == "INTERNAL":
                knowledge_result = KnowledgeResult(query="", search_results=[], filtered_results=[], retrieval_time=0.0)
                formatted_knowledge = "No external search required, you should rely on internal knowledge."
            else:
                knowledge_result = self.retriever.retrieve_knowledge(state["text"], prev_round_summary)
                formatted_knowledge = self.format_filtered_results(last_num, knowledge_result.filtered_results[:5])
                last_num += min(len(knowledge_result.filtered_results), 5)
                retrieved_knowledge_only.append(formatted_knowledge)

            query_list.append(knowledge_result.query)
            external_knowledge_list.append(formatted_knowledge)

            idx = round_idx - 1
            mem = state['short_term_memories']
            short_term_context = mem[idx] if idx < len(mem) and mem[idx] else "No prior round yet."

            memory_context = {
                "long_term_memory": state.get("long_term_memory") or "No debate history yet.",
                "short_term_memories": short_term_context
            }

            # ==================== Believer ====================
            # 1. Draft Stage
            believer_draft = self._invoke_llm_for_multimodal(
                system_prompt=SYSTEM_PROMPT_DEBATE.strip(),
                user_prompt=BELIEVER_DRAFT_USER.format(
                    domain_profile=domain_profile,
                    round_idx=round_idx,
                    news_text=state["text"],
                    reverse_image=reverse_image,
                    formatted_knowledge=formatted_knowledge,
                    **memory_context
                ).strip(),
                image_base64=state['image']
            )

            # 2. Self-Criticism Stage
            believer_criticism = self._invoke_llm_for_multimodal(
                system_prompt=SYSTEM_PROMPT_DEBATE.strip(),
                user_prompt=BELIEVER_SELF_CRITICISM_USER.format(
                    news_text=state["text"],
                    believer_draft_resp=believer_draft,
                    reverse_image=reverse_image,
                    formatted_knowledge=formatted_knowledge
                ).strip(),
                image_base64=state['image']
            )

            # 3. Refine Stage
            believer_final = self._invoke_llm_for_multimodal(
                system_prompt=SYSTEM_PROMPT_DEBATE.strip(),
                user_prompt=BELIEVER_REFINE_USER.format(
                    domain_profile=domain_profile,
                    round_idx=round_idx,
                    news_text=state["text"],
                    short_term_memories=memory_context["short_term_memories"],
                    believer_draft_resp=believer_draft,
                    believer_self_criticism_resp=believer_criticism,
                    reverse_image=reverse_image,
                    formatted_knowledge=formatted_knowledge
                ).strip(),
                image_base64=state['image']
            )

            # ==================== Skeptic ====================
            skeptic_draft = self._invoke_llm_for_multimodal(
                system_prompt=SYSTEM_PROMPT_DEBATE.strip(),
                user_prompt=SKEPTIC_DRAFT_USER.format(
                    domain_profile=domain_profile,
                    round_idx=round_idx,
                    news_text=state["text"],
                    reverse_image=reverse_image,
                    formatted_knowledge=formatted_knowledge,
                    **memory_context
                ).strip(),
                image_base64=state['image']
            )

            skeptic_criticism = self._invoke_llm_for_multimodal(
                system_prompt=SYSTEM_PROMPT_DEBATE.strip(),
                user_prompt=SKEPTIC_SELF_CRITICISM_USER.format(
                    news_text=state["text"],
                    skeptic_draft_resp=skeptic_draft,
                    reverse_image=reverse_image,
                    formatted_knowledge=formatted_knowledge
                ).strip(),
                image_base64=state['image']
            )

            skeptic_final = self._invoke_llm_for_multimodal(
                system_prompt=SYSTEM_PROMPT_DEBATE.strip(),
                user_prompt=SKEPTIC_REFINE_USER.format(
                    domain_profile=domain_profile,
                    round_idx=round_idx,
                    news_text=state["text"],
                    short_term_memories=memory_context["short_term_memories"],
                    skeptic_draft_resp=skeptic_draft,
                    skeptic_self_criticism_resp=skeptic_criticism,
                    reverse_image=reverse_image,
                    formatted_knowledge=formatted_knowledge
                ).strip(),
                image_base64=state['image']
            )

            round_text = (
                f"Round {round_idx} - Believer Draft:\n{believer_draft}\n\n"
                f"Round {round_idx} - Believer Criticism:\n{believer_criticism}\n\n"
                f"Round {round_idx} - Believer Final:\n{believer_final}\n\n"
                f"Round {round_idx} - Skeptic Draft:\n{skeptic_draft}\n\n"
                f"Round {round_idx} - Skeptic Criticism:\n{skeptic_criticism}\n\n"
                f"Round {round_idx} - Skeptic Final:\n{skeptic_final}"
            )
            # debate_rounds.append(round_text)
            debate_log.append(round_text)

            current_round_debate = f"Round {round_idx}:\nBeliever:\n{believer_final}\n\nSkeptic:\n{skeptic_final}"

            # ==================== Memory update ====================
            # short term memory
            short_term_memory = self._summarize_short_term_memory(current_round_debate)
            # short_term_memory = "Round {} - Short Memory Summary: {}".format(round_idx, short_term_memory)
            state["short_term_memories"].append(short_term_memory)

            # Long term memory
            state["long_term_memory"] = self._update_long_term_memory(
                state["long_term_memory"], current_round_debate
            )
            # print(state["long_term_memory"])

            prev_round_summary = short_term_memory

        # state["debate_rounds"] = "\n".join(debate_rounds)
        state["debate_log"] = "\n".join(debate_log)
        state["query_list"] = query_list
        state["external_knowledge"] = external_knowledge_list
        state["retrieved_knowledge_only"] = retrieved_knowledge_only
        return state

    # AuthenticitySummarizer Agent
    def _authenticity_summarizer(self, state: NewsState) -> NewsState:

        text = state["text"]
        resp = self._invoke_llm_for_text(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=AuthenticitySummarizer.format(text=text, long_term_memory=state['long_term_memory'])
        )
        state["debate_summary"] = resp
        return state

    def _classify_news_multi_agent(self, state: NewsState) -> NewsState:
        text = state["text"]
        debate_summary = state["debate_summary"]
        image_base64 = state['image']
        reverse_image = state['captions']
        # external_knowledge = state["external_knowledge"]
        # image_analysis = state["image_analysis"]
        # entity = state["entity"]

        patterns = {
            "text_debate": r"\[News text-related debate\]\s*(.*?)(?=\[News image-related debate\]|\Z)",
            "image_debate": r"\[News image-related debate\]\s*(.*?)(?=\[News text-News image consistency debate\]|\Z)",
            "consistency_debate": r"\[News text-News image consistency debate\]\s*(.*?)\Z"
        }

        flags = re.DOTALL

        text_debate = re.search(patterns["text_debate"], debate_summary, flags).group(1).strip()
        image_debate = re.search(patterns["image_debate"], debate_summary, flags).group(1).strip()
        consistency_debate = re.search(patterns["consistency_debate"], debate_summary, flags).group(1).strip()

        # last_non_empty = next((item for item in reversed(external_knowledge) if item.strip()), "No external knowledge was retrieved (rely solely on internal common-sense reasoning).")

        retrieved_knowledge_only = state.get("retrieved_knowledge_only")
        retrieved_knowledge_only = ''.join(retrieved_knowledge_only)

        if not retrieved_knowledge_only:
            retrieved_knowledge_only = "No external knowledge was retrieved (rely solely on internal common-sense reasoning)."
        
        # print(retrieved_knowledge_only)
        # ============ Agent 1: Text Authenticity Agent ============
        text_agent_resp = self._invoke_llm_for_multimodal(
            system_prompt=TEXT_AGENT_SYSTEM_PROMPT.strip(),
            user_prompt=TEXT_AGENT_USER_PROMPT.format(
                text=text,
                debate_summary=text_debate,
                external_knowledge=retrieved_knowledge_only,
                reverse_image=reverse_image
            ).strip(),
            image_base64=None,
            json_format=True
        )
        text_data = self._safe_parse(text_agent_resp)
        text_authentic = text_data.get("conclusion", "unknown") == "Authentic"
        # ============ Agent 2: Image Authenticity Agent ============
        image_agent_resp = self._invoke_llm_for_multimodal(
            system_prompt=IMAGE_AGENT_SYSTEM_PROMPT.strip(),
            user_prompt=IMAGE_AGENT_USER_PROMPT.format(
                debate_summary=image_debate
            ).strip(),
            image_base64=image_base64,
            json_format=True
        )
        image_data = self._safe_parse(image_agent_resp)
        image_authentic = image_data.get("conclusion", "unknown") == "Authentic"

        # ============ Agent 3: Consistency Agent ============
        consistency_agent_resp = self._invoke_llm_for_multimodal(
            system_prompt=CONSISTENCY_AGENT_SYSTEM_PROMPT.strip(),
            user_prompt=CONSISTENCY_AGENT_USER_PROMPT.format(
                text=text,
                debate_summary=consistency_debate,
                external_knowledge=retrieved_knowledge_only,
                reverse_image=reverse_image
            ).strip(),
            image_base64=image_base64,
            json_format=True
        )
        cons_data = self._safe_parse(consistency_agent_resp)
        consistent = cons_data.get("conclusion", "unknown") == "Consistent"

        if not text_authentic:
            final_label = "textual_veracity_distortion"
        elif not image_authentic:
            final_label = "visual_veracity_distortion"
        elif not consistent:
            final_label = "mismatch"
        else:
            final_label = "original"

        # print(text_authentic)
        # print(image_authentic)
        # print(consistent)
        state.update({
            "label": final_label,
            "text_reason": text_data.get("reason", ""),
            "image_reason": image_data.get("reason", ""),
            "consistency_reason": cons_data.get("reason", "")
        })
        return state


    def _safe_parse(self, resp):
        try:
            return resp if isinstance(resp, dict) else json.loads(resp)
        except:
            return {}

    def analyze(self, text: str, image_path: str, captions: str = "") -> Dict[str, Any]:
        image_base64 = self.encode_image(image_path)
        init_state: NewsState = {
            "text": text,
            "image": image_base64,
            "image_analysis": "",
            "entity": [],
            "domain_tags": [''],
            "domain_profile": '',
            "debate_log": "",
            # "debate_rounds": "",
            "debate_summary": "",
            "label": "",
            "text_reason": "",
            "image_reason": "",
            "consistency_reason": "",
            "current_round": 0,
            "query_list": [],
            "external_knowledge": [],
            "retrieved_knowledge_only": [],
            "short_term_memories": [],
            "long_term_memory": "",
            "captions": captions  
        }
        result = self.app.invoke(init_state)
        return result