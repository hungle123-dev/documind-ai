from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from typing import Dict, List
from langchain.schema import Document
from config.settings import settings
from loguru import logger

credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
client = APIClient(credentials)


class ResearchAgent:
    def __init__(self):
        logger.info("Initializing ResearchAgent...")
        self.model = ModelInference(
            model_id="meta-llama/llama-3-2-90b-vision-instruct",
            credentials=credentials,
            project_id="skills-network",
            params={
                "max_tokens": 300,
                "temperature": 0.3,
            }
        )
        logger.info("ResearchAgent initialized.")

    def sanitize_response(self, response_text: str) -> str:
        return response_text.strip()

    def generate_prompt(self, question: str, context: str) -> str:
        return f"""You are an AI assistant designed to provide precise and factual answers based on the given context.

**Instructions:**
- Answer the following question using only the provided context.
- Be clear, concise, and factual.
- Return as much information as you can get from the context.

**Question:** {question}
**Context:**
{context}

**Provide your answer below:**"""

    def generate(self, question: str, documents: List[Document]) -> Dict:
        logger.debug(f"ResearchAgent.generate | question='{question}' | docs={len(documents)}")

        context = "\n\n".join([doc.page_content for doc in documents])
        logger.debug(f"Context length: {len(context)} chars")

        prompt = self.generate_prompt(question, context)

        try:
            response = self.model.chat(
                messages=[{"role": "user", "content": prompt}]
            )
            logger.debug("LLM response received.")
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            raise RuntimeError("Failed to generate answer.") from e

        try:
            llm_response = response['choices'][0]['message']['content'].strip()
            logger.debug(f"Raw response preview: {llm_response[:120]}...")
        except (IndexError, KeyError) as e:
            logger.error(f"Unexpected response structure: {e}")
            llm_response = "I cannot answer this question based on the provided documents."

        draft_answer = self.sanitize_response(llm_response) if llm_response \
            else "I cannot answer this question based on the provided documents."

        logger.info(f"Answer generated ({len(draft_answer)} chars)")

        return {
            "draft_answer": draft_answer,
            "context_used": context,
        }
