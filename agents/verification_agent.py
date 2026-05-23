from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from typing import Dict, List
from langchain.schema import Document
from loguru import logger

credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
client = APIClient(credentials)


class VerificationAgent:
    def __init__(self):
        logger.info("Initializing VerificationAgent...")
        self.model = ModelInference(
            model_id="ibm/granite-4-h-small",
            credentials=credentials,
            project_id="skills-network",
            params={
                "max_tokens": 200,
                "temperature": 0.0,
            }
        )
        logger.info("VerificationAgent initialized.")

    def sanitize_response(self, response_text: str) -> str:
        return response_text.strip()

    def generate_prompt(self, answer: str, context: str) -> str:
        return f"""You are an AI assistant designed to verify the accuracy and relevance of answers based on provided context.

**Instructions:**
- Verify the following answer against the provided context.
- Check for:
  1. Direct/indirect factual support (YES/NO)
  2. Unsupported claims (list any if present)
  3. Contradictions (list any if present)
  4. Relevance to the question (YES/NO)
- Respond in the exact format specified below without adding any unrelated information.

**Format:**
Supported: YES/NO
Unsupported Claims: [item1, item2, ...]
Contradictions: [item1, item2, ...]
Relevant: YES/NO
Additional Details: [Any extra information or explanations]

**Answer:** {answer}
**Context:**
{context}

**Respond ONLY with the above format.**"""

    def parse_verification_response(self, response_text: str) -> Dict:
        try:
            lines = response_text.split('\n')
            verification = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().capitalize()
                    value = value.strip()
                    if key in {"Supported", "Unsupported claims", "Contradictions", "Relevant", "Additional details"}:
                        if key in {"Unsupported claims", "Contradictions"}:
                            if value.startswith('[') and value.endswith(']'):
                                items = value[1:-1].split(',')
                                verification[key] = [i.strip().strip('"').strip("'") for i in items if i.strip()]
                            else:
                                verification[key] = []
                        elif key == "Additional details":
                            verification[key] = value
                        else:
                            verification[key] = value.upper()
            for key in ["Supported", "Unsupported Claims", "Contradictions", "Relevant", "Additional Details"]:
                if key not in verification:
                    if key in {"Unsupported Claims", "Contradictions"}:
                        verification[key] = []
                    elif key == "Additional Details":
                        verification[key] = ""
                    else:
                        verification[key] = "NO"
            return verification
        except Exception as e:
            logger.error(f"Error parsing verification response: {e}")
            return None

    def format_verification_report(self, verification: Dict) -> str:
        supported = verification.get("Supported", "NO")
        unsupported_claims = verification.get("Unsupported Claims", [])
        contradictions = verification.get("Contradictions", [])
        relevant = verification.get("Relevant", "NO")
        additional_details = verification.get("Additional Details", "")

        report = f"**Supported:** {supported}\n"
        report += f"**Unsupported Claims:** {', '.join(unsupported_claims) if unsupported_claims else 'None'}\n"
        report += f"**Contradictions:** {', '.join(contradictions) if contradictions else 'None'}\n"
        report += f"**Relevant:** {relevant}\n"
        report += f"**Additional Details:** {additional_details if additional_details else 'None'}\n"
        return report

    def check(self, answer: str, documents: List[Document]) -> Dict:
        logger.debug(f"VerificationAgent.check | answer_len={len(answer)} | docs={len(documents)}")

        context = "\n\n".join([doc.page_content for doc in documents])
        logger.debug(f"Context length: {len(context)} chars")

        prompt = self.generate_prompt(answer, context)

        try:
            response = self.model.chat(
                messages=[{"role": "user", "content": prompt}]
            )
            logger.debug("LLM response received.")
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            raise RuntimeError("Failed to verify answer.") from e

        try:
            llm_response = response['choices'][0]['message']['content'].strip()
            logger.debug(f"Raw response preview: {llm_response[:120]}...")
        except (IndexError, KeyError) as e:
            logger.error(f"Unexpected response structure: {e}")
            verification_report = {
                "Supported": "NO", "Unsupported Claims": [],
                "Contradictions": [], "Relevant": "NO",
                "Additional Details": "Invalid response structure from the model."
            }
            return {
                "verification_report": self.format_verification_report(verification_report),
                "context_used": context,
            }

        sanitized = self.sanitize_response(llm_response) if llm_response else ""

        if not sanitized:
            logger.warning("LLM returned empty response.")
            verification_report = {
                "Supported": "NO", "Unsupported Claims": [],
                "Contradictions": [], "Relevant": "NO",
                "Additional Details": "Empty response from the model."
            }
        else:
            verification_report = self.parse_verification_response(sanitized)
            if verification_report is None:
                logger.warning("LLM response did not match expected format — using default.")
                verification_report = {
                    "Supported": "NO", "Unsupported Claims": [],
                    "Contradictions": [], "Relevant": "NO",
                    "Additional Details": "Failed to parse the model's response."
                }

        formatted = self.format_verification_report(verification_report)
        logger.debug(f"Verification report:\n{formatted}")

        return {
            "verification_report": formatted,
            "context_used": context,
        }
