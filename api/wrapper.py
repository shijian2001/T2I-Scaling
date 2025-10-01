from openai import AsyncOpenAI
from typing import Dict, Any
import logging
import asyncio

logger = logging.getLogger(__name__)


class QAWrapper:

    SUPPORTED_REASONING_MODELS = [""]

    def __init__(self, model_name: str, api_key: str, max_retries: int = 5):

        self.model_name = model_name
        self.api_key = api_key
        self.max_retries = max_retries
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=""
        )

        self.stats = {
            "calls": 0,
            "errors": 0,
            "retries": 0
        }

    async def qa(self, system_prompt: str, user_prompt: str = "", rational: bool = False) -> Any:
        if rational and self.model_name not in self.SUPPORTED_REASONING_MODELS:
            raise ValueError(f"Model {self.model_name} does not support reasoning")
        for attempt in range(self.max_retries):
            try:
                if rational:
                    return await self._qa_with_reasoning(system_prompt, user_prompt)
                else:
                    return await self._qa_standard(system_prompt, user_prompt)

            except Exception as e:
                self.stats["errors"] += 1
                self.stats["retries"] += 1

                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")

                if attempt == self.max_retries - 1:
                    raise
                retry_delay = 2 ** attempt
                await asyncio.sleep(retry_delay)

    async def _qa_standard(self, system_prompt: str, user_prompt: str) -> Dict[str, str]:
        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': user_prompt
                }
            ],
            stream=False,
            temperature=1
        )

        self.stats["calls"] += 1
        return {
            "answer": completion.choices[0].message.content,
            "rational": ""
        }

    async def _qa_with_reasoning(self, system_prompt: str, user_prompt: str) -> Dict[str, str]:
        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': user_prompt
                },
                {
                    "role": "assistant",
                    "content": "<think>\n"
                }
            ],
            stream=False,
            temperature=1
        )

        self.stats["calls"] += 1
        return {
            "answer": completion.choices[0].message.content,
            "rational": completion.choices[0].message.reasoning_content
        }

    def get_stats(self) -> Dict[str, int]:
        return self.stats.copy()