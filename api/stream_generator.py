import asyncio
from typing import AsyncGenerator, List, Dict, Any, Optional, Callable
import logging
from .async_pool import APIPool

logger = logging.getLogger(__name__)


class StreamGenerator:

    def __init__(
            self,
            model_name: str,
            api_keys: List[str],
            max_concurrent_per_key: int = 300,
            max_retries: int = 5,
            rational: bool = False
    ):
        self.api_pool = APIPool(
            model_name=model_name,
            api_keys=api_keys,
            max_concurrent_per_key=max_concurrent_per_key
        )
        self.max_retries = max_retries
        self.rational = rational

        self.total_concurrency = max_concurrent_per_key * len(api_keys)
        self.semaphore = asyncio.Semaphore(self.total_concurrency)

        logger.info(f"Initialized StreamGenerator with total concurrency: {self.total_concurrency}")

    async def generate_stream(
            self,
            prompts: List[str],
            system_prompt: str = "",
            validate_func: Optional[Callable[[str], Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        tasks = set()

        try:
            for prompt in prompts:
                task = asyncio.create_task(
                    self._generate_single(
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        validate_func=validate_func
                    )
                )
                tasks.add(task)
                task.add_done_callback(tasks.discard)

                if len(tasks) >= self.total_concurrency:
                    done, _ = await asyncio.wait(
                        tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for completed_task in done:
                        result = await completed_task
                        if result is not None:
                            yield result

            while tasks:
                done, _ = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                for completed_task in done:
                    result = await completed_task
                    if result is not None:
                        yield result

        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def generate_stream_with_index(
            self,
            prompts_with_index: List[tuple],
            system_prompt: str = "",
            validate_func: Optional[Callable[[str], Any]] = None
    ) -> AsyncGenerator[tuple, None]:

        tasks = set()

        try:
            for index, prompt in prompts_with_index:
                task = asyncio.create_task(
                    self._generate_single_with_index(
                        index=index,
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        validate_func=validate_func
                    )
                )
                tasks.add(task)
                task.add_done_callback(tasks.discard)

                if len(tasks) >= self.total_concurrency:
                    done, _ = await asyncio.wait(
                        tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for completed_task in done:
                        result = await completed_task
                        if result is not None:
                            yield result

            while tasks:
                done, _ = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                for completed_task in done:
                    result = await completed_task
                    if result is not None:
                        yield result

        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _generate_single_with_index(
            self,
            index: int,
            system_prompt: str,
            user_prompt: str,
            validate_func: Optional[Callable[[str], Any]] = None
    ) -> Optional[tuple]:

        retry_count = 0
        while retry_count < self.max_retries:
            async with self.semaphore:
                try:
                    response = await self.api_pool.execute(
                        "qa",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        rational=self.rational
                    )

                    answer = response["answer"]

                    if validate_func is not None:
                        if not validate_func(answer):
                            logger.warning(f"Answer validation failed, retrying (attempt {retry_count + 1})")
                            retry_count += 1
                            continue
                        else:
                            answer = validate_func(answer)
                   
                    return index, answer

                except Exception as e:
                    logger.warning(f"Generation error: {e}, retrying (attempt {retry_count + 1})")
                    retry_count += 1

        logger.error(f"Max retries reached for prompt: {user_prompt}")
        return None

    async def _generate_single(
            self,
            system_prompt: str,
            user_prompt: str,
            validate_func: Optional[Callable[[str], Any]] = None
    ) -> Optional[Dict[str, Any]]:

        retry_count = 0
        while retry_count < self.max_retries:
            async with self.semaphore:
                try:
                    response = await self.api_pool.execute(
                        "qa",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        rational=self.rational
                    )

                    answer = response["answer"]

                    # If validation function exists, validate the answer
                    if validate_func is not None:
                        if not validate_func(answer):
                            logger.warning(f"Answer validation failed, retrying (attempt {retry_count + 1})")
                            retry_count += 1
                            continue
                        else:
                            answer = validate_func(answer)

                    return answer

                except Exception as e:
                    logger.warning(f"Generation error: {e}, retrying (attempt {retry_count + 1})")
                    retry_count += 1

        logger.error(f"Max retries reached for prompt: {user_prompt}")
        return None