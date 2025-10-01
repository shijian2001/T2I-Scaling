import asyncio
import random
from typing import List, Dict, Any, TypeVar
import logging
from .wrapper import QAWrapper

T = TypeVar('T')
logger = logging.getLogger(__name__)


class APIPool:
    def __init__(
            self,
            model_name: str,
            api_keys: List[str],
            max_concurrent_per_key: int = 300,
    ):
        if not api_keys:
            raise ValueError("At least one API key is required")

        self.api_instances = [
            QAWrapper(model_name, api_key)
            for api_key in api_keys
        ]

        self.active_requests = [0] * len(self.api_instances)

        self.max_concurrent_per_key = max_concurrent_per_key

        self.semaphores = [
            asyncio.Semaphore(max_concurrent_per_key)
            for _ in range(len(self.api_instances))
        ]

        self.stats = {
            "total_calls": 0,
            "total_errors": 0,
            "total_retries": 0,
            "api_instances": len(self.api_instances),
            "api_distribution": [0] * len(self.api_instances)
        }

        # Lock for updating stats
        self.stats_lock = asyncio.Lock()

        logger.info(f"Initialized API pool with {len(api_keys)} API keys, "
                    f"{max_concurrent_per_key} max concurrent requests per key")

    @property
    def len_keys(self) -> int:

        return len(self.api_instances)

    async def execute(self, method_name: str, *args, **kwargs) -> Any:
        api_index = await self._select_optimal_api_instance()

        async with self.stats_lock:
            self.stats["api_distribution"][api_index] += 1

        async with self.semaphores[api_index]:
            self.active_requests[api_index] += 1

            try:
                api_instance = self.api_instances[api_index]
                method = getattr(api_instance, method_name)

                result = await method(*args, **kwargs)

                async with self.stats_lock:
                    self.stats["total_calls"] += 1

                return result

            except Exception as e:
                async with self.stats_lock:
                    self.stats["total_errors"] += 1

                raise

            finally:
                self.active_requests[api_index] -= 1

    async def _select_optimal_api_instance(self) -> int:
        min_active = min(self.active_requests)
        candidates = [
            i for i, count in enumerate(self.active_requests)
            if count == min_active
        ]

        return random.choice(candidates)

    async def get_stats(self) -> Dict[str, Any]:
        for i, api in enumerate(self.api_instances):
            instance_stats = api.get_stats()
            async with self.stats_lock:
                self.stats[f"instance_{i}_calls"] = instance_stats["calls"]
                self.stats[f"instance_{i}_errors"] = instance_stats["errors"]

        async with self.stats_lock:
            return self.stats.copy()