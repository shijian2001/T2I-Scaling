import json
import re
from typing import Dict, Any, Optional


class JSONParser:

    @staticmethod
    def parse(response: str) -> Optional[Dict[str, Any]]:

        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response
        try:
            parsed_data = json.loads(json_str)
            if isinstance(parsed_data, dict):
                return parsed_data
            return None
        except json.JSONDecodeError:
            return None