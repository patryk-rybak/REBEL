"""
Defines the base class for attacks. Every attack must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAttack(ABC):
    def __init__(self):
        self.logs = []

    @abstractmethod
    def run(self, target, hacker, judge, data, *args, **kwargs) -> Dict[str, Any]:
        """
        General attack loop.
        """
        pass
