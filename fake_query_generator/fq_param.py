from typing import Any
from typing import Callable
import dataclasses
import urllib.parse


@dataclasses.dataclass
class FQParam:
    name: str
    faker_method: Callable[[], Any]
    param_appearance_rate: float = 1.0
    do_url_encode: bool = True

    def generate(self) -> str:
        if self.do_url_encode:
            return f"{self.name}={urllib.parse.quote(self.faker_method())}"
        return f"{self.name}={self.faker_method()}"
