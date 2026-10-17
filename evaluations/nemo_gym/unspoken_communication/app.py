"""NeMo Gym resources server: `unspoken_communication`.

Evaluates (and can train) a coordination policy for assistive communication:
one typed tool call per task, scored on decoupled valid / action / safety
rewards (see scoring.py). Drop this folder into Gym's `resources_servers/`.
"""

from typing import Any, Dict, List

from pydantic import Field

from nemo_gym.base_resources_server import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    SimpleResourcesServer,
)

try:
    from .scoring import score
except ImportError:  # run as a script from this folder
    from scoring import score


class UnspokenConfig(BaseResourcesServerConfig):
    pass


class UnspokenVerifyRequest(BaseVerifyRequest):
    category: str = ""
    plan_request: Dict[str, Any] = Field(default_factory=dict)
    expected: Dict[str, Any] = Field(default_factory=dict)


class UnspokenVerifyResponse(BaseMultiRewardVerifyResponse):
    valid: float = 0.0
    action: float = 0.0
    safety: float = 0.0
    category: str = ""
    reasons: List[str] = Field(default_factory=list)


class UnspokenResourcesServer(SimpleResourcesServer):
    config: UnspokenConfig

    async def verify(self, body: UnspokenVerifyRequest) -> UnspokenVerifyResponse:
        calls = [o for o in body.response.output if getattr(o, "type", None) == "function_call"]
        call = {"name": calls[0].name, "arguments": calls[0].arguments} if calls else None
        s = score({"plan_request": body.plan_request, "expected": body.expected}, call)
        if len(calls) > 1:  # exactly one step per decision
            s["valid"] = 0.0
            s["reasons"].append("more than one tool call")
        components = {"valid": s["valid"], "action": s["action"], "safety": s["safety"]}
        return UnspokenVerifyResponse(
            **body.model_dump(), reward=sum(components.values()), reward_components=components,
            reasons=s["reasons"], **components,
        )


if __name__ == "__main__":
    UnspokenResourcesServer.run_webserver()
