"""Connect enterprise planning to the already configured compatible Qwen service."""
import json

from rag_assist.localstack.llm import QwenClient,ModelUnavailable
from .gateway import Gateway


class PlanningQwenClient(QwenClient):
    """Keep the existing answer/verification protocol; share the enterprise planner."""

    def _json(self,system,user,max_tokens=1600,*,schema=None,model=None):
        if model is not None and model!=self.model:
            raise ModelUnavailable("MODEL_ALIAS_NOT_CONFIGURED")
        if schema is not None:
            # This compatible endpoint uses JSON-object output; Pydantic checks it afterwards.
            system+="\n输出必须遵守JSON Schema："+json.dumps(schema,ensure_ascii=False)
        content=user if isinstance(user,str) else json.dumps(user,ensure_ascii=False,allow_nan=False)
        return super()._json(system,content,max_tokens)

    def interpret(self,question,context,catalog):
        return Gateway.interpret(self,question,context,catalog)

    def hypothesis(self,question,anchors):
        return Gateway.hypothesis(self,question,anchors)
