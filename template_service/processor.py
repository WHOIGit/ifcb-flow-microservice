"""Stateless template processor."""

from typing import List

from pydantic import BaseModel, Field

from stateless_microservice import BaseProcessor, StatelessAction


class EchoRequest(BaseModel):
    """Request payload for echo service."""

    content: str = Field(..., description="Input payload string.")


class EchoProcessor(BaseProcessor):
    """Processor for echoing requests."""

    @property
    def name(self) -> str:
        return "echo"

    def get_stateless_actions(self) -> List[StatelessAction]:
        return [
            StatelessAction(
                name="echo",
                path="/echo",
                request_model=EchoRequest,
                handler=self.handle_echo,
                summary="Echo the incoming request.",
                description="Echo the incoming request.",
                tags=("echo",),
                media_type="text/plain",
            ),
        ]

    async def handle_echo(self, payload: EchoRequest):
        """Echo the incoming request."""

        return payload.content
