from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """
    Public health status returned by the inference service.

    This endpoint confirms that the HTTP application is available. It does
    not load the model or execute an ECG analysis.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ok"]
    service: str
    version: str
