from typing import Literal

from pydantic import BaseModel, Field


class QueryIntent(BaseModel):
    """Typed parse of a free-text shopping query."""

    route: Literal["search", "recommend", "similar_to"] = "search"
    text: str = ""  # the product phrase to search for (English); for similar_to, the reference item
    price_min: float | None = None
    price_max: float | None = None
    brand: str | None = None
    category: str | None = None
    min_rating: float | None = Field(default=None, ge=0, le=5)
    must_have: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
