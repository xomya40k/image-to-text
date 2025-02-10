from pydantic import BaseModel


class ExtractionBase(BaseModel):
    status: str


class ExtractionResponse(ExtractionBase):
    text: str
    image: str | None = None