from __future__ import annotations

from textvqa_proj.models.base import BaseModelAdapter
from textvqa_proj.models.blip2 import Blip2Adapter
from textvqa_proj.models.fake import FakeAnsweringAdapter
from textvqa_proj.models.internvl import InternVL25Adapter
from textvqa_proj.models.llava import LlavaHFAdapter
from textvqa_proj.models.ocr_lexical import OCRLexicalAdapter
from textvqa_proj.models.qwen_vl import Qwen25VLAdapter

MODEL_REGISTRY: dict[str, type[BaseModelAdapter]] = {
    FakeAnsweringAdapter.adapter_name: FakeAnsweringAdapter,
    Qwen25VLAdapter.adapter_name: Qwen25VLAdapter,
    Blip2Adapter.adapter_name: Blip2Adapter,
    LlavaHFAdapter.adapter_name: LlavaHFAdapter,
    InternVL25Adapter.adapter_name: InternVL25Adapter,
    OCRLexicalAdapter.adapter_name: OCRLexicalAdapter,
}


def create_adapter(adapter_name: str, settings: object) -> BaseModelAdapter:
    try:
        adapter_cls = MODEL_REGISTRY[adapter_name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown adapter {adapter_name!r}. Available: {available}") from exc
    return adapter_cls(settings)
