from __future__ import annotations

import sys
import types
from pathlib import Path

from textvqa_proj.config import Settings
from textvqa_proj.models.qwen_vl import Qwen25VLAdapter


def test_qwen_adapter_loads_saved_lora_adapter_and_processor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_calls: list[tuple[str, dict[str, object]]] = []
    processor_calls: list[tuple[str, dict[str, object]]] = []
    peft_calls: list[tuple[object, str, bool]] = []

    class FakeModel:
        def __init__(self) -> None:
            self.device = None
            self.eval_called = False
            self.dtype = object()

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            model_calls.append((source, kwargs))
            return cls()

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = types.SimpleNamespace(padding_side=None)

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            processor_calls.append((source, kwargs))
            return cls()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter_path, is_trainable=False):
            peft_calls.append((model, adapter_path, is_trainable))
            return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            Qwen2_5_VLForConditionalGeneration=FakeModel,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(PeftModel=FakePeftModel),
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        types.SimpleNamespace(process_vision_info=lambda conversations: ([], [])),
    )

    import textvqa_proj.models.qwen_vl as qwen_vl

    monkeypatch.setattr(
        qwen_vl,
        "resolve_pretrained_source",
        lambda repo_or_path, **_: str(tmp_path / "base-snapshot"),
    )

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    processor_dir = tmp_path / "processor"
    processor_dir.mkdir()

    settings = Settings()
    settings.runtime.device_order = ["cpu"]
    settings.model.adapter = "qwen2_5_vl"
    settings.model.model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    settings.model.processor_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    settings.model.adapter_path = str(adapter_dir)
    settings.model.processor_path = str(processor_dir)

    adapter = Qwen25VLAdapter(settings)
    adapter.load()

    assert model_calls == [
        (
            str(tmp_path / "base-snapshot"),
            {
                "torch_dtype": __import__("torch").float16,
                "trust_remote_code": False,
                "low_cpu_mem_usage": True,
                "local_files_only": True,
            },
        )
    ]
    assert processor_calls == [
        (
            str(processor_dir),
            {
                "local_files_only": True,
            },
        )
    ]
    assert peft_calls == [(adapter._model, str(adapter_dir), False)]
    assert adapter._model.device == "cpu"
    assert adapter._model.eval_called is True
    assert adapter._processor.tokenizer.padding_side == "left"
