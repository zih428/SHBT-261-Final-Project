from textvqa_proj.config import Settings
from textvqa_proj.training.runner import _build_training_arguments_kwargs


def test_training_arguments_kwargs_supports_legacy_trainingarguments() -> None:
    settings = Settings()

    kwargs = _build_training_arguments_kwargs(
        settings,
        output_dir="out",
        has_eval=True,
        dataloader_num_workers=4,
        device="mps",
        accepted_names={
            "output_dir",
            "evaluation_strategy",
            "use_mps_device",
        },
    )

    assert kwargs["evaluation_strategy"] == "steps"
    assert "eval_strategy" not in kwargs
    assert kwargs["use_mps_device"] is True


def test_training_arguments_kwargs_supports_modern_trainingarguments() -> None:
    settings = Settings()

    kwargs = _build_training_arguments_kwargs(
        settings,
        output_dir="out",
        has_eval=False,
        dataloader_num_workers=4,
        device="mps",
        accepted_names={
            "output_dir",
            "eval_strategy",
        },
    )

    assert kwargs["eval_strategy"] == "no"
    assert "evaluation_strategy" not in kwargs
    assert "use_mps_device" not in kwargs
