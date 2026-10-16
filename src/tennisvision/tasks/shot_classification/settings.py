import os
from dataclasses import dataclass

from tennisvision.core.settings import RuntimeSettings

@dataclass(frozen=True)
class ShotClassificationSettings:
    runtime: RuntimeSettings

    model_uri: str | None
    run_id: str | None
    model_name: str


def load_shot_classification_settings(
    runtime: RuntimeSettings,
) -> ShotClassificationSettings:
    return ShotClassificationSettings(
        runtime=runtime,
        model_uri=os.getenv(
            "SHOT_CLASSIFICATION_MODEL_URI"
        ) or None,
        run_id=os.getenv(
            "SHOT_CLASSIFICATION_RUN_ID"
        ) or None,
        model_name=os.getenv(
            "SHOT_CLASSIFICATION_MODEL_NAME",
            "mobilenet_v3_large",
        ),
    )
