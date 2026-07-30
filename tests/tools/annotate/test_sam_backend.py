from pathlib import Path

from tools.annotate.backends.sam import SAMTextDetector


def test_sam_text_detector_disables_ultralytics_output() -> None:
    class FakePredictor:
        def __init__(self, overrides: dict) -> None:
            self.overrides = overrides

    detector = SAMTextDetector(
        model_path=Path("weight/sam.pt"),
        label="hand",
        conf=0.25,
        prompt="hand",
        predictor_cls=FakePredictor,
    )

    assert detector.predictor.overrides["save"] is False
    assert detector.predictor.overrides["verbose"] is False
