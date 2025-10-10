from .predictor import MultimodalSignLM

try:
    from .predict_keypoint import KeyPointDetect
except ModuleNotFoundError:
    KeyPointDetect = None
