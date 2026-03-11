from models import *

from tasks.CoLocationRecognition import CoLocationRecognition
from tasks.DominantElementExtraction import DominantElementExtraction
from tasks.SingleToPanoMatching import SingleToPanoMatching
from tasks.TemporalCoLocationRecognition import TemporalCoLocationRecognition
from tasks.FutureSceneIdentification import FutureSceneIdentification
from tasks.PixelChangeRecognition import PixelChangeRecognition
from tasks.TemporalSequenceReasoning import TemporalSequenceReasoning
from tasks.SceneLevelChangeRecognition import SceneLevelChangeRecognition
from tasks.GlobalPerception import GlobalPerception
from tasks.LocalPerception import LocalPerception
from tasks.ComparativePerceptualAnalysis import ComparativePerceptualAnalysis

task_map = {
    "CoLocationRecognition": CoLocationRecognition,
    "DominantElementExtraction": DominantElementExtraction,
    "SingleToPanoMatching": SingleToPanoMatching,
    "TemporalCoLocationRecognition": TemporalCoLocationRecognition,
    "FutureSceneIdentification": FutureSceneIdentification,
    "PixelChangeRecognition": PixelChangeRecognition,
    "TemporalSequenceReasoning": TemporalSequenceReasoning,
    "SceneLevelChangeRecognition": SceneLevelChangeRecognition,
    "GlobalPerception": GlobalPerception,
    "LocalPerception": LocalPerception,
    "ComparativePerceptualAnalysis": ComparativePerceptualAnalysis
}

