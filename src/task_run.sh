CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python UrbanFeel_main.py \
        --model-type gemma3  \
        --model_name google/gemma-3-4b-it \
        --task_name ComparativePerceptualAnalysis \
        --num-images 1 \
        --gpus-num 1 \
        --seed 42 \
        --image-folder /path/to/UrbanFeel/images \
        --json-path /path/to/UrbanFeel/json/CoLocationRecognition.json  \
        --output-path /path/to/UrbanFeel/outputs \
        --checkpoint-path # path to sace checkpoint files. default UrbanFeel/checkpoints


# task_map = {
#     "CoLocationRecognition": CoLocationRecognition,
#     "DominantElementExtraction": DominantElementExtraction,
#     "SingleToPanoMatching": SingleToPanoMatching,
#     "TemporalCoLocationRecognition": TemporalCoLocationRecognition,
#     "FutureSceneIdentification": FutureSceneIdentification,
#     "PixelChangeRecognition": PixelChangeRecognition,
#     "TemporalSequenceReasoning": TemporalSequenceReasoning,
#     "SceneLevelChangeRecognition": SceneLevelChangeRecognition,
#     "GlobalPerception": GlobalPerception,
#     "LocalPerception": LocalPerception,
#     "ComparativePerceptualAnalysis": ComparativePerceptualAnalysis
# }

