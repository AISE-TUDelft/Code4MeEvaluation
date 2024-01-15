from CodeGPT import create_predict_fn

codegpt_java = {
    "name": "CodeGPT-small-java-adaptedGPT2",
    "generate": create_predict_fn("microsoft/CodeGPT-small-java-adaptedGPT2", True),
    "supports_left_context": True,
    "supports_right_context": False,
}
