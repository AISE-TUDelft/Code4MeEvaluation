from CodeGPT import create_predict_fn

codegpt_py = {
    "name": "CodeGPT-small-py-adaptedGPT2",
    "generate": create_predict_fn("microsoft/CodeGPT-small-py-adaptedGPT2"),
    "supports_left_context": True,
    "supports_right_context": False,
}
