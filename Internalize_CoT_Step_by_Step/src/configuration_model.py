from transformers import PretrainedConfig

class ImplicitModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        gpt2_config=None,
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = base_model
        self.gpt2_config = gpt2_config
        super().__init__(**kwargs)

