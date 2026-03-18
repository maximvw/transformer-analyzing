from transformers import PretrainedConfig

class ImplicitModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        gpt2_config=None,
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = base_model if isinstance(base_model, str) else 'gpt2'
        self.gpt2_config = gpt2_config
        super().__init__(**kwargs)

