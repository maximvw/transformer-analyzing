import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    StoppingCriteriaList,
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteria,
    LogitsProcessor,
    GPT2Config,
    GPT2LMHeadModel,
    PretrainedConfig,
)


def get_sep_position(input_ids, sep_id, skip=0):
    batch_size = input_ids.shape[0]
    sep_positions = input_ids.new_zeros(batch_size).long()
    for batch_id in range(batch_size):
        mask = input_ids[batch_id].eq(sep_id)
        sep_position = mask.nonzero()[0, -1].item()
        for _ in range(skip):
            mask[sep_position] = False
            sep_position = mask.nonzero()[0, -1].item()
        sep_positions[batch_id] = sep_position
    return sep_positions


class DoubleEOSStoppingCriteria(StoppingCriteria):
    """
    Stop generation only after generating two EOSs, such as  z <eos> y <eos>
    """

    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.init = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        eos_count = (input_ids == self.eos_token_id).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        return done.all()


class DoubleEOSLogitsProcessor(LogitsProcessor):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.init = False

    def __call__(self, input_ids, scores):
        eos_count = (input_ids == self.eos_token_id).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        if done.any():
            scores[done, :] = float("-inf")
            scores[done, self.eos_token_id] = 0
        return scores


class ImplicitModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model="gpt2",
        gpt2_config=None,
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = base_model
        self.gpt2_config = gpt2_config
        super().__init__(**kwargs)


class ImplicitModel(nn.Module):
    def __init__(self, config, reinitialize_weights=False):
        super().__init__()
        self.config = config  # Store the config as an attribute

        # Create GPT2Config from the base_model configuration
        if isinstance(config.base_model, dict):
            gpt2_config = GPT2Config(**config.base_model)
        elif isinstance(config.base_model, str):
            # If base_model is a string (e.g., 'gpt2'), use from_pretrained
            self.base_model = GPT2LMHeadModel.from_pretrained(config.base_model)
            gpt2_config = self.base_model.config
        else:
            # Ensure we have a GPT2Config object
            if not isinstance(config.base_model, GPT2Config):
                gpt2_config = GPT2Config(
                    n_layers=12,
                    n_head=12,
                    n_embd=768,
                )
            else:
                gpt2_config = config.base_model

        # Only create base_model if not already created (in the str case)
        if not hasattr(self, "base_model"):
            self.base_model = GPT2LMHeadModel(gpt2_config)

        if reinitialize_weights:
            print("Reinitializing model weights!")
            self.base_model.apply(self.base_model._init_weights)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def forward(self, input_ids, position_ids=None, output_attentions=False):
        if position_ids is not None:
            outputs = self.base_model.forward(
                input_ids=input_ids,
                output_attentions=output_attentions,
                position_ids=position_ids,
            )
        else:
            outputs = self.base_model.forward(
                input_ids=input_ids, output_attentions=output_attentions
            )
        return outputs

    def compute_loss(
        self, input_ids, labels, position_ids=None, output_attentions=False
    ):
        outputs = self.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        logits = outputs.logits

        labels_pred = logits.argmax(-1)
        mask = labels[..., 1:].ge(0)
        correct_tokens = ((labels_pred[..., :-1] == labels[..., 1:]) * mask).sum()
        total_tokens = mask.sum()
        token_accuracy = correct_tokens / total_tokens

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        outputs.loss = loss
        outputs.token_accuracy = token_accuracy
        outputs.total_correct = correct_tokens
        outputs.total_loss = loss * total_tokens
        outputs.total_tokens = total_tokens
        return outputs

    def generate(
        self,
        input_ids,
        max_new_tokens=512,
        num_beams=1,
        stop_on_two_eos=True,
        position_ids=None,
    ):
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        batch_size = input_ids.shape[0]

        # Since there's one eos after CoT and another after final answer, we need to wait for two eos
        generation_config = GenerationConfig.from_model_config(self.base_model.config)
        if hasattr(generation_config, "pad_token_id"):
            # generation_config.pad_token_id = -1 #TODO: this might not be necessary
            generation_config.pad_token_id = None  # TODO: this might not be necessary
        if stop_on_two_eos:
            generation_config.eos_token_id = -1
            logits_processor = LogitsProcessorList(
                [DoubleEOSLogitsProcessor(self.tokenizer.eos_token_id)]
            )
            stopping_criteria = StoppingCriteriaList(
                [DoubleEOSStoppingCriteria(self.tokenizer.eos_token_id)]
            )
        else:
            logits_processor = None
            stopping_criteria = None

        if sep_positions.eq(sep_positions[0]).all():
            input_ids = input_ids[:, : sep_positions[0] + 1]
            if position_ids is not None:
                position_ids = position_ids[:, : sep_positions[0] + 1]
            if position_ids is not None:
                beam_output = self.base_model.generate(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=True,
                    num_return_sequences=1,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                )
            else:
                beam_output = self.base_model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=True,
                    num_return_sequences=1,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                )

            # Not sure why this is needed. Commenting out.
            # beam_output = beam_output.unsqueeze(1)
        else:
            beam_output = []
            for i in range(batch_size):
                input_ids_i = input_ids[i : i + 1]
                sep_positions_i = sep_positions[i : i + 1]
                input_ids_i = input_ids_i[:, : sep_positions_i + 1]
                if position_ids is not None:
                    position_ids_i = position_ids[i : i + 1, : sep_positions_i + 1]
                else:
                    position_ids_i = None
                if position_ids_i is not None:
                    beam_output_i = self.base_model.generate(
                        input_ids=input_ids_i,
                        position_ids=position_ids_i,
                        generation_config=generation_config,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        early_stopping=True,
                        num_return_sequences=1,
                        logits_processor=logits_processor,
                        stopping_criteria=stopping_criteria,
                    )
                else:
                    beam_output_i = self.base_model.generate(
                        input_ids=input_ids_i,
                        generation_config=generation_config,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        early_stopping=True,
                        num_return_sequences=1,
                        logits_processor=logits_processor,
                        stopping_criteria=stopping_criteria,
                    )
                beam_output.append(beam_output_i)

        return beam_output

    @classmethod
    def from_pretrained(self, pretrained_path):
        config = ImplicitModelConfig.from_pretrained(pretrained_path)
        model = ImplicitModel(config)
        state_dict = torch.load(os.path.join(pretrained_path, "state_dict.bin"))
        model.load_state_dict(state_dict, strict=True)
        return model

    def save_pretrained(self, save_directory):
        print(f"Saving to {save_directory}")
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "state_dict.bin"))
