"""Model loading and text generation helpers."""
from functools import lru_cache

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig


@lru_cache(maxsize=None)
def load_model_and_tokenizer(model_name: str):
    """Load and cache a seq2seq model + tokenizer by Hugging Face model name."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Few-shot prompts are built as [exemplar 1][exemplar 2][exemplar 3][target
    # dialogue], with the dialogue to summarize LAST. Tokenizers truncate from
    # the right by default, which would silently cut off the target dialogue
    # itself whenever the prompt runs long. Truncate from the left instead, so
    # if anything gets dropped it's the earliest exemplar, not the target.
    tokenizer.truncation_side = "left"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


def summarize(
    prompt: str,
    model_name: str,
    max_new_tokens: int = 60,
    max_input_tokens: int = 1024,
    **gen_kwargs,
) -> str:
    """Generate a summary for `prompt` using the given model.

    `max_input_tokens` bounds the encoder input length explicitly (FLAN-T5's
    relative position embeddings tolerate longer-than-512 inputs reasonably
    well). Combined with left-side truncation, this keeps the target dialogue
    intact even for 3-shot prompts. Extra keyword arguments (e.g.
    do_sample=True, temperature=0.5) are passed straight through to
    GenerationConfig.
    """
    model, tokenizer = load_model_and_tokenizer(model_name)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens
    )
    config = GenerationConfig(max_new_tokens=max_new_tokens, **gen_kwargs)
    output_ids = model.generate(inputs["input_ids"], generation_config=config)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
