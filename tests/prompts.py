"""Prompt set shared by the correctness-anchor tests: name -> (prompt, max_new_tokens).

A prompt is either plain text or a chat message list (rendered with the chat
template). The set covers short and long prompts, code, non-Latin text and a
long generation.
"""

from __future__ import annotations

_LONG_TEXT = (
    "The history of computing is often told as a sequence of machines, but it is equally a history of ideas "
    "about representation: how numbers, text, images and eventually programs themselves can be encoded as "
    "patterns of symbols and manipulated by rules. "
) * 12

PROMPTS = {
    "short_en": ("The capital of France is", 128),
    "long_en": (_LONG_TEXT + "\nSummarize the passage above in three sentences:", 128),
    "code": ("def quicksort(arr):\n    \"\"\"Sort a list of integers.\"\"\"\n", 128),
    "zh": ("请用三句话介绍一下长城的历史。", 128),
    "chat": ([{"role": "user", "content": "Explain what a KV cache is in one paragraph."}], 128),
    "long_gen": ("Write a short story about a lighthouse keeper who finds a message in a bottle.", 512),
}


def encode(tokenizer, prompt) -> list[int]:
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False
        )
    return tokenizer(prompt).input_ids
