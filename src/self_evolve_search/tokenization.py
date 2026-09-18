"""Explicit token-ID output across Transformers chat-template versions."""

def chat_ids(tokenizer,messages):
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    if not isinstance(ids,list) or not ids or any(type(token) is not int for token in ids):
        raise TypeError('A single conversation must produce a nonempty list of integer token IDs')
    return ids
