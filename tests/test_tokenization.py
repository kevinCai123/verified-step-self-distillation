import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from self_evolve_search.agent import Client
from self_evolve_search.tokenization import chat_ids

def test_transformers_chat_result_is_integer_ids_and_real_prompt_length():
    backend=Tokenizer(WordLevel({'[UNK]':0,'hello':1,'world':2},unk_token='[UNK]'))
    backend.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',chat_template="{% for message in messages %}{{ message['content'] }} {% endfor %}")
    messages=[{'role':'user','content':'hello world hello world hello'}]
    ids=chat_ids(tokenizer,messages)
    assert ids==[1,2,1,2,1]
    assert torch.tensor([ids],dtype=torch.long).shape==(1,5)
    assert Client('http://127.0.0.1:8093','test',tokenizer).prompt_tokens(messages)==5
