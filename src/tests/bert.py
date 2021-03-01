#!/usr/bin/env python

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

sample = '''
I'm just gonna cut to the chase - my name is Hermione, and I hate it. I've been getting "ten points for Gryffindor!" jokes from teachers since I was in kindergarten, other kids constantly quoting the books and movies to me. My mom is a huge Harry Potter fan, and while I did enjoy it when I was little, I've always been into more stereotypical "girly" things like makeup, nail art, and baking, etc. My mom loves all the big fantasy/sci-fi franchises, and I'm basically just neutral towards them. When I was 7, I wanted to redecorate my room with Hello Kitty stuff, but my mom got so upset I changed my mind just to make her happy. I didn't redecorate my room until I was 12.
'''

tokens = tokenizer.tokenize(sample)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f' Sentence: {sample}')
print(f'   Tokens: {tokens}')
print(f'Token IDs: {token_ids}')

encoding = tokenizer.encode_plus(
        sample,
        max_length=512,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt' #pytorch
        )

print(encoding['input_ids'])
