from prompt_toolkit.completion.base import Completer, Completion
from prompt_toolkit.shortcuts.prompt import CompleteStyle
from prompt_toolkit import prompt
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel, BertForMaskedLM
from transformers import pipeline
import re
import time
import transformers
import math
import copy
import json
import os

def basicPreprocess(text):
    processed_text = text.lower()
    processed_text = re.sub(r"-\n", "", processed_text) # when a word is line broken
    processed_text = re.sub(r"\n", ' ', processed_text)
    processed_text = re.sub(r" +", " ", processed_text)
    return processed_text

import logging
logging.basicConfig(level=logging.INFO)

# fill_mask = pipeline("fill-mask", model="allenai/scibert_scivocab_uncased", tokenizer="allenai/scibert_scivocab_uncased", topk=100)

fill_mask_newriter = pipeline("fill-mask", model="./newriter2021_2", tokenizer="./newriter2021_2")

modelo2 = "book"
fill_mask2 = pipeline("fill-mask", model=modelo2, tokenizer=modelo2)

def score(sentence, pipe):
    tokenize_input = pipe.tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([pipe.tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = pipe.model(tensor_input, labels=tensor_input)
    return math.exp(loss[0])

class GPTCompleter(Completer):
    def __init__(self):
        self.loading = 0
    
    def predict(self, text, word):
        while word[-1] not in {'.','?','!','~','\n',','} and len(word) < 20:
            try:
                #myText = text+word+"[MASK]"

                results = fill_mask_newriter(text.lower() + word + " [MASK]", topk=1)

                token = fill_mask_newriter.tokenizer.convert_ids_to_tokens(list(results)[0]["token"])

                prefix = ' '
                
                while token.startswith('#') and len(token) > 0:
                    prefix = ''
                    replacada = token.replace('#', "")
                    token = replacada
                
                # Get the predicted next sub-word
                word += prefix + token
            except:
                pass
        
        suggestions = []

        final_texts = []
        final_texts.append(word + " (" + str(score(word, fill_mask_newriter)) + ") [OG]")
        
        word_separated = word.split(" ")

        for j, palavra in enumerate(word_separated):
            my_array = copy.copy(word_separated)
            my_array[j] = "[MASK]"
            word_joined = " ".join(my_array)

            results = fill_mask2(text.lower() + word_joined)

            for i,res in enumerate(list(results)):
                token = fill_mask2.tokenizer.convert_ids_to_tokens(res["token"])
                suggestions.append([
                    token,
                    score(" ".join(my_array[0:j]) + " " + token + " " + " ".join(my_array[j+1:]), fill_mask_newriter),
                    " ".join(my_array[0:j]) + " _" + token + "_ " + " ".join(my_array[j+1:])
                ])
        
        def compare(it):
            return it[1]

        suggestions = sorted(suggestions, key=compare)[0:5]
        
        for s in suggestions:
            final_texts.append(s[2] + " (" + str(s[1]) + ")")

        return final_texts

    def get_completions(self, document, complete_event):
        # Keep count of how many completion generators are running.
        self.loading += 1
        text = document.text.strip()

        try:
            results = fill_mask_newriter(text + " [MASK]")

            words = [fill_mask_newriter.tokenizer.convert_ids_to_tokens(a["token"]) for a in list(results)]

            for word in words:
                for p in self.predict(text, word):
                    yield Completion(p, 0)

        finally:
            # We use try/finally because this generator can be closed if the
            # input text changes before all completions are generated.
            self.loading -= 1


slow_completer = GPTCompleter()

# Add a bottom toolbar that display when completions are loading.
def bottom_toolbar():
    return " Loading completions... " if slow_completer.loading > 0 else ""

# Display prompt.
text = prompt(
    "Type: ",
    completer=slow_completer,
    complete_in_thread=True,
    complete_while_typing=True,
    bottom_toolbar=bottom_toolbar,
    complete_style=CompleteStyle.MULTI_COLUMN,
)
print("You said: %s" % text)