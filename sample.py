#!/usr/bin/env python

import sys
import transformers
import torch
import numpy as np
import plotly.graph_objects as go
import os
from tqdm import tqdm
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "true"

DEFAULT_CUDA_DEVICE = 1
DEFAULT_IMG_DIR = 'img'
DEFAULT_IMG_FORMAT = 'png'
DEFAULT_EX_PATH = 'examples.txt'

if torch.cuda.is_available():
    device = torch.device(f'cuda:{DEFAULT_CUDA_DEVICE}')
else:
    device = torch.device('cpu')

lm = None
tokenizer = None

def load_model(model_name='distilgpt2'):
    print(f"Loading models+tokenizer [{model_name}] ...")
    
    global lm, tokenizer
    
    lm = transformers.AutoModelForCausalLM.from_pretrained('distilgpt2')
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilgpt2')

    # Move the model to the correct device
    lm.to(device)

    print("Done.")
    print()


# Default data for experiments
examples = []
examples_tok = []


# Utilities
def load_examples():
    print("Loading examples:")
    
    global examples, examples_tok
    
    with open(DEFAULT_EX_PATH, 'r') as f:
        print("\tCalculating num examples ...")
        examples = [line.strip() for line in tqdm(f)]
        total = len(examples)
        print("\tTokenizing examples ...")
        examples_tok = [
            tokenizer(example, return_tensors='pt')['input_ids'] \
            for example in tqdm(examples)
        ]
    print("Done.")
    print()

def dict_from_output(output):
    logits = output.logits[0, -1, :].detach().cpu().numpy(),
    softmax = torch.nn.functional.softmax(
        output.logits,
        dim=2
    )[0, -1, :].detach().cpu().numpy()
    sorted_idx = np.argsort(softmax)[::-1]
    # hidden_states is a tuple; -1 indexes the last layer
    # from the last layer's hidden states, access that
    #    of the only batch, and the last sequence idx
    hidden = output.hidden_states[-1][0, -1, :].detach().cpu().numpy()

    return {
        'logits': logits[0],
        'softmax': softmax,
        'sorted_idx': sorted_idx,
        'hidden': hidden
    }

def tokenize(prompt):
    if type(prompt) == str:
        prompt_tokens = tokenizer(prompt, return_tensors='pt')
        prompt_tokens = prompt_tokens.to(device)['input_ids']
    elif type(prompt) == torch.Tensor:
        prompt_tokens = prompt
    else:
        raise Exception(f"tokenize got unexpected input type {type(prompt)}")
    
    return prompt_tokens.to(device)
            
def next(prompt):
    prompt_tokens = tokenize(prompt)
        
    output = lm(prompt_tokens, output_hidden_states=True)
    
    return dict_from_output(output)

def top_k(softmax, sorted_idx, logits=None, hidden=None, k=10):
    return [tokenizer.decode(sorted_idx[i]) for i in range(k)]

def graph(
        softmax,
        sorted_idx,
        logits=None,
        hidden=None,
        k=None,
        log=True,
        bar=False,
        true_idx=None,
        out_path=None,
        caption=''
):
    x = np.array(list(range(0, len(softmax))))
    probs = softmax[sorted_idx]
    # If we've been supplied the index of the ground-truth
    # next word, check that we can display it (i.e.,
    # that it falls in the top-k indices we're going
    # to show -- or else, that we haven't been given
    # a top-k value)
    if true_idx:
        rank = int(np.where(sorted_idx==true_idx)[0])
        # print(f"Ground truth next word is {rank}-th; k={k}")
        if not k or rank < k:
            true_idx = rank
        else:
            true_idx = None

        # print(f"\tso setting {true_idx=}")

    if k:
        x = x[:k]
        probs = probs[:k]

    fig = go.Figure()

    if bar and k:
        fig.add_trace(
            go.Bar(
                x=[tokenizer.decode(sorted_idx[i]) for i in range(k)],
                y=probs,
                name=f'Next Token Distribution (top {k})'
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=probs,
                mode='lines',
                name=(
                    f'Next Token Distribution (top {k})' if k \
                    else 'Next Token Distribution'
                )
            )
        )
        if true_idx:
            # print(f"Adding mark at {true_idx}")
            fig.add_trace(
                go.Scatter(
                    x=[true_idx],
                    y=[probs[true_idx]],
                    mode='markers',
                    marker_symbol='circle',
                    marker_color='red',
                    marker_size=12,
                    name=f"Ground truth next token ({true_idx}-th)"
                )
            )

    if log and not bar:
        fig.update_xaxes(type="log")

    fig.update_layout(
        title=caption if caption else 'Probability mass per token in vocab',
        xaxis_title='Token idx by probability mass',
        yaxis_title='Probability mass'
    )

    if not out_path:
        fig.show()
    else:
        fig.write_image(out_path)
        


# TODO:
# Make this use batches instead of single prompt
def sample(lm, tokenizer, prompt, top_p=0.9, repetition_penalty=0.5, max_length=500):
    prompt_tokens = tokenizer(prompt, return_tensors='pt')['input_ids']
    output_tensor = lm.generate(
        prompt_tokens,
        top_p=top_p,
        max_length=max_length,
        repetition_penalty=repetition_penalty
    )
    output_text = tokenizer.batch_decode(output_tensor)
   
    return output_text
   

if __name__ == '__main__':
    if len(sys.argv) == 2:
        model_name = sys.argv[1]
    else:
        model_name = 'distilgpt2'

    # First load the model
    load_model(model_name)
    # Then load the example texts
    load_examples()

    print("Processing example texts ...")
    for ex_num, ex in enumerate(examples_tok):
        num_tok = len(ex[0])
        print()
        print(f"[{ex_num} : {num_tok} toks]: '{examples[ex_num]}'")
        print()

        out_dir_base = Path(DEFAULT_IMG_DIR) / Path(f'ex_{ex_num}')
        
        for curr_idx in tqdm(range(1, num_tok), total=num_tok):
            prompt_tok = ex[:,:curr_idx]
            prompt_text = tokenizer.batch_decode(prompt_tok)
            true_idx = int(ex[:,curr_idx])
            true_text = tokenizer.batch_decode([true_idx])
            
            out_dir_log = out_dir_base / Path(f'log')
            out_dir_linear = out_dir_base / Path(f'linear')
            out_dir_log.mkdir(parents=True, exist_ok=True)
            out_dir_linear.mkdir(parents=True, exist_ok=True)

            ex_name = Path(f'word_{curr_idx}.{DEFAULT_IMG_FORMAT}')
            out_path_log = out_dir_log / ex_name
            out_path_linear = out_dir_linear / ex_name

            # Now call graph, but edit to allow caption
            # caption, or x-axis title, should have prompt in NL
            # make sure to include true_idx
            graph(
                **next(prompt_tok),
                caption=f'[{prompt_text}] : [{true_text}] (log-scale)',
                true_idx=true_idx,
                k=100,
                log=True,
                out_path=out_path_log
            )

            graph(
                **next(prompt_tok),
                caption=f'[{prompt_text}] : [{true_text}] (linear-scale)',
                true_idx=true_idx,
                k=100,
                log=False,
                out_path=out_path_linear
            )

            # Now write code to accumulate num times true_idx was in top_k
            # top_k should be tried at both 100 and 1000 -- maybe have list
            # of top_ks, and run through them each
            continue
            
        
    exit()
