import argparse
from pathlib import Path

import pickle

import re

import torch

import transformers

from tqdm import tqdm
import json

def clean_explanation(explanation):
    """
    Cleans up the explanation by removing the final answer enclosed in \\boxed{} 
    and other concluding phrases like 'best answer:', 'final answer:', or 'conclusion:'.
    """
    # Remove the content within \boxed{}
    explanation = re.sub(r'\\boxed\{.*?\}', '', explanation)
    
    # Remove concluding phrases
    explanation = re.sub(r'(best answer:|final answer:|conclusion:).*$', '', explanation, flags=re.IGNORECASE)
    
    return explanation.strip()

def main(args):

    if args.model_pair == "nv":
        small_model_id = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
        large_model_file_id = "nvidia_Llama-3.1-Nemotron-Nano-8B-v1"
    elif args.model_pair == "r1":
        small_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        large_model_file_id = "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B"

    model_kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    tokenizer = transformers.AutoTokenizer.from_pretrained(small_model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    large_model_explanations = [x for x in list(Path(args.output_dir).iterdir()) if large_model_file_id in str(x)]
    
    large_model_explanations = [json.load(open(x, "r")) for x in large_model_explanations]

    large_model_explanations = [
        clean_explanation(entry[0]["generated_text"][-1]["content"])
        for entry in large_model_explanations
    ]

    large_model_explanations = [clean_explanation(x) for x in large_model_explanations]

    with open(args.data, "rb") as f:
        data = pickle.load(f)

    data = data[args.starting_index:100]

    for i, item in enumerate(data):
        item["explanation"] = large_model_explanations[i+args.starting_index]

    for item in data:
        item["legal_moves"] = [str(x) for x in item["legal_moves"]]

    prompt = "Given the following chess board position: {board} and an explanation of a possible move {explanation} choose the best next move from the following options: {legal_moves}. " \
    "Respond with the best move in algebraic notation from the list of possible legal moves" \
    "Please reason step by step, and put your final answer within \\boxed{{}}"

    if args.model_pair == "nv":
        messages = [
            [
            {"role": "system", "content": "detailed thinking on"},
            {"role": "user", "content": prompt.format(**x)},
            {"role": "assistant", "content": "<think>\n</think>"}
            ]
            for x in data
        ]

    else:
        messages = [
            [{"role": "user","content": prompt.format(**x)}]
            for x in data
        ]

    pipeline = transformers.pipeline(
        "text-generation",
        model=small_model_id,
        tokenizer=tokenizer,
        max_new_tokens=32768,
        batch_size=args.batch_size,
        **model_kwargs
    )

    for i in tqdm(range(0, len(messages), args.batch_size), desc="Generating text"):
        batch = messages[i:i + args.batch_size]
        
        outputs = pipeline(batch)

        for j, out in enumerate(outputs):
            with open(f"outputs/initial_test_outputs_{small_model_id.replace('/', '_')}_paired_with_{large_model_file_id}_{args.starting_index+i+j}.json", "w") as f:
                f.write(json.dumps(out))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pair')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--starting_index', type=int, default=0)
    parser.add_argument('--output_dir')
    parser.add_argument('--data')

    args = parser.parse_args()

    main(args)
