import argparse
import pickle

import torch

import transformers

from tqdm import tqdm
import json

def main(args):

    model_id = args.model_id

    model_kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(args.data, "rb") as f:
        data = pickle.load(f)

    data = data[args.starting_index:100]

    for item in data:
        item["legal_moves"] = [str(x) for x in item["legal_moves"]]

    prompt = "Given the following chess board position: {board}, choose the best next move from the following options: {legal_moves}. " \
    "Respond with the best move in algebraic notation from the list of possible legal moves" \
    "Please reason step by step, and put your final answer within \\boxed{{}}"

    if "nvidia" in model_id:
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
        model=model_id,
        tokenizer=tokenizer,
        max_new_tokens=32768,
        batch_size=args.batch_size,
        **model_kwargs
    )

    for i in tqdm(range(0, len(messages), args.batch_size), desc="Generating text"):
        batch = messages[i:i + args.batch_size]
        
        outputs = pipeline(batch)

        for j, out in enumerate(outputs):
            with open(f"outputs/initial_test_outputs_{model_id.replace('/', '_')}_{args.starting_index+i+j}.json", "w") as f:
                f.write(json.dumps(out))
            

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--starting_index', type=int, default=0)
    parser.add_argument('--data')
    args = parser.parse_args()

    main(args)
