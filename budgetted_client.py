import torch
from typing import Any, Dict
from transformers import pipeline, AutoTokenizer


class ThinkingBudgetClient:
    def __init__(self, model_name_or_path: str, tokenizer_name_or_path: str, thinking_mode: str = "on"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.thinking_mode = thinking_mode

        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        self.pipeline = pipeline(
            "text-generation",
            model=model_name_or_path,
            tokenizer=self.tokenizer,
            temperature=0.6,
            top_p=0.95,
            **model_kwargs
        )

        self.model_name_or_path = model_name_or_path

    def chat_completion(
        self,
        prompt: str,
        max_thinking_budget: int = 512,
        max_tokens: int = 1024,
        **kwargs,
    ) -> Dict[str, Any]:
        assert (
            max_tokens > max_thinking_budget
        ), f"thinking budget must be smaller than maximum new tokens. Given {max_tokens=} and {max_thinking_budget=}"

        # Step 1: Reasoning
        if "Nemotron" in self.model_name_or_path:
            reasoning_prompt = [
                {"role": "system", "content": f"detailed thinking on"},
                {"role": "user", "content": prompt}
            ]
        else:
            reasoning_prompt = [
                {"role": "user", "content": prompt}
            ]

        reasoning_response = self.pipeline(
            reasoning_prompt,
            max_length=max_thinking_budget,
            **kwargs,
        )[0]["generated_text"]

        if self.thinking_mode == "on" and not reasoning_response.endswith("</think>"):
            reasoning_response = f"{reasoning_response.strip()}.\n</think>"

        reasoning_tokens_len = len(
            self.tokenizer.encode(reasoning_response, add_special_tokens=False)
        )

        remaining_tokens = max_tokens - reasoning_tokens_len
        
        assert (
            remaining_tokens > 0
        ), f"remaining_tokens must be positive. Given {remaining_tokens=}. Increase the max_tokens or lower the max_thinking_budget."

        answer_prompt = f"{prompt}\nReasoning:\n{reasoning_response}\nAnswer:"

        if "Nemotron" in self.model_name_or_path:
            reasoning_prompt = [
                {"role": "system", "content": f"detailed thinking off"},
                {"role": "user", "content": answer_prompt}
            ]
        else:
            reasoning_prompt = [
                {"role": "user", "content": answer_prompt}
            ]

        answer_response = self.pipeline(
            answer_prompt,
            max_length=remaining_tokens,
            **kwargs,
        )[0]["generated_text"]

        answer_tokens_len = len(
            self.tokenizer.encode(answer_response, add_special_tokens=False)
        )

        remaining_tokens -= answer_tokens_len
        
        assert (
            remaining_tokens > 0
        ), f"remaining_tokens must be positive. Given {remaining_tokens=}. Increase the max_tokens or adjust the reasoning/answer budget."

        # Step 3: Post-hoc explanation
        explanation_prompt = f"{prompt}\nReasoning:\n{reasoning_response}\nAnswer:\n{answer_response}\nWhy did you give this answer?\nExplanation:"

        if "Nemotron" in self.model_name_or_path:
            reasoning_prompt = [
                {"role": "system", "content": f"detailed thinking off"},
                {"role": "user", "content": explanation_prompt}
            ]
        else:
            reasoning_prompt = [
                {"role": "user", "content": explanation_prompt}
            ]

        explanation_response = self.pipeline(
            explanation_prompt,
            max_length=remaining_tokens,
            **kwargs,
        )[0]["generated_text"]

        return {
            "reasoning": reasoning_response.strip().strip("</think>").strip(),
            "answer": answer_response.strip(),
            "post_hoc_explanation": explanation_response.strip(),
        }
