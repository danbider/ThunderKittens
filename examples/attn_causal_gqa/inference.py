import fire
import os
import sys
import time
import gradio as gr

import torch
from transformers import AutoTokenizer, AutoConfig

from llama_recipes.inference.safety_utils import get_safety_checker, AgentType
from llama_recipes.inference.model_utils import load_peft_model

from accelerate.utils import is_xpu_available

from custom_llama_model import CustomLlamaForCausalLM

def load_custom_llama_model(model_name, quantization, use_fast_kernels):
    config = AutoConfig.from_pretrained(model_name)
    model = CustomLlamaForCausalLM.from_pretrained(model_name, config=config)
    
    if quantization:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
    if use_fast_kernels:
        # Additional setup if needed for fast kernels
        pass
    
    model = model.to("cuda:0")
    
    return model

def main(
    model_name,
    peft_model: str = None,
    quantization: bool = False,
    max_new_tokens = 20,
    prompt_file: str = None,
    seed: int = 42,
    do_sample: bool = True,
    min_length: int = None,
    use_cache: bool = True,
    top_p: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    length_penalty: int = 1,
    enable_azure_content_safety: bool = False,
    enable_sensitive_topics: bool = False,
    enable_salesforce_content_safety: bool = True,
    enable_llamaguard_content_safety: bool = False,
    max_padding_length: int = None,
    use_fast_kernels: bool = False,
    **kwargs
):
    def inference(user_prompt, temperature, top_p, top_k, max_new_tokens, **kwargs):
        safety_checker = get_safety_checker(
            enable_azure_content_safety,
            enable_sensitive_topics,
            enable_salesforce_content_safety,
            enable_llamaguard_content_safety
        )

        safety_results = [check(user_prompt) for check in safety_checker]
        are_safe = all([r[1] for r in safety_results])
        if are_safe:
            print("User prompt deemed safe.")
            print(f"User prompt:\n{user_prompt}")
        else:
            print("User prompt deemed unsafe.")
            for method, is_safe, report in safety_results:
                if not is_safe:
                    print(method)
                    print(report)
            print("Skipping the inference as the prompt is not safe.")
            sys.exit(1)  # Exit the program with an error status

        if is_xpu_available():
            torch.xpu.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        model = load_custom_llama_model(model_name, quantization, use_fast_kernels)
        if peft_model:
            model = load_peft_model(model, peft_model)

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
        if is_xpu_available():
            batch = {k: v.to("xpu") for k, v in batch.items()}
        else:
            batch = {k: v.to("cuda") for k, v in batch.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )
        e2e_inference_time = (time.perf_counter() - start) * 1000
        print(f"The inference time is {e2e_inference_time} ms")
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        safety_results = [check(output_text, agent_type=AgentType.AGENT, user_prompt=user_prompt) for check in safety_checker]
        are_safe = all([r[1] for r in safety_results])
        if are_safe:
            print("User input and model output deemed safe.")
            print(f"Model output:\n{output_text}")
        else:
            print("Model output deemed unsafe.")
            for method, is_safe, report in safety_results:
                if not is_safe:
                    print(method)
                    print(report)
        return output_text

    if prompt_file is not None:
        assert os.path.exists(prompt_file), f"Provided Prompt file does not exist {prompt_file}"
        with open(prompt_file, "r") as f:
            user_prompt = "\n".join(f.readlines())
        inference(user_prompt, temperature, top_p, top_k, max_new_tokens)
    elif not sys.stdin.isatty():
        user_prompt = "\n".join(sys.stdin.readlines())
        inference(user_prompt, temperature, top_p, top_k, max_new_tokens)
    else:
        gr.Interface(
            fn=inference,
            inputs=[
                gr.components.Textbox(lines=9, label="User Prompt", placeholder="none"),
                gr.components.Slider(minimum=0, maximum=1, value=1.0, label="Temperature"),
                gr.components.Slider(minimum=0, maximum=1, value=1.0, label="Top p"),
                gr.components.Slider(minimum=0, maximum=100, step=1, value=50, label="Top k"),
                gr.components.Slider(minimum=1, maximum=2000, step=1, value=200, label="Max tokens"),
            ],
            outputs=[
                gr.components.Textbox(lines=5, label="Output"),
            ],
            title="Meta Llama3 Playground",
            description="https://github.com/facebookresearch/llama-recipes",
        ).queue().launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    fire.Fire(main)
