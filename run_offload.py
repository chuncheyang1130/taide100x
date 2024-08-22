import torch
import torch.nn as nn
# from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, 
    LlamaForCausalLM, 
    LlamaRMSNorm, 
    LlamaRotaryEmbedding
)
import argparse
import time

from models.wrapper import NaiveWrapper, OffloadWrapper
from models.hooks import offload_to_cpu

from accelerate import (
    cpu_offload, 
    init_empty_weights, 
    load_checkpoint_and_dispatch, 
    infer_auto_device_map
)

def main(args):

    print("Loading model...")

    device = torch.device("cuda:0")

    # load LLM

    if args.mode == "naive":
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_path,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
    elif args.mode == "offload":
        config = AutoConfig.from_pretrained(args.llm_path)
        with init_empty_weights():
            llm = AutoModelForCausalLM.from_config(config)
        checkpoint = "./llama_model"
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, use_fast=False)

    if args.mode == "naive":
        model = NaiveWrapper()
        model.set_llm(llm)
    elif args.mode == "offload":
        model = OffloadWrapper()
        model.set_llm(llm, checkpoint)
        
    model.set_tokenizer(tokenizer)
    model.eval()


    print("Warming up model...")

    # input message
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    input_message = "Hello."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_message},
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
    _  = model.generate(input_ids, temperature=args.temp, max_length=args.max_new_token, do_sample=args.do_sample)

    # generate response
    print("Generating response...")

    # input message
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    input_message = "What's the best way to start learning a new language?"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_message},
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
    prompt = tokenizer.decode(input_ids[0])
    
    repetitions = 5
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]

    for i in range(repetitions):
        start_events[i].record()
        output_ids = model.generate(input_ids, temperature=args.temp, max_length=args.max_new_token, do_sample=args.do_sample)
        end_events[i].record()

    torch.cuda.synchronize()
    
    output = model.tokenizer.decode(output_ids[0][input_ids.shape[1]:])

    if not args.no_print_message:
        print("\nPrompt:")
        print(prompt)
        print("\nModel response:")
        print(output)
        print("\n-----------------------------------")
        print("Input tokens:", len(input_ids[0]))
        print("Output tokens:", len(output_ids[0][input_ids.shape[1]:]))
    
    # if not args.no_print_time:
    #     print("Time:", end_time - start_time)

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    latency = np.median(times) # median is more robust to outliers
    print(f"Latency: {latency}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.6,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--llm-path",
        "-llm",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="LLM model path.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Whether to do sampling. (Default is False)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="offload",
        help="The mode of model generation",
    )
    parser.add_argument(
        "-nm",
        "--no-print-message",
        action="store_true",
        help="Print the message.",
    )
    parser.add_argument(
        "-nt",
        "--no-print-time",
        action="store_true",
        help="Record the time.",
    )
    args = parser.parse_args()
    
    main(args)