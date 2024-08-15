import torch
import torch.nn as nn
# from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm, LlamaRotaryEmbedding
import argparse
import time

from models.model import NaiveWrapper
from models.hooks import offload_to_cpu
from models.big_modeling import cpu_offload

def main(args):

    print("Loading model...")

    # load LLM
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, use_fast=False)
    
    model = NaiveWrapper()
    model.set_tokenizer(tokenizer)
    model.set_llm(llm)

    model.eval()

    print("Adding hook to model...")

    #TODO: build a type list
    hook_module_list = (LlamaDecoderLayer, nn.Embedding, nn.Linear, LlamaRMSNorm, LlamaRotaryEmbedding)
    # model = offload_to_cpu(model, hook_module_list)
    model = cpu_offload(
        model,
        execution_device=torch.device("cuda:0"),
        offload_buffers=True
    )

    print("Warming up model...")

    # input message
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    input_message = "Hello."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_message},
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
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
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
    prompt = tokenizer.decode(input_ids[0])
    
    start_time = time.time()
    output_ids = model.generate(input_ids, temperature=args.temp, max_length=args.max_new_token, do_sample=args.do_sample)
    end_time = time.time()
    
    output = model.tokenizer.decode(output_ids[0][input_ids.shape[1]:])

    if not args.no_print_message:
        print("\nPrompt:")
        print(prompt)
        print("\nModel response:")
        print(output)
        print("\n-----------------------------------")
        print("Input tokens:", len(input_ids[0]))
        print("Output tokens:", len(output_ids[0][input_ids.shape[1]:]))
    
    if not args.no_print_time:
        print("Time:", end_time - start_time)

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