# Source: https://github.com/huggingface/accelerate/blob/v0.33.0/src/accelerate/hooks.py

import functools
from typing import Dict, List, Mapping, Optional, Union

import torch
import torch.nn as nn

import gc
import copy

from accelerate.state import PartialState
from accelerate.utils import (
    PrefixedDataset,
    find_device,
    named_module_tensors,
    send_to_device,
    set_module_tensor_to_device,
)
from accelerate.utils.modeling import get_non_persistent_buffers
from accelerate.utils.other import recursive_getattr

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding


_accelerate_added_attributes = ["to", "cuda", "npu", "xpu", "mlu", "musa"]


class ModelHook:
    """
    A hook that contains callbacks to be executed just before and after the forward method of a model. The difference
    with PyTorch existing hooks is that they get passed along the kwargs.

    Class attribute:
    - **no_grad** (`bool`, *optional*, defaults to `False`) -- Whether or not to execute the actual forward pass under
      the `torch.no_grad()` context manager.
    """

    no_grad = False

    def init_hook(self, module):
        """
        To be executed when the hook is attached to the module.

        Args:
            module (`torch.nn.Module`): The module attached to this hook.
        """
        return module

    def pre_forward(self, module, *args, **kwargs):
        """
        To be executed just before the forward method of the model.

        Args:
            module (`torch.nn.Module`): The module whose forward pass will be executed just after this event.
            args (`Tuple[Any]`): The positional arguments passed to the module.
            kwargs (`Dict[Str, Any]`): The keyword arguments passed to the module.

        Returns:
            `Tuple[Tuple[Any], Dict[Str, Any]]`: A tuple with the treated `args` and `kwargs`.
        """
        return args, kwargs

    def post_forward(self, module, output):
        """
        To be executed just after the forward method of the model.

        Args:
            module (`torch.nn.Module`): The module whose forward pass been executed just before this event.
            output (`Any`): The output of the module.

        Returns:
            `Any`: The processed `output`.
        """
        return output

    def detach_hook(self, module):
        """
        To be executed when the hook is detached from a module.

        Args:
            module (`torch.nn.Module`): The module detached from this hook.
        """
        return module


class SequentialHook(ModelHook):
    """
    A hook that can contain several hooks and iterates through them at each event.
    """

    def __init__(self, *hooks):
        self.hooks = hooks

    def init_hook(self, module):
        for hook in self.hooks:
            module = hook.init_hook(module)
        return module

    def pre_forward(self, module, *args, **kwargs):
        for hook in self.hooks:
            args, kwargs = hook.pre_forward(module, *args, **kwargs)
        return args, kwargs

    def post_forward(self, module, output):
        for hook in self.hooks:
            output = hook.post_forward(module, output)
        return output

    def detach_hook(self, module):
        for hook in self.hooks:
            module = hook.detach_hook(module)
        return module


def add_hook_to_module(module: nn.Module, hook: ModelHook, append: bool = False):
    """
    Adds a hook to a given module. This will rewrite the `forward` method of the module to include the hook, to remove
    this behavior and restore the original `forward` method, use `remove_hook_from_module`.

    <Tip warning={true}>

    If the module already contains a hook, this will replace it with the new hook passed by default. To chain two hooks
    together, pass `append=True`, so it chains the current and new hook into an instance of the `SequentialHook` class.

    </Tip>

    Args:
        module (`torch.nn.Module`):
            The module to attach a hook to.
        hook (`ModelHook`):
            The hook to attach.
        append (`bool`, *optional*, defaults to `False`):
            Whether the hook should be chained with an existing one (if module already contains a hook) or not.

    Returns:
        `torch.nn.Module`: The same module, with the hook attached (the module is modified in place, so the result can
        be discarded).
    """

    if append and (getattr(module, "_hf_hook", None) is not None):
        old_hook = module._hf_hook
        remove_hook_from_module(module)
        hook = SequentialHook(old_hook, hook)

    if hasattr(module, "_hf_hook") and hasattr(module, "_old_forward"):
        # If we already put some hook on this module, we replace it with the new one.
        old_forward = module._old_forward
    else:
        old_forward = module.forward
        module._old_forward = old_forward

    module = hook.init_hook(module)
    module._hf_hook = hook

    def new_forward(module, *args, **kwargs):
        args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
        if module._hf_hook.no_grad:
            with torch.no_grad():
                output = module._old_forward(*args, **kwargs)
        else:
            output = module._old_forward(*args, **kwargs)
        return module._hf_hook.post_forward(module, output)

    # Overriding a GraphModuleImpl forward freezes the forward call and later modifications on the graph will fail.
    # Reference: https://pytorch.slack.com/archives/C3PDTEV8E/p1705929610405409
    if "GraphModuleImpl" in str(type(module)):
        module.__class__.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)
    else:
        module.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)

    return module


def remove_hook_from_module(module: nn.Module, recurse=False):
    """
    Removes any hook attached to a module via `add_hook_to_module`.

    Args:
        module (`torch.nn.Module`): The module to attach a hook to.
        recurse (`bool`, **optional**): Whether to remove the hooks recursively

    Returns:
        `torch.nn.Module`: The same module, with the hook detached (the module is modified in place, so the result can
        be discarded).
    """

    if hasattr(module, "_hf_hook"):
        module._hf_hook.detach_hook(module)
        delattr(module, "_hf_hook")

    if hasattr(module, "_old_forward"):
        # Overriding a GraphModuleImpl forward freezes the forward call and later modifications on the graph will fail.
        # Reference: https://pytorch.slack.com/archives/C3PDTEV8E/p1705929610405409
        if "GraphModuleImpl" in str(type(module)):
            module.__class__.forward = module._old_forward
        else:
            module.forward = module._old_forward
        delattr(module, "_old_forward")

    # Remove accelerate added warning hooks from dispatch_model
    for attr in _accelerate_added_attributes:
        module.__dict__.pop(attr, None)

    if recurse:
        for child in module.children():
            remove_hook_from_module(child, recurse)

    return module

class BlockwiseHook(ModelHook):

    """
    User offload hook

    Args:
        execution_device: 
            Where the module should be put during inferencing
        memory_limit:
            Maximum gpu memory usage (GB)
    """

    def __init__(
        self, 
        module_name: str,
        execution_device: Optional[Union[int, str, torch.device]] = "cuda:0",
        memory_limit: float = 0.0
    ):
        self.execution_device = execution_device
        self.memory_limit = memory_limit
        self.no_grad = True
        self.module_cache = None
        self.module_name = module_name
    
    def init_hook(self, module):
        return module.to("cpu")

    def pre_forward(self, module, *args, **kwargs):

        if self.memory_limit > 0:
            gpu_mem_used = torch.cuda.memory_reserved(self.execution_device)
            gpu_mem_used = gpu_mem_used / (1024 * 1024 * 1024)
            print(f"[Debug] GPU memory usage: {gpu_mem_used} GB")

            size_in_bytes = 0
            
            for _ , param in module.named_parameters():
                size_in_bytes += param.element_size() * param.numel()

            for _ , param in module.named_buffers():
                size_in_bytes += param.element_size() * param.numel()

            size_in_GB = size_in_bytes / (1024 * 1024 * 1024)
            
            # TODO: try to clean cuda cache here, not done
            if gpu_mem_used + size_in_GB > self.memory_limit:
                print(f"[Offload Warning] Possible memory usage: {gpu_mem_used+size_in_GB}")
                torch.cuda.empty_cache()
        
        #TODO: deep copy module
        module_copy = copy.deepcopy(module)

        return module_copy.to(self.execution_device), send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device)

    def post_forward(self, module, output):
        # print("Post Forwarding...")
        del module

        #TODO: check if this operation can be used in pre_foward
        torch.cuda.empty_cache()

        return output

def add_hook_to_block(
    module: nn.Module,
    hook: ModelHook
):
    
    old_forward = module.forward
    module._old_forward = old_forward

    module = hook.init_hook(module)
    module._hook = hook

    def new_forward(module, *args, **kwargs):
        module_gpu, args, kwargs = module._hook.pre_forward(module, *args, **kwargs)
        # gpu_mem_used = torch.cuda.memory_reserved("cuda:0")
        # gpu_mem_used = gpu_mem_used / (1024 * 1024 * 1024)
        # print(f"[Debug] Current gpu usage: {gpu_mem_used}")

        if module_gpu._hook.no_grad:
            with torch.no_grad():
                output = module_gpu._old_forward(*args, **kwargs)
        else:
            output = module_gpu._old_forward(*args, **kwargs)

        # print(f"Check output device: {output.device}")
        # output = module._hook.post_forward(module_gpu)
        del module_gpu, args, kwargs
        torch.cuda.empty_cache()

        return output

    module.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)

    return module

def attach_blockwise_hook(
    module: nn.Module,
    hook_module_type: tuple,
    module_name: str = ""
):

    """
    Recursively attach hook to module
    """

    if isinstance(module, (LlamaDecoderLayer, nn.Embedding, nn.Linear, LlamaRMSNorm, LlamaRotaryEmbedding)):
        print(f"Hook add on {module_name}")
        add_hook_to_block(module, BlockwiseHook(module_name, torch.device("cuda:0"), 8.0))
        return

    for child_name, child in module.named_children():
        child_name = f"{module_name}.{child_name}" if len(module_name) > 0 else child_name

        attach_blockwise_hook(
            child,
            hook_module_type,
            child_name
        )

def offload_to_cpu(
    module: nn.Module,
    hook_module_type: tuple
):

    attach_blockwise_hook(module, hook_module_type)
    return module