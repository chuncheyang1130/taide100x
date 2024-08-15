# Source: https://github.com/huggingface/accelerate/blob/v0.33.0/src/accelerate/hooks.py

import functools
from typing import Dict, List, Mapping, Optional, Union

import torch
import torch.nn as nn

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

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm, LlamaRotaryEmbedding


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

        gpu_mem_used = torch.cuda.memory_reserved(self.execution_device)
        gpu_mem_used = gpu_mem_used / (1024 * 1024 * 1024)
        print(f"[Debug] Current GPU memory usage: {gpu_mem_used} GB")

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


class AlignDevicesHook(ModelHook):
    """
    A generic `ModelHook` that ensures inputs and model weights are on the same device for the forward pass of the
    associated module, potentially offloading the weights after the forward pass.

    Args:
        execution_device (`torch.device`, *optional*):
            The device on which inputs and model weights should be placed before the forward pass.
        offload (`bool`, *optional*, defaults to `False`):
            Whether or not the weights should be offloaded after the forward pass.
        io_same_device (`bool`, *optional*, defaults to `False`):
            Whether or not the output should be placed on the same device as the input was.
        weights_map (`Mapping[str, torch.Tensor]`, *optional*):
            When the model weights are offloaded, a (potentially lazy) map from param names to the tensor values.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to include the associated module's buffers when offloading.
        place_submodules (`bool`, *optional*, defaults to `False`):
            Whether to place the submodules on `execution_device` during the `init_hook` event.
    """

    def __init__(
        self,
        execution_device: Optional[Union[int, str, torch.device]] = None,
        offload: bool = False,
        io_same_device: bool = False,
        weights_map: Optional[Mapping] = None,
        offload_buffers: bool = False,
        place_submodules: bool = False,
        skip_keys: Optional[Union[str, List[str]]] = None,
        tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]] = None,
    ):
        self.execution_device = execution_device
        self.offload = offload
        self.io_same_device = io_same_device
        self.weights_map = weights_map
        self.offload_buffers = offload_buffers
        self.place_submodules = place_submodules
        self.skip_keys = skip_keys

        # Will contain the input device when `io_same_device=True`.
        self.input_device = None
        self.param_original_devices = {}
        self.buffer_original_devices = {}
        self.tied_params_names = set()

        # The hook pre_forward/post_forward need to have knowledge of this dictionary, as with offloading we want to avoid duplicating memory
        # for tied weights already loaded on the target execution device.
        self.tied_params_map = tied_params_map

    def __repr__(self):
        return (
            f"AlignDevicesHook(execution_device={self.execution_device}, offload={self.offload}, "
            f"io_same_device={self.io_same_device}, offload_buffers={self.offload_buffers}, "
            f"place_submodules={self.place_submodules}, skip_keys={repr(self.skip_keys)})"
        )

    def init_hook(self, module):
        # In case the AlignDevicesHook is on meta device, ignore tied weights as data_ptr() is then always zero.
        if self.execution_device == "meta" or self.execution_device == torch.device("meta"):
            self.tied_params_map = None

        if not self.offload and self.execution_device is not None:
            for name, _ in named_module_tensors(module, recurse=self.place_submodules):
                set_module_tensor_to_device(module, name, self.execution_device, tied_params_map=self.tied_params_map)
        elif self.offload:
            self.original_devices = {
                name: param.device for name, param in named_module_tensors(module, recurse=self.place_submodules)
            }
            if self.weights_map is None:
                self.weights_map = {
                    name: param.to("cpu")
                    for name, param in named_module_tensors(
                        module, include_buffers=self.offload_buffers, recurse=self.place_submodules
                    )
                }
            for name, _ in named_module_tensors(
                module, include_buffers=self.offload_buffers, recurse=self.place_submodules, remove_non_persistent=True
            ):
                # When using disk offloading, we can not rely on `weights_map[name].data_ptr()` as the reference pointer,
                # as we have no guarantee that safetensors' `file.get_tensor()` will always give the same pointer.
                # As we have no reliable way to track the shared data pointer of tied weights in this case, we use tied_params_names: List[str]
                # to add on the fly pointers to `tied_params_map` in the pre_forward call.
                if (
                    self.tied_params_map is not None
                    and recursive_getattr(module, name).data_ptr() in self.tied_params_map
                ):
                    self.tied_params_names.add(name)

                set_module_tensor_to_device(module, name, "meta")

            if not self.offload_buffers and self.execution_device is not None:
                for name, _ in module.named_buffers(recurse=self.place_submodules):
                    set_module_tensor_to_device(
                        module, name, self.execution_device, tied_params_map=self.tied_params_map
                    )
            elif self.offload_buffers and self.execution_device is not None:
                for name in get_non_persistent_buffers(module, recurse=self.place_submodules):
                    set_module_tensor_to_device(
                        module, name, self.execution_device, tied_params_map=self.tied_params_map
                    )

        return module

    def pre_forward(self, module, *args, **kwargs):

        gpu_mem_used = torch.cuda.memory_reserved(self.execution_device)
        gpu_mem_used = gpu_mem_used / (1024 * 1024 * 1024)       
        print(f"[Debug] Before module be sent to GPU, the memory usage is {gpu_mem_used} GB")

        size_in_bytes = 0
        
        for _ , param in module.named_parameters():
            size_in_bytes += param.element_size() * param.numel()

        for _ , param in module.named_buffers():
            size_in_bytes += param.element_size() * param.numel()

        size_in_GB = size_in_bytes / (1024 * 1024 * 1024)
        print(f"[Debug] Expected gpu memory usage after module sent to GPU is {size_in_GB} GB")


        if self.io_same_device:
            self.input_device = find_device([args, kwargs])
        if self.offload:
            self.tied_pointers_to_remove = set()

            for name, _ in named_module_tensors(
                module,
                include_buffers=self.offload_buffers,
                recurse=self.place_submodules,
                remove_non_persistent=True,
            ):
                fp16_statistics = None
                value = self.weights_map[name]
                if "weight" in name and name.replace("weight", "SCB") in self.weights_map.keys():
                    if value.dtype == torch.int8:
                        fp16_statistics = self.weights_map[name.replace("weight", "SCB")]

                # In case we are using offloading with tied weights, we need to keep track of the offloaded weights
                # that are loaded on device at this point, as we will need to remove them as well from the dictionary
                # self.tied_params_map in order to allow to free memory.
                if name in self.tied_params_names and value.data_ptr() not in self.tied_params_map:
                    self.tied_params_map[value.data_ptr()] = {}

                if (
                    value is not None
                    and self.tied_params_map is not None
                    and value.data_ptr() in self.tied_params_map
                    and self.execution_device not in self.tied_params_map[value.data_ptr()]
                ):
                    self.tied_pointers_to_remove.add((value.data_ptr(), self.execution_device))

                set_module_tensor_to_device(
                    module,
                    name,
                    self.execution_device,
                    value=value,
                    fp16_statistics=fp16_statistics,
                    tied_params_map=self.tied_params_map,
                )

        return send_to_device(args, self.execution_device), send_to_device(
            kwargs, self.execution_device, skip_keys=self.skip_keys
        )

    def post_forward(self, module, output):

        gpu_mem_used = torch.cuda.memory_reserved(self.execution_device)
        gpu_mem_used = gpu_mem_used / (1024 * 1024 * 1024)       
        print(f"[Debug] After module be sent to CPU, the memory usage is {gpu_mem_used} GB")

        if self.offload:
            for name, _ in named_module_tensors(
                module,
                include_buffers=self.offload_buffers,
                recurse=self.place_submodules,
                remove_non_persistent=True,
            ):
                set_module_tensor_to_device(module, name, "meta")
                if type(module).__name__ == "Linear8bitLt":
                    module.state.SCB = None
                    module.state.CxB = None

            # We may have loaded tied weights into self.tied_params_map (avoiding to load them several times in e.g. submodules): remove them from
            # this dictionary to allow the garbage collector to do its job.
            for value_pointer, device in self.tied_pointers_to_remove:
                del self.tied_params_map[value_pointer][device]
            self.tied_pointers_to_remove = set()

        if self.io_same_device and self.input_device is not None:
            output = send_to_device(output, self.input_device, skip_keys=self.skip_keys)

        

        return output

    def detach_hook(self, module):
        if self.offload:
            for name, device in self.original_devices.items():
                if device != torch.device("meta"):
                    set_module_tensor_to_device(module, name, device, value=self.weights_map.get(name, None))
        return module


def attach_execution_device_hook(
    module: torch.nn.Module,
    execution_device: Union[int, str, torch.device],
    skip_keys: Optional[Union[str, List[str]]] = None,
    preload_module_classes: Optional[List[str]] = None,
    tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]] = None,
):
    """
    Recursively attaches `AlignDevicesHook` to all submodules of a given model to make sure they have the right
    execution device

    Args:
        module (`torch.nn.Module`):
            The module where we want to attach the hooks.
        execution_device (`int`, `str` or `torch.device`):
            The device on which inputs and model weights should be placed before the forward pass.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        tied_params_map (Optional[Dict[int, Dict[torch.device, torch.Tensor]]], *optional*, defaults to `None`):
            A map of data pointers to dictionaries of devices to already dispatched tied weights. For a given execution
            device, this parameter is useful to reuse the first available pointer of a shared weight for all others,
            instead of duplicating memory.
    """
    if not hasattr(module, "_hf_hook") and len(module.state_dict()) > 0:
        add_hook_to_module(
            module,
            AlignDevicesHook(execution_device, skip_keys=skip_keys, tied_params_map=tied_params_map),
        )

    # Break the recursion if we get to a preload module.
    if preload_module_classes is not None and module.__class__.__name__ in preload_module_classes:
        return

    for child in module.children():
        attach_execution_device_hook(child, execution_device, tied_params_map=tied_params_map)


def attach_align_device_hook(
    module: torch.nn.Module,
    execution_device: Optional[torch.device] = None,
    offload: bool = False,
    weights_map: Optional[Mapping] = None,
    offload_buffers: bool = False,
    module_name: str = "",
    skip_keys: Optional[Union[str, List[str]]] = None,
    preload_module_classes: Optional[List[str]] = None,
    tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]] = None,
):
    """
    Recursively attaches `AlignDevicesHook` to all submodules of a given model that have direct parameters and/or
    buffers.

    Args:
        module (`torch.nn.Module`):
            The module where we want to attach the hooks.
        execution_device (`torch.device`, *optional*):
            The device on which inputs and model weights should be placed before the forward pass.
        offload (`bool`, *optional*, defaults to `False`):
            Whether or not the weights should be offloaded after the forward pass.
        weights_map (`Mapping[str, torch.Tensor]`, *optional*):
            When the model weights are offloaded, a (potentially lazy) map from param names to the tensor values.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to include the associated module's buffers when offloading.
        module_name (`str`, *optional*, defaults to `""`):
            The name of the module.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        tied_params_map (Optional[Dict[int, Dict[torch.device, torch.Tensor]]], *optional*, defaults to `None`):
            A map of data pointers to dictionaries of devices to already dispatched tied weights. For a given execution
            device, this parameter is useful to reuse the first available pointer of a shared weight for all others,
            instead of duplicating memory.
    """
    # Attach the hook on this module if it has any direct tensor.

    directs = named_module_tensors(module)
    full_offload = (
        offload and preload_module_classes is not None and module.__class__.__name__ in preload_module_classes
    )

    if len(list(directs)) > 0 or full_offload or isinstance(module, (LlamaDecoderLayer, nn.Embedding, nn.Linear, LlamaRMSNorm, LlamaRotaryEmbedding)):
        if weights_map is not None:
            prefix = f"{module_name}." if len(module_name) > 0 else ""
            prefixed_weights_map = PrefixedDataset(weights_map, prefix)
        else:
            prefixed_weights_map = None
        hook = AlignDevicesHook(
            execution_device=execution_device,
            offload=offload,
            weights_map=prefixed_weights_map,
            offload_buffers=offload_buffers,
            place_submodules=full_offload,
            skip_keys=skip_keys,
            tied_params_map=tied_params_map,
        )

        print(f"[Info] Adding hook on {module_name}")
        add_hook_to_module(module, hook, append=True)

    # We stop the recursion in case we hit the full offload.
    if full_offload:
        return

    # Recurse on all children of the module.
    for child_name, child in module.named_children():
        child_name = f"{module_name}.{child_name}" if len(module_name) > 0 else child_name
    
        attach_align_device_hook(
            child,
            execution_device=execution_device,
            offload=offload,
            weights_map=weights_map,
            offload_buffers=offload_buffers,
            module_name=child_name,
            preload_module_classes=preload_module_classes,
            skip_keys=skip_keys,
            tied_params_map=tied_params_map,
        )


def remove_hook_from_submodules(module: nn.Module):
    """
    Recursively removes all hooks attached on the submodules of a given model.

    Args:
        module (`torch.nn.Module`): The module on which to remove all hooks.
    """
    remove_hook_from_module(module)
    for child in module.children():
        remove_hook_from_submodules(child)


def attach_align_device_hook_on_blocks(
    module: nn.Module,
    execution_device: Optional[Union[torch.device, Dict[str, torch.device]]] = None,
    offload: Union[bool, Dict[str, bool]] = False,
    weights_map: Mapping = None,
    offload_buffers: bool = False,
    module_name: str = "",
    skip_keys: Optional[Union[str, List[str]]] = None,
    preload_module_classes: Optional[List[str]] = None,
    tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]] = None,
):
    """
    Attaches `AlignDevicesHook` to all blocks of a given model as needed.

    Args:
        module (`torch.nn.Module`):
            The module where we want to attach the hooks.
        execution_device (`torch.device` or `Dict[str, torch.device]`, *optional*):
            The device on which inputs and model weights should be placed before the forward pass. It can be one device
            for the whole module, or a dictionary mapping module name to device.
        offload (`bool`, *optional*, defaults to `False`):
            Whether or not the weights should be offloaded after the forward pass. It can be one boolean for the whole
            module, or a dictionary mapping module name to boolean.
        weights_map (`Mapping[str, torch.Tensor]`, *optional*):
            When the model weights are offloaded, a (potentially lazy) map from param names to the tensor values.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to include the associated module's buffers when offloading.
        module_name (`str`, *optional*, defaults to `""`):
            The name of the module.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        tied_params_map (Optional[Dict[int, Dict[torch.device, torch.Tensor]]], *optional*, defaults to `None`):
            A map of data pointers to dictionaries of devices to already dispatched tied weights. For a given execution
            device, this parameter is useful to reuse the first available pointer of a shared weight for all others,
            instead of duplicating memory.
    """
    # If one device and one offload, we've got one hook.
    if not isinstance(execution_device, Mapping) and not isinstance(offload, dict):
        if not offload:
            hook = AlignDevicesHook(
                execution_device=execution_device,
                io_same_device=True,
                skip_keys=skip_keys,
                place_submodules=True,
                tied_params_map=tied_params_map,
            )
            add_hook_to_module(module, hook)
        else:
            attach_align_device_hook(
                module,
                execution_device=execution_device,
                offload=True,
                weights_map=weights_map,
                offload_buffers=offload_buffers,
                module_name=module_name,
                skip_keys=skip_keys,
                tied_params_map=tied_params_map,
            )
        return

    if not isinstance(execution_device, Mapping):
        execution_device = {key: execution_device for key in offload.keys()}
    if not isinstance(offload, Mapping):
        offload = {key: offload for key in execution_device.keys()}

    if module_name in execution_device and module_name in offload and not offload[module_name]:
        hook = AlignDevicesHook(
            execution_device=execution_device[module_name],
            offload_buffers=offload_buffers,
            io_same_device=(module_name == ""),
            place_submodules=True,
            skip_keys=skip_keys,
            tied_params_map=tied_params_map,
        )
        add_hook_to_module(module, hook)
        attach_execution_device_hook(module, execution_device[module_name], tied_params_map=tied_params_map)
    elif module_name in execution_device and module_name in offload:
        attach_align_device_hook(
            module,
            execution_device=execution_device[module_name],
            offload=True,
            weights_map=weights_map,
            offload_buffers=offload_buffers,
            module_name=module_name,
            skip_keys=skip_keys,
            preload_module_classes=preload_module_classes,
            tied_params_map=tied_params_map,
        )
        if not hasattr(module, "_hf_hook"):
            hook = AlignDevicesHook(
                execution_device=execution_device[module_name],
                io_same_device=(module_name == ""),
                skip_keys=skip_keys,
                tied_params_map=tied_params_map,
            )
            add_hook_to_module(module, hook)
        attach_execution_device_hook(
            module,
            execution_device[module_name],
            preload_module_classes=preload_module_classes,
            skip_keys=skip_keys,
            tied_params_map=tied_params_map,
        )
    elif module_name == "":
        hook = AlignDevicesHook(
            execution_device=execution_device.get(""),
            io_same_device=True,
            skip_keys=skip_keys,
            tied_params_map=tied_params_map,
        )
        add_hook_to_module(module, hook)

    for child_name, child in module.named_children():
        child_name = f"{module_name}.{child_name}" if len(module_name) > 0 else child_name
        attach_align_device_hook_on_blocks(
            child,
            execution_device=execution_device,
            offload=offload,
            weights_map=weights_map,
            offload_buffers=offload_buffers,
            module_name=child_name,
            preload_module_classes=preload_module_classes,
            skip_keys=skip_keys,
            tied_params_map=tied_params_map,
        )


class CpuOffload(ModelHook):
    """
    Offloads a model on the CPU until its forward pass is called. The model will not be offloaded back to the CPU after
    the forward, the user needs to call the `init_hook` method again for this.

    Args:
        execution_device(`str`, `int` or `torch.device`, *optional*):
            The device on which the model should be executed. Will default to the MPS device if it's available, then
            GPU 0 if there is a GPU, and finally to the CPU.
        prev_module_hook (`UserCpuOffloadHook`, *optional*):
            The hook sent back by [`cpu_offload_with_hook`] for a previous model in the pipeline you are running. If
            passed, its offload method will be called just before the forward of the model to which this hook is
            attached.
    """

    def __init__(
        self,
        execution_device: Optional[Union[str, int, torch.device]] = None,
        prev_module_hook: Optional["UserCpuOffloadHook"] = None,
    ):
        self.prev_module_hook = prev_module_hook

        self.execution_device = execution_device if execution_device is not None else PartialState().default_device

    def init_hook(self, module):
        return module.to("cpu")

    def pre_forward(self, module, *args, **kwargs):
        if self.prev_module_hook is not None:
            self.prev_module_hook.offload()
        module.to(self.execution_device)
        return send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device)


class UserCpuOffloadHook:
    """
    A simple hook grouping a model and a `ModelHook`, which provides easy APIs for to call the init method of the hook
    or remove it entirely.
    """

    def __init__(self, model, hook):
        self.model = model
        self.hook = hook

    def offload(self):
        self.hook.init_hook(self.model)

    def remove(self):
        remove_hook_from_module(self.model)





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

        #TODO: deep copy module
        module_copy = copy.deepcopy(module)
        print(f"[Info] module {self.module_name} is ready to be sent to gpu")

        if self.memory_limit > 0:
            gpu_mem_used = torch.cuda.memory_reserved(self.execution_device)
            gpu_mem_used = gpu_mem_used / (1024 * 1024 * 1024)

            size_in_bytes = 0
            
            for _ , param in module.named_parameters():
                size_in_bytes += param.element_size() * param.numel()

            for _ , param in module.named_buffers():
                size_in_bytes += param.element_size() * param.numel()

            size_in_GB = size_in_bytes / (1024 * 1024 * 1024)

            print(f"[Info] GPU memory usage: {gpu_mem_used} GB")
            print(f"[Info] Module parameter memory usage: {size_in_GB} GB")
            
            # TODO: try to clean cuda cache here, not done
            if gpu_mem_used + size_in_GB > self.memory_limit:
                print(f"[Offload Warning] Possible memory usage: {gpu_mem_used+size_in_GB}")
                torch.cuda.empty_cache()

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
        # torch.cuda.empty_cache()

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