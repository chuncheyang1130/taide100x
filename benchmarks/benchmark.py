import torch

@torch.no_grad() # Time per output token
def benchmark_tpot(model, past_key_values, tokens, repetitions=100):
    # Get the number of previous tokens
    prev_tokens = past_key_values.get_seq_length()

    # Warmup steps
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(repetitions):
            _ = model.run(tokens, past_key_values=past_key_values)
            past_key_values.crop(prev_tokens)
    torch.cuda.current_stream().wait_stream(s)
    
    # Capture CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _ = model.run(tokens, past_key_values=past_key_values)
    # actually not required to crop past_key_values, since cudagraph will replay and read and write to the same memory locations
    # past_key_values.crop(prev_tokens)
    
    # Start and end events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    
    # Run the benchmark
    for i in range(repetitions):
        flush_cache()
        start_events[i].record()
        graph.replay() 
        # _ = model.run(tokens, past_key_values=past_key_values)
        end_events[i].record()
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    latency = np.median(times) # median is more robust to outliers
    
    return latency