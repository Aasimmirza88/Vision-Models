import torch
import time
import os


def profile_model(model, input_size, device, save_dir):
    results = {}

    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results["total_parameters"] = total_params
    results["trainable_parameters"] = trainable_params
    results["model_size_mb"] = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - start) / 50
    results["inference_time_per_batch_ms"] = elapsed * 1000
    results["throughput_images_per_sec"] = 1 / elapsed

    try:
        from torchsummary import summary
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        summary(model, input_size=(3, input_size, input_size))
        summary_str = sys.stdout.getvalue()
        sys.stdout = old_stdout
        results["model_summary"] = summary_str
    except Exception:
        results["model_summary"] = "torchsummary not available"

    try:
        from thop import profile
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        results["flops_G"] = flops / 1e9
        results["params_M"] = params / 1e6
    except ImportError:
        results["flops_G"] = None
        results["params_M"] = None

    os.makedirs(save_dir, exist_ok=True)
    import json
    with open(os.path.join(save_dir, "profiling.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        return {"allocated_GB": allocated, "reserved_GB": reserved}
    return {"allocated_GB": 0, "reserved_GB": 0}


def check_oom_risk(model, batch_size, input_size):
    try:
        dummy = torch.randn(batch_size, 3, input_size, input_size)
        model(dummy)
        return True, "No OOM risk detected"
    except RuntimeError as e:
        if "out of memory" in str(e):
            return False, f"OOM risk: {str(e)}"
        raise