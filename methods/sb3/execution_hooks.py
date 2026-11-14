"""Read-only scheduling hints; importing this module requires no ML backend."""


def resource_hints(cfg, *, paths):
    model = cfg.get("model") or {}
    native = model.get("sb3_native_policy", False) if hasattr(model, "get") else False
    # The adapter fixes Theta model policies to CPU. Native policies may use a GPU.
    return {"accelerator": bool(native)}
