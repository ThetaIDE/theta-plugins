"""Install a synchronous, removable SB3 post-train-call boundary for mechanistic monitors."""
from __future__ import annotations


class PostUpdateHook:
    def __init__(self, algorithm, manager):
        self.algorithm, self.manager = algorithm, manager
        self.had_override = "train" in algorithm.__dict__
        self.previous_override = algorithm.__dict__.get("train")
        self.original = algorithm.train
        self.closed = False
        self.updates = max((entry["last"] or 0 for entry in getattr(manager, "entries", [])), default=0)
        optimizer = getattr(algorithm.policy, "optimizer", None)
        if optimizer is None or not hasattr(optimizer, "register_step_post_hook"):
            raise ValueError("Post-update diagnostics require an optimizer with step post hooks")
        def stepped(*_):
            self.updates += 1
        self.optimizer_handle = optimizer.register_step_post_hook(stepped)

        def train(*args, **kwargs):
            previous = self.updates
            result = self.original(*args, **kwargs)
            current = self.updates
            if current > previous:
                manager.collect(int(algorithm.num_timesteps), current)
            return result

        self.wrapper = train
        algorithm.train = train

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.optimizer_handle.remove()
        if self.algorithm.__dict__.get("train") is self.wrapper:
            if self.had_override:
                self.algorithm.train = self.previous_override
            else:
                del self.algorithm.train


def monitored_learn(algorithm, manager, **kwargs):
    hook = PostUpdateHook(algorithm, manager) if manager.entries else None
    try:
        result = algorithm.learn(**kwargs)
        if hook is not None:
            manager.collect(int(algorithm.num_timesteps), hook.updates, final=True)
        return result
    finally:
        if hook is not None:
            hook.close()
        manager.close()
