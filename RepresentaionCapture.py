# Usage: 
# with RepresentationCapture(model.layers) as token_list:
#     output = model(x)
# analyze(token_list)

class RepresentationCapture:
    def __init__(self, layers):
        self.layers = layers
        self.captured = []
        self.handles = []

    def _hook(self, module, input, output):
        # We store a detached copy to avoid keeping the graph alive
        self.captured.append(output.detach())

    def __enter__(self):
        # Clear previous captures if reused
        self.captured.clear()
        # Register hooks for each layer
        for layer in self.layers:
            handle = layer.register_forward_hook(self._hook)
            self.handles.append(handle)
        return self.captured

    def __exit__(self, exc_type, exc_value, traceback):
        # This part runs even if an error occurs inside the 'with' block
        for handle in self.handles:
            handle.remove()
        self.handles.clear()