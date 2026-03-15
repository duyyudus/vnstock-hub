# Lazy initialization to avoid circular import deadlock
_initialized = False

def _ensure_initialized():
    """Ensure vnstock environment is initialized (called on first use)."""
    global _initialized
    if _initialized:
        return

    _initialized = True
