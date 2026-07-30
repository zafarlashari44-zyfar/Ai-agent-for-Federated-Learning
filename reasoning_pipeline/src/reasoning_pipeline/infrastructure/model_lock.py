from _thread import RLock
from threading import Lock
from weakref import WeakKeyDictionary

from torch import nn

_MODEL_LOCKS: WeakKeyDictionary[nn.Module, RLock] = WeakKeyDictionary()
_MODEL_LOCKS_GUARD = Lock()


def shared_model_lock(model: nn.Module) -> RLock:
    """Return the single re-entrant lock associated with a model instance."""
    with _MODEL_LOCKS_GUARD:
        lock = _MODEL_LOCKS.get(model)
        if lock is None:
            lock = RLock()
            _MODEL_LOCKS[model] = lock
        return lock
