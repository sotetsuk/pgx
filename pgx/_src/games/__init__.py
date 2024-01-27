from typing import Optional, Protocol, TypeVar, runtime_checkable

from jax import Array

# NOTE: These protocols are WIP and subject to change.

T = TypeVar("T")  # GameState


@runtime_checkable
class GameProtocol(Protocol[T]):
    def init(self) -> T: ...

    def step(self, state: T, action: Array) -> T: ...

    def observe(self, state: T, color: Optional[Array] = None) -> Array: ...

    def legal_action_mask(self, state: T) -> Array: ...

    def is_terminal(self, state: T) -> Array: ...

    def rewards(self, state: T) -> Array: ...
