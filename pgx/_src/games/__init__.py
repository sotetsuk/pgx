from jax import Array
from typing import Protocol, Optional, TypeVar, runtime_checkable


# NOTE: These protocols are WIP and subject to change.

T = TypeVar('T')  # GameState


@runtime_checkable
class TwoPlayerPerfectInfoGame(Protocol[T]):

    def init(self) -> T:
        ...

    def step(self, state: T, action: Array) -> T:
        ...

    def observe(self, state: T, color: Optional[Array] = None) -> Array:
        ...

    def legal_action_mask(self, state: T) -> Array:
        ...

    def is_terminal(self, state: T) -> Array:
        ...

    def returns(self, state: T) -> Array:
        ...
