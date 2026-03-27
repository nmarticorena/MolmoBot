import abc
from typing import Union, List


class AbstractActionSpace(abc.ABC):
    @abc.abstractmethod
    def _get_singleton_action(
        self, action_string: str = "", action_quantity: float = None
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def get_action(
        self,
        action_strings: Union[str, List[str]],
        action_quantities: Union[float, List[float]] = None,
    ):
        raise NotImplementedError
