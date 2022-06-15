from ..base import HPOBase


@HPOBase.register("naive")
class NaiveHPO(HPOBase):
    @property
    def is_sequential(self) -> bool:
        return False


__all__ = ["NaiveHPO"]
