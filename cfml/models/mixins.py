class StrMixin:
    @property
    def name(self):
        return getattr(self, "_name", "")

    @property
    def kwargs(self):
        return getattr(self, "_kwargs", {})

    def __str__(self):
        if not self.kwargs:
            kwarg_str = ""
        else:
            kwarg_str = "\n".join([" " * 2 + f"{k} : {v}" for k, v in self.kwargs.items()])
            kwarg_str = f"\n{{\n{kwarg_str}\n}}"
        return f"{type(self).__name__}({self.name}){kwarg_str}"

    __repr__ = __str__


__all__ = ["StrMixin"]
