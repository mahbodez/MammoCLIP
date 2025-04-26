class Dictable:
    """
    Base class for objects that can be converted to and from a dictionary.
    """

    def to_dict(self):
        """
        Convert the object to a dictionary, recursively unwinding child Dictables.
        """

        def unwind(value):
            if isinstance(value, Dictable):
                return value.to_dict()
            elif isinstance(value, list):
                return [unwind(v) for v in value]
            elif isinstance(value, tuple):
                return tuple(unwind(v) for v in value)
            elif isinstance(value, dict):
                return {k: unwind(v) for k, v in value.items()}
            else:
                return value

        attrs = {
            k: unwind(v)
            for k, v in self.__dict__.items()
            if isinstance(
                v, (int, float, str, bool, list, dict, tuple, type(None), Dictable)
            )
            and not k.startswith("_")
        }
        d = {"class_": self.name, "attrs_": attrs}
        return d

    @classmethod
    def from_dict(cls, config: dict):
        """
        Create an object from a dictionary.
        """
        return cls(**config)

    @property
    def name(self):
        """
        Return the name of the class.
        """
        raise NotImplementedError("name property must be implemented in the subclass")

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
