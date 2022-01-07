class Bunch(dict):
    def __init__(self, dictionary=dict(), **kwds):
        super().__init__(**kwds)
        self.__dict__ = self
        for key, val in dictionary.items():
            self.__dict__[key] = val
        for key, val in self.items():
            if isinstance(val, dict):
                self.__dict__[key] = Bunch(val)
            elif isinstance(val, list):
                for i in range(len(val)):
                    if isinstance(val[i], dict):
                        val[i] = Bunch(val[i])
