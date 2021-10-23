class Test():
    def __init__(self):
        self._foo = dict()

    @property
    def _foo(self):
        print("Reading")

    @_foo.setter
    def _foo(self, test):
        print("Setting ", test)

test = Test()
test._foo