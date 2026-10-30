"""Expected experiment stops, distinct from software exceptions."""


class ExperimentGateError(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__(details["message"])
