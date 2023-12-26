class Pipeline:
    def __init__(self, filters, predictor):
        self.filters = filters
        self.predictor = predictor

    def __call__(self, signal):
        for filter in self.filters:
            signal = filter(signal)
        return self.predictor(signal)
