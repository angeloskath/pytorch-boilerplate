from ..factory import ObjectFactory, EmptyFactory, FactoryList, CallableFactory


class Callback:
    def on_prepare_experiment(self, experiment):
        pass

    def on_train_start(self, experiment):
        pass

    def on_train_stop(self, experiment):
        pass

    def on_epoch_start(self, experiment):
        pass

    def on_epoch_stop(self, experiment):
        pass

    def on_train_batch_start(self, experiment):
        pass

    def on_train_batch_stop(self, experiment):
        pass

    def on_validation_start(self, experiment):
        pass

    def on_validation_stop(self, experiment):
        pass

    def on_val_batch_start(self, experiment):
        pass

    def on_val_batch_stop(self, experiment):
        pass


class CallbackList(Callback):
    def __init__(self, *callbacks):
        self.callbacks = callbacks

    def on_prepare_experiment(self, experiment):
        for c in self.callbacks:
            c.on_prepare_experiment(experiment)

    def on_train_start(self, experiment):
        for c in self.callbacks:
            c.on_train_start(experiment)

    def on_train_stop(self, experiment):
        for c in self.callbacks:
            c.on_train_stop(experiment)

    def on_epoch_start(self, experiment):
        for c in self.callbacks:
            c.on_epoch_start(experiment)

    def on_epoch_stop(self, experiment):
        for c in self.callbacks:
            c.on_epoch_stop(experiment)

    def on_train_batch_start(self, experiment):
        for c in self.callbacks:
            c.on_train_batch_start(experiment)

    def on_train_batch_stop(self, experiment):
        for c in self.callbacks:
            c.on_train_batch_stop(experiment)

    def on_validation_start(self, experiment):
        for c in self.callbacks:
            c.on_validation_start(experiment)

    def on_validation_stop(self, experiment):
        for c in self.callbacks:
            c.on_validation_stop(experiment)

    def on_val_batch_start(self, experiment):
        for c in self.callbacks:
            c.on_val_batch_start(experiment)

    def on_val_batch_stop(self, experiment):
        for c in self.callbacks:
            c.on_val_batch_stop(experiment)


class CallbackListFactory(FactoryList):
    def __init__(self, *callbacks):
        factories = []
        for c in callbacks:
            if isinstance(c, ObjectFactory):
                factories.append(c)
            elif isinstance(c, Callback):
                factories.append(EmptyFactory(c))
            elif callable(c):
                factories.append(CallableFactory(c))
            else:
                raise ValueError(("CallbackListFactory expects either "
                                  "factories, callbacks or callables."))
        super().__init__(factories)

    def from_dict(self, arguments):
        return CallbackList(*super().from_dict(arguments))

