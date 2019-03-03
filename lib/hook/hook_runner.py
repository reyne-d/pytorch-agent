import logging
logger = logging.getLogger("__main__")


class _Hook(object):
    def __init__(self):
        self._id = id(self)

    def begin(self, data_collector):
        # send request args
        pass

    def before_run(self, data_collector):
        pass

    def after_run(self, data_collector):
        pass

    def end(self, data_collector):
        pass

    @property
    def id(self):
        return self._id


class _HookRunner(object):
    """ A class to manage hooks, it will be called in agent.run"""
    def __init__(self, hooks):
        for h in hooks:
            assert isinstance(h, _Hook)
        self._hooks = hooks

    def call_hooks_begin(self, data_collector):
        for hook in self._hooks:
            hook.begin(data_collector)

    def call_hooks_before_run(self, data_collector):
        for hook in self._hooks:
            hook.before_run(data_collector)

    def call_hooks_after_run(self, data_collector):
        for hook in self._hooks:
            hook.after_run(data_collector)

    def call_hooks_end(self, data_collector):
        hook_output = {}
        for hook in self._hooks:
            output = hook.end(data_collector)
            if output is not None:
                hook_output[hook.id] = output
        return hook_output


class HookRunner(_HookRunner):
    """
        A interface of _HookRunner.
    """
    def __init__(self, hooks):
        super(HookRunner, self).__init__(hooks)



