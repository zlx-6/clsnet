from .hook import HOOKS, Hook
from .lr_updater import annealing_cos, annealing_linear, format_param


class MomentumUpdaterHook(Hook):

    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.9):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_momentum" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio

        self.base_momentum = []  # initial momentum for all param groups
        self.regular_momentum = [
        ]  # expected momentum if no warming up is performed

    def _set_momentum(self, runner, momentum_groups):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group, mom in zip(optim.param_groups,
                                            momentum_groups[k]):
                    if 'momentum' in param_group.keys():
                        param_group['momentum'] = mom
                    elif 'betas' in param_group.keys():
                        param_group['betas'] = (mom, param_group['betas'][1])
        else:
            for param_group, mom in zip(runner.optimizer.param_groups,
                                        momentum_groups):
                if 'momentum' in param_group.keys():
                    param_group['momentum'] = mom
                elif 'betas' in param_group.keys():
                    param_group['betas'] = (mom, param_group['betas'][1])

    def get_momentum(self, runner, base_momentum):
        raise NotImplementedError

    def get_regular_momentum(self, runner):
        if isinstance(runner.optimizer, dict):
            momentum_groups = {}
            for k in runner.optimizer.keys():
                _momentum_group = [
                    self.get_momentum(runner, _base_momentum)
                    for _base_momentum in self.base_momentum[k]
                ]
                momentum_groups.update({k: _momentum_group})
            return momentum_groups
        else:
            return [
                self.get_momentum(runner, _base_momentum)
                for _base_momentum in self.base_momentum
            ]

    def get_warmup_momentum(self, cur_iters):

        def _get_warmup_momentum(cur_iters, regular_momentum):
            if self.warmup == 'constant':
                warmup_momentum = [
                    _momentum / self.warmup_ratio
                    for _momentum in self.regular_momentum
                ]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 -
                                                           self.warmup_ratio)
                warmup_momentum = [
                    _momentum / (1 - k) for _momentum in self.regular_mom
                ]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_momentum = [
                    _momentum / k for _momentum in self.regular_mom
                ]
            return warmup_momentum

        if isinstance(self.regular_momentum, dict):
            momentum_groups = {}
            for key, regular_momentum in self.regular_momentum.items():
                momentum_groups[key] = _get_warmup_momentum(
                    cur_iters, regular_momentum)
            return momentum_groups
        else:
            return _get_warmup_momentum(cur_iters, self.regular_momentum)

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint,
        # if 'initial_momentum' is not saved,
        # it will be set according to the optimizer params
        if isinstance(runner.optimizer, dict):
            self.base_momentum = {}
            for k, optim in runner.optimizer.items():
                for group in optim.param_groups:
                    if 'momentum' in group.keys():
                        group.setdefault('initial_momentum', group['momentum'])
                    else:
                        group.setdefault('initial_momentum', group['betas'][0])
                _base_momentum = [
                    group['initial_momentum'] for group in optim.param_groups
                ]
                self.base_momentum.update({k: _base_momentum})
        else:
            for group in runner.optimizer.param_groups:
                if 'momentum' in group.keys():
                    group.setdefault('initial_momentum', group['momentum'])
                else:
                    group.setdefault('initial_momentum', group['betas'][0])
            self.base_momentum = [
                group['initial_momentum']
                for group in runner.optimizer.param_groups
            ]

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return
        self.regular_mom = self.get_regular_momentum(runner)
        self._set_momentum(runner, self.regular_mom)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.by_epoch:
            self.regular_mom = self.get_regular_momentum(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_momentum(runner, self.regular_mom)
            else:
                warmup_momentum = self.get_warmup_momentum(cur_iter)
                self._set_momentum(runner, warmup_momentum)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_momentum(runner, self.regular_mom)
            else:
                warmup_momentum = self.get_warmup_momentum(cur_iter)
                self._set_momentum(runner, warmup_momentum)


@HOOKS.register_module()
class CosineAnnealingMomentumUpdaterHook(MomentumUpdaterHook):

    def __init__(self, min_momentum=None, min_momentum_ratio=None, **kwargs):
        assert (min_momentum is None) ^ (min_momentum_ratio is None)
        self.min_momentum = min_momentum
        self.min_momentum_ratio = min_momentum_ratio
        super(CosineAnnealingMomentumUpdaterHook, self).__init__(**kwargs)

    def get_momentum(self, runner, base_momentum):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        if self.min_momentum_ratio is not None:
            target_momentum = base_momentum * self.min_momentum_ratio
        else:
            target_momentum = self.min_momentum
        return annealing_cos(base_momentum, target_momentum,
                             progress / max_progress)