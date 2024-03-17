# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import warnings

import paddle
import paddle.optimizer

from .state import AcceleratorState, GradientState
from .utils import honor_type


def move_to_device(state, device):
    if isinstance(state, (list, tuple)):
        return honor_type(state, (move_to_device(t, device) for t in state))
    elif isinstance(state, dict):
        return type(state)({k: move_to_device(v, device) for k, v in state.items()})
    elif isinstance(state, paddle.Tensor):
        return state.to(device)
    return state


class AcceleratedOptimizer(paddle.optimizer.Optimizer):
    """
    Internal wrapper around a torch optimizer.

    Conditionally will perform `step` and `zero_grad` if gradients should be synchronized when performing gradient
    accumulation.

    Args:
        optimizer (`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
        device_placement (`bool`, *optional*, defaults to `True`):
            Whether or not the optimizer should handle device placement. If so, it will place the state dictionary of
            `optimizer` on the right device.
        scaler (`torch.cuda.amp.grad_scaler.GradScaler`, *optional*):
            The scaler to use in the step function if training with mixed precision.
    """

    def __init__(self, optimizer, device_placement=True, scaler=None):
        self.optimizer = optimizer
        self.scaler = scaler
        self.accelerator_state = AcceleratorState()
        self.gradient_state = GradientState()
        device_placement = False
        self.device_placement = device_placement
        self._is_overflow = False

        if self.scaler is not None:
            self._accelerate_step_called = False
            self._optimizer_original_step_method = self.optimizer.step
            self._optimizer_patched_step_method = patch_optimizer_step(self, self.optimizer.step)

        # Handle device placement
        if device_placement:
            state_dict = self.optimizer.state_dict()
            self.optimizer.set_state_dict(state_dict)

    @property
    def state(self):
        return self.optimizer.state

    @state.setter
    def state(self, state):
        self.optimizer.state = state

    @property
    def param_groups(self):
        return self.optimizer._param_groups

    @param_groups.setter
    def param_groups(self, param_groups):
        self.optimizer._param_groups = param_groups

    @property
    def defaults(self):
        return self.optimizer.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optimizer.defaults = defaults

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self.optimizer.set_state_dict(state_dict)

    set_state_dict = load_state_dict

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self, set_to_zero=None):
        if self.gradient_state.sync_gradients:
            accept_arg = "set_to_zero" in inspect.signature(self.optimizer.clear_grad).parameters
            if accept_arg:
                if set_to_zero is None:
                    set_to_zero = True
                self.optimizer.clear_grad(set_to_zero=set_to_zero)
            else:
                if set_to_zero is not None:
                    raise ValueError("`set_to_zero` for Optimizer.clear_grad` is not supported by this optimizer.")
                self.optimizer.clear_grad()

    clear_grad = zero_grad

    def step(self, closure=None):
        if self.gradient_state.sync_gradients:
            if self.scaler is not None:
                self.optimizer.step = self._optimizer_patched_step_method

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if not self._accelerate_step_called:
                    # If the optimizer step was skipped, gradient overflow was detected.
                    self._is_overflow = True
                else:
                    self._is_overflow = False
                # Reset the step method to the original one
                self.optimizer.step = self._optimizer_original_step_method
                # Reset the indicator
                self._accelerate_step_called = False
            else:
                self.optimizer.step()

    def _switch_parameters(self, parameters_map):
        for param_group in self.param_groups:
            param_group["params"] = [parameters_map.get(p, p) for p in param_group["params"]]

    @property
    def is_overflow(self):
        """Whether or not the optimizer step was done, or skipped because of gradient overflow."""
        warnings.warn(
            "The `is_overflow` property is deprecated and will be removed in version 1.0 of Accelerate use "
            "`optimizer.step_was_skipped` instead.",
            FutureWarning,
        )
        return self._is_overflow

    @property
    def step_was_skipped(self):
        """Whether or not the optimizer step was skipped."""
        return self._is_overflow

    def __getstate__(self):
        _ignored_keys = [
            "_accelerate_step_called",
            "_optimizer_original_step_method",
            "_optimizer_patched_step_method",
        ]
        return {k: v for k, v in self.__dict__.items() if k not in _ignored_keys}

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.scaler is not None:
            self._accelerate_step_called = False
            self._optimizer_original_step_method = self.optimizer.step
            self._optimizer_patched_step_method = patch_optimizer_step(self, self.optimizer.step)


def patch_optimizer_step(accelerated_optimizer: AcceleratedOptimizer, method):
    def patched_step(*args, **kwargs):
        accelerated_optimizer._accelerate_step_called = True
        return method(*args, **kwargs)

    return patched_step