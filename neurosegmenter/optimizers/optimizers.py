from tensorflow import keras
from keras import optimizers as keras_optimizers
from neurosegmenter.plugins import Plugin, PluginType, PluginParameter
from neurosegmenter.plugins import register_plugin
from neurosegmenter.optimizers import Optimizer
from keras.optimizers import Adam

plugins = []

class RMSpropOptimizer(Plugin, Optimizer):
    name: str = "RMSprop Optimizer"
    description: str = "RMSprop Optimizer"
    type: PluginType = PluginType.OPTIMIZER
    parameters: list[PluginParameter] = [
        PluginParameter(
            name="learning_rate",
            description="Learning rate",
            type=float,
            default=0.001,
            path="optimizers.rmsprop_optimizer.learning_rate",
        ),
        PluginParameter(
            name="rho",
            description="Discounting factor for the history/coming gradient.",
            default=0.9,
            type=float,
            path="optimizers.rmsprop_optimizer.rho",
        ),
        PluginParameter(
            name="momentum",
            description="Momentum.",
            default=0.0,
            type=float,
            path="optimizers.rmsprop_optimizer.momentum",
        ),
        PluginParameter(
            name="epsilon",
            description="Small float added to the denominator to improve numerical stability.",
            type=float,
            default=1e-07,
            path="optimizers.rmsprop_optimizer.epsilon",
        ),
        PluginParameter(
            name="centered",
            type=bool,
            description="If True, gradients are normalized by the estimated variance of the gradient; if False, by the uncentered second moment. Setting this to True will usually help convergence.",
            default=False,
            path="optimizers.rmsprop_optimizer.centered",
        ),
        PluginParameter(
            name="weight_decay",
            description="Weight decay regularization.",
            type=float,
            default=None,
            path="optimizers.rmsprop_optimizer.weight_decay",
        ),
        PluginParameter(
            name="clipnorm",
            type=float,
            description="Clip gradients by norm.",
            default=None,
            path="optimizers.rmsprop_optimizer.clipnorm",
        ),
        PluginParameter(
            name="clipvalue",
            description="Clip gradients by value individually.",
            type=float,
            default=None,
            path="optimizers.rmsprop_optimizer.clipvalue"
        ),
        PluginParameter(
            name="global_clipnorm",
            type=float,
            description="Clip gradients by norm globally.",
            default=None,
            path="optimizers.rmsprop_optimizer.global_clipnorm"
        ),
        PluginParameter(
            name="use_ema",
            description="Use exponential moving average.",
            type=bool,
            default=False,
            path="optimizers.rmsprop_optimizer.use_ema"
        ),
        PluginParameter(
            name="ema_momentum",
            description="Exponential moving average momentum.",
            type=float,
            default=0.99,
            path="optimizers.rmsprop_optimizer.ema_momentum"
        ),
        PluginParameter(
            name="ema_overwrite_frequency",
            description="Int or None, defaults to None. Only used if use_ema=True. Every ema_overwrite_frequency steps of iterations, we overwrite the model variable by its moving average. If None, the optimizer # noqa: E501 does not overwrite model variables in the middle of training, and you need to explicitly overwrite the variables at the end of training by calling optimizer.finalize_variable_values() (which updates the model # noqa: E501 variables in-place). When using the built-in fit() training loop, this happens automatically after the last epoch, and you don't need to do anything.",
            type=int,
            default=None,
            path="optimizers.rmsprop_optimizer.ema_overwrite_frequency"
        ),
        PluginParameter(
            name="jit_compile",
            description="Use XLA.",
            type=bool,
            default=True,
            path="optimizers.rmsprop_optimizer.jit_compile"
        )
    ]
    learning_rate: float
    rho: float
    momentum: float
    epsilon: float
    centered: bool
    weight_decay: float
    clipnorm: float
    clipvalue: float
    global_clipnorm: float
    use_ema: bool
    ema_momentum: float
    ema_overwrite_frequency: int
    jit_compile: bool
    
    def get_optimizer(self) -> keras_optimizers.Optimizer:
        optimizer = keras_optimizers.RMSprop(
            learning_rate=self.learning_rate,
            rho=self.rho,
            momentum=self.momentum,
            epsilon=self.epsilon,
            centered=self.centered,
            weight_decay=self.weight_decay,
            clipnorm=self.clipnorm,
            clipvalue=self.clipvalue,
            global_clipnorm=self.global_clipnorm,
            use_ema=self.use_ema,
            ema_momentum=self.ema_momentum,
            ema_overwrite_frequency=self.ema_overwrite_frequency,
            jit_compile=self.jit_compile)
        return optimizer




class AdamOptimizer(Plugin, Optimizer):
    name: str = "Adam Optimizer"
    description: str = "Adam Optimizer"
    type: PluginType = PluginType.OPTIMIZER
    parameters: list[PluginParameter] = [
        PluginParameter(
            name="learning_rate",
            description="Learning rate",
            type=float,
            default=0.001,
            path="optimizers.adam_optimizer.learning_rate",
        ),
        PluginParameter(
            name="beta_1",
            description="Exponential decay rate for the first moment estimates.",
            default=0.9,
            type=float,
            path="optimizers.adam_optimizer.beta_1",
        ),
        PluginParameter(
            name="beta_2",
            description="Exponential decay rate for the second moment estimates.",
            default=0.999,
            type=float,
            path="optimizers.adam_optimizer.beta_2",
        ),
        PluginParameter(
            name="epsilon",
            description="Small float added to the denominator to improve numerical stability.",
            type=float,
            default=1e-07,
            path="optimizers.adam_optimizer.epsilon",
        ),
        PluginParameter(
            name="amsgrad",
            description="Whether to apply the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond.",
            default=False,
            type=bool,
            path="optimizers.adam_optimizer.amsgrad",
        ),
        PluginParameter(
            name="weight_decay",
            description="Weight decay regularization.",
            type=float,
            default=None,
            path="optimizers.adam_optimizer.weight_decay",
        ),
        PluginParameter(
            name="clipnorm",
            type=float,
            description="Clip gradients by norm.",
            default=None,
            path="optimizers.adam_optimizer.clipnorm",
        ),
        PluginParameter(
            name="clipvalue",
            description="Clip gradients by value individually.",
            type=float,
            default=None,
            path="optimizers.adam_optimizer.clipvalue"
        ),
        PluginParameter(
            name="global_clipnorm",
            type=float,
            description="Clip gradients by norm globally.",
            default=None,
            path="optimizers.adam_optimizer.global_clipnorm"
        ),
        PluginParameter(
            name="use_ema",
            description="Use exponential moving average.",
            type=bool,
            default=False,
            path="optimizers.adam_optimizer.use_ema"
        ),
        PluginParameter(
            name="ema_momentum",
            description="Exponential moving average momentum.",
            type=float,
            default=0.99,
            path="optimizers.adam_optimizer.ema_momentum"
        ),
        PluginParameter(
            name="ema_overwrite_frequency",
            description="Int or None, defaults to None. Only used if use_ema=True. Every ema_overwrite_frequency steps of iterations, we overwrite the model variable by its moving average. If None, the optimizer # noqa: E501 does not overwrite model variables in the middle of training, and you need to explicitly overwrite the variables at the end of training by calling optimizer.finalize_variable_values() (which updates the model # noqa: E501 variables in-place). When using the built-in fit() training loop, this happens automatically after the last epoch, and you don't need to do anything.",
            type=int,
            default=None,
            path="optimizers.adam_optimizer.ema_overwrite_frequency"
        ),
        PluginParameter(
            name="jit_compile",
            description="Use XLA.",
            type=bool,
            default=True,
            path="optimizers.adam_optimizer.jit_compile"
        )
    ]
    learning_rate: float
    beta_1: float
    beta_2: float
    epsilon: float
    amsgrad: bool
    weight_decay: float
    clipnorm: float
    clipvalue: float
    global_clipnorm: float
    use_ema: bool
    ema_momentum: float
    ema_overwrite_frequency: int
    jit_compile: bool
    
    def get_optimizer(self) -> keras_optimizers.Optimizer:
        optimizer = Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            amsgrad=self.amsgrad,
            weight_decay=self.weight_decay,
            clipnorm=self.clipnorm,
            clipvalue=self.clipvalue,
            global_clipnorm=self.global_clipnorm,
            use_ema=self.use_ema,
            ema_momentum=self.ema_momentum,
            ema_overwrite_frequency=self.ema_overwrite_frequency,
            jit_compile=self.jit_compile
        )
        return optimizer
    
    
plugins.append(AdamOptimizer)