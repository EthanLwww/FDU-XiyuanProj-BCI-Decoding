"""Optional independent-band frontends. Import this module before loading FB checkpoints."""
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.constraints import max_norm

from bci.EEGModels_tf import EEGNet, EEGConformer
from bci.filter_bank import DEFAULT_BANDS, FILTER_BANK_MODES, validate_residual_mode


@tf.keras.utils.register_keras_serializable(package="FingerBCI")
class BandSlice(L.Layer):
    """A serializable slice; unlike an ordinary Conv2D it cannot mix bands."""
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)

    def call(self, inputs):
        return inputs[..., self.index:self.index + 1]

    def get_config(self):
        return {**super().get_config(), "index": self.index}


@tf.keras.utils.register_keras_serializable(package="FingerBCI")
class BandFusion(L.Layer):
    """Global softmax weighting or an equal-weight control over aligned features."""
    def __init__(self, mode="weighted", **kwargs):
        super().__init__(**kwargs)
        if mode not in ("weighted", "mean"):
            raise ValueError("BandFusion supports weighted or mean.")
        self.mode = mode

    def build(self, input_shape):
        self.n_bands = len(input_shape)
        if self.n_bands < 1 or any(tuple(s) != tuple(input_shape[0]) for s in input_shape):
            raise ValueError("Fusion requires nonempty, equal-shaped band features.")
        if self.mode == "weighted":
            self.logits = self.add_weight(name="band_logits", shape=(self.n_bands,),
                                          initializer="zeros", trainable=True)
        super().build(input_shape)

    def band_weights(self):
        if self.mode == "weighted":
            return tf.nn.softmax(self.logits)
        return tf.ones((self.n_bands,), dtype=self.compute_dtype) / self.n_bands

    def call(self, inputs):
        stacked = tf.stack(inputs, axis=-1)
        return tf.reduce_sum(stacked * tf.cast(self.band_weights(), stacked.dtype), axis=-1)

    def get_config(self):
        return {**super().get_config(), "mode": self.mode}


def get_band_weights(model):
    """Return global softmax weights (mean: uniform); off/concat have no scalar weights."""
    for layer in model.layers:
        if isinstance(layer, BandFusion):
            return layer.band_weights().numpy().tolist()
    return None


@tf.keras.utils.register_keras_serializable(package="FingerBCI")
class BroadbandResidual(L.Layer):
    """Add an aligned broadband branch with a fixed or sigmoid-gated coefficient."""
    def __init__(self, mode="gated", **kwargs):
        super().__init__(**kwargs)
        if mode not in ("fixed", "gated"):
            raise ValueError("BroadbandResidual supports fixed or gated.")
        self.mode = mode

    def build(self, input_shape):
        if len(input_shape) != 2 or tuple(input_shape[0]) != tuple(input_shape[1]):
            raise ValueError("Residual fusion requires two equal-shaped feature tensors.")
        if self.mode == "gated":
            self.gate_logit = self.add_weight(
                name="gate_logit", shape=(),
                initializer=tf.keras.initializers.Constant(-2.1972245773362196),
                trainable=True)  # logit(0.1)
        super().build(input_shape)

    def coefficient(self):
        if self.mode == "gated":
            return tf.nn.sigmoid(self.gate_logit)
        return tf.constant(1., dtype=self.compute_dtype)

    def call(self, inputs):
        main, broadband = inputs
        return main + tf.cast(self.coefficient(), main.dtype) * broadband

    def get_config(self):
        return {**super().get_config(), "mode": self.mode}


def get_residual_info(model):
    """Report the actual coefficient in this model (not a frequency importance score)."""
    for layer in model.layers:
        if isinstance(layer, BroadbandResidual):
            return dict(mode=layer.mode, alpha=float(layer.coefficient().numpy()),
                        initial_alpha=.1 if layer.mode == "gated" else 1.)
    return None


def _add_broadband_residual(main, input_layer, chans, dropout, architecture, mode):
    if mode == "off":
        return main
    x = BandSlice(len(DEFAULT_BANDS), name="broadband_input")(input_layer)
    if architecture == "EEGNet":
        x = L.Conv2D(8, (1, 32), padding="same", use_bias=False, name="broadband_temporal")(x)
        x = L.BatchNormalization()(x)
        x = L.DepthwiseConv2D((chans, 1), depth_multiplier=2, use_bias=False,
                              depthwise_constraint=max_norm(1.), name="broadband_spatial")(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("elu")(x)
        x = L.AveragePooling2D((1, 4))(x)
        x = L.Dropout(dropout)(x)
        x = L.SeparableConv2D(16, (1, 16), padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("elu")(x)
        x = L.AveragePooling2D((1, 8))(x)
        x = L.Dropout(dropout)(x)
    else:
        # Original Conformer CNN: ordinary spatial Conv2D, not the subband depthwise variant.
        x = L.Conv2D(40, (1, 25), padding="valid", name="broadband_temporal")(x)
        x = L.Conv2D(40, (chans, 1), padding="valid", name="broadband_spatial")(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("elu")(x)
        x = L.AveragePooling2D((1, 75), strides=(1, 15))(x)
        x = L.Dropout(dropout)(x)
        x = L.Conv2D(40, (1, 1), name="broadband_projection")(x)
        x = L.Reshape((-1, 40))(x)
    x = L.Activation("linear", name="broadband_features")(x)
    x = L.Dense(int(main.shape[-1]), use_bias=False, name="broadband_alignment")(x)
    x = L.LayerNormalization(center=False, scale=False, name="broadband_normalized")(x)
    return BroadbandResidual(mode=mode, name="broadband_residual")([main, x])


def _fuse(features, mode, embedding_size=None):
    if mode == "concat":
        x = L.Concatenate(axis=-1, name="band_concat")(features)
        if embedding_size is not None:
            x = L.Dense(embedding_size, name="band_concat_projection")(x)
        return x
    return BandFusion(mode=mode, name="band_fusion")(features)


def _eegnet(input_layer, nb_classes, chans, dropout, mode, broadband_residual):
    features = []
    for band in range(len(DEFAULT_BANDS)):
        prefix = "band_" + str(band)
        x = BandSlice(band, name=prefix + "_input")(input_layer)
        x = L.Conv2D(8, (1, 32), padding="same", use_bias=False,
                     name=prefix + "_temporal")(x)
        x = L.BatchNormalization()(x)
        x = L.DepthwiseConv2D((chans, 1), depth_multiplier=2, use_bias=False,
                              depthwise_constraint=max_norm(1.),
                              name=prefix + "_spatial")(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("elu")(x)
        x = L.AveragePooling2D((1, 4))(x)
        x = L.Dropout(dropout)(x)
        x = L.SeparableConv2D(16, (1, 16), padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("elu")(x)
        x = L.AveragePooling2D((1, 8))(x)
        x = L.Dropout(dropout)(x)
        # Non-trainable affine scale prevents a branch-specific gain immediately
        # before fusion from trivially compensating for its softmax coefficient.
        x = L.LayerNormalization(center=False, scale=False,
                                 name=prefix + "_features")(x)
        features.append(x)
    x = _fuse(features, mode)
    x = _add_broadband_residual(x, input_layer, chans, dropout, "EEGNet", broadband_residual)
    x = L.Flatten(name="flatten")(x)
    x = L.Dense(nb_classes, kernel_constraint=max_norm(.25), name="dense")(x)
    return L.Activation("softmax", name="softmax")(x)


def _conformer(input_layer, nb_classes, chans, dropout, depth, mode, broadband_residual):
    features = []
    emb_size, heads = 40, 10
    for band in range(len(DEFAULT_BANDS)):
        prefix = "band_" + str(band)
        x = BandSlice(band, name=prefix + "_input")(input_layer)
        x = L.Conv2D(40, (1, 25), padding="valid", name=prefix + "_temporal")(x)
        x = L.DepthwiseConv2D((chans, 1), use_bias=False,
                              depthwise_constraint=max_norm(1.),
                              name=prefix + "_spatial")(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("elu")(x)
        x = L.AveragePooling2D((1, 75), strides=(1, 15))(x)
        x = L.Dropout(dropout)(x)
        x = L.Conv2D(emb_size, (1, 1), name=prefix + "_projection")(x)
        x = L.Reshape((-1, emb_size))(x)
        x = L.LayerNormalization(center=False, scale=False,
                                 name=prefix + "_features")(x)
        features.append(x)
    x = _fuse(features, mode, embedding_size=emb_size)
    x = _add_broadband_residual(x, input_layer, chans, dropout, "EEGConformer", broadband_residual)
    # Same encoder dimensions/head as the current EEGModels_tf Conformer.
    for i in range(depth):
        residual = x
        x = L.LayerNormalization()(x)
        x = L.MultiHeadAttention(num_heads=heads, key_dim=emb_size // heads)(x, x)
        x = L.Dropout(dropout)(x)
        x = L.Add()([residual, x])
        residual = x
        x = L.LayerNormalization()(x)
        x = L.Dense(emb_size * 4, activation="gelu")(x)
        x = L.Dropout(.5)(x)
        x = L.Dense(emb_size)(x)
        x = L.Dropout(dropout)(x)
        x = L.Add()([residual, x])
    x = L.Flatten()(x)
    x = L.Dense(256, activation="elu")(x)
    x = L.Dropout(.5)(x)
    x = L.Dense(32, activation="elu")(x)
    x = L.Dropout(.3)(x)
    return L.Dense(nb_classes, activation="softmax")(x)


def build_model(model_arch, nb_classes, Chans, Samples, mode="off", dropout=.5, depth=3,
                broadband_residual="off"):
    """Construct the original model, four-band variant, or five-plane residual variant."""
    if mode not in FILTER_BANK_MODES:
        raise ValueError("Unknown filter-bank mode: " + str(mode))
    validate_residual_mode(mode, broadband_residual)
    if model_arch not in ("EEGNet", "EEGConformer"):
        raise ValueError("Unknown model architecture: " + str(model_arch))
    if Chans < 1 or nb_classes < 2 or depth < 1:
        raise ValueError("Positive channel/depth counts and at least two classes are required.")
    minimum = 32 if model_arch == "EEGNet" else 99
    if Samples < minimum:
        raise ValueError(model_arch + " requires at least " + str(minimum) + " samples.")
    if mode == "off":
        if model_arch == "EEGNet":
            return EEGNet(nb_classes, Chans=Chans, Samples=Samples, dropoutRate=dropout,
                          kernLength=32, F1=8, D=2, F2=16, dropoutType="Dropout")
        return EEGConformer(nb_classes, Chans=Chans, Samples=Samples, drop_p=dropout, depth=depth)
    input_planes = len(DEFAULT_BANDS) + int(broadband_residual != "off")
    inputs = L.Input(shape=(Chans, Samples, input_planes), name="eeg_bands")
    if model_arch == "EEGNet":
        outputs = _eegnet(inputs, nb_classes, Chans, dropout, mode, broadband_residual)
    else:
        outputs = _conformer(inputs, nb_classes, Chans, dropout, depth, mode, broadband_residual)
    suffix = "" if broadband_residual == "off" else "_BB_" + broadband_residual
    return tf.keras.Model(inputs, outputs, name=model_arch + "_FB_" + mode + suffix)
