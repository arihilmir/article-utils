import keras
from keras import ops
from keras import layers as l

from .tcn import TCN

def gru_block(hidden_size, layers, out_size=24, input_shape=(72, 4), bsz=4):
    model = keras.Sequential()
    model.add(l.Input(input_shape))
    for _ in range(layers-1):
        model.add(l.GRU(hidden_size, return_sequences=True))

    model.add(l.GRU(hidden_size))
    model.add(l.Dense(out_size, activation='silu'))
    return model


def get_big_model(num_units, num_layers, out_size=24, input_shape=(72, 4), bsz=4):
    model = keras.Sequential()
    model.add(l.Input((72, 4), batch_size=4))

    for _ in range(num_layers):
        model.add(l.GRU(num_units, return_sequences=True))

    model.add(l.GRU(num_units))
    model.add(l.Dense(64, activation='silu'))
    model.add(l.Dense(24))
    return model


def separated_model(hidden_size, layers, out_size=24, input_shape=(72, 4), bsz=4):
    inputs = l.Input(input_shape, bsz)
    pc, w = ops.split(inputs, [1], axis=-1)
    for _ in range(layers-1):
        pc = l.GRU(hidden_size, return_sequences=True)(pc)
        w = l.GRU(hidden_size, return_sequences=True)(w)
    pc = l.GRU(hidden_size)(pc)
    w = l.GRU(hidden_size)(w)
    xs = ops.stack([pc, w], axis=-1)
    out = l.GRU(2*hidden_size, name='stacking_gru')(xs)
    out = l.Dense(128, activation='silu', name='stacking_dense_1')(out)
    out = l.Dense(out_size, name='stacking_dense_2')(out)
    model = keras.Model(inputs=inputs, outputs=out, name="Stacking")
    return model


def stacking_model(models_paths: list[str], input_shape=(72, 4), bsz=4):
    inputs = l.Input(input_shape, batch_size=bsz)
    models = [keras.saving.load_model(path) for path in models_paths]
    for j in range(len(models)):
        m = models[j]
        m.name = f'block_{j}_' + m.name
        for i in range(len(m.layers)):
            m.layers[i].name = f'model{i}_' + m.layers[i].name
            m.layers[i].trainable = False

    separate_outputs = [ops.expand_dims(m(inputs), axis=-1) for m in models]
    merge = l.Concatenate()(separate_outputs)
    out = l.GRU(64, name='stacking_gru')(merge)
    out = l.Dense(128, activation='silu', name='stacking_dense_1')(out)
    out = l.Dense(24, name='stacking_dense_2')(out)
    model = keras.Model(inputs=inputs, outputs=out, name="Stacking")
    return model

# MARK: TCN

def tcn_gru_block(
    nb_filters: int,
    kernel_size: int,
    nb_stacks: int,
    dilations: list[int],
    gru_units: int = 128,
    activation: str = 'relu',
    out_size: int = 24,
    input_shape: tuple[int, int] = (72, 4),
    bsz: int = 4
):
    i = l.Input(shape=input_shape, batch_size=bsz)
    o = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,
        return_sequences=True,
        activation='relu'
    )(i)
    o = l.GRU(gru_units, recurrent_activation='silu', stateful=True)(o)
    o = l.Dense(out_size, activation='silu')(o)
    o = l.Reshape((out_size, 1))(o)
    return keras.Model(i, o)


def tcn_ensemble(models_paths: list[str], input_shape=(72, 4), bsz=4):
    inputs = l.Input(input_shape, batch_size=bsz)
    models = [keras.saving.load_model(path) for path in models_paths]
    for j in range(len(models)):
        m = models[j]
        m.name = f'block_{j}_' + m.name
        for i in range(len(m.layers)):
            m.layers[i].name = f'model{i}_' + m.layers[i].name
            m.layers[i].trainable = False

    separate_outputs = [m(inputs) for m in models]
    merge = l.Concatenate()(separate_outputs)
    out = l.GRU(128, name='stacking_gru')(merge)
    out = l.Dense(128, activation='silu', name='stacking_dense_1')(out)
    out = l.Dense(24, name='stacking_dense_2')(out)
    model = keras.Model(inputs=inputs, outputs=out, name="Stacking")
    return model
