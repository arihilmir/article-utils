import keras
from keras import ops
from keras import layers as l

def lstm_block(hidden_size, layers, out_size=24, input_shape=(72, 4)):
    model = keras.Sequential()
    model.add(l.Input(input_shape))
    for _ in range(layers-1):
        model.add(l.GRU(hidden_size, return_sequences=True))

    model.add(l.GRU(hidden_size))
    model.add(l.Dense(out_size, activation='silu'))
    return model

def stacking_model(models_paths: list[str], input_shape = (72,4), bsz=4):
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
    out =  l.GRU(64, name='stacking_gru')(merge)
    out = l.Dense(128, activation='silu', name='stacking_dense_1')(out)
    out = l.Dense(24, name='stacking_dense_2')(out)
    model = keras.Model(inputs=inputs, outputs=out, name="Stacking")
    return model