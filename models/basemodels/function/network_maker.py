from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def get_transformed_base_model(base_model):
    for layer in base_model.layers:
        layer.trainable = False
    
    input_layer = Input(shape=base_model.input_shape[1:])

    base_model_output = base_model(input_layer)

    base_output_units = base_model_output.shape[-1]

    dense_layer = Dense(base_output_units, activation='relu')(base_model_output)
    # 이렇게 쓰면 1억개 넘는 패러미터가 또 나옴, 더블 덴스레이어를 붙여도 수천만 단위, 기본 덴스를 붙여도 천만
    # quadruple_dense_layer = Dense(base_output_units * 4, activation='relu')(double_dense_layer) 
    # half_quadruple_dense_layer = Dense(base_output_units * 2, activation='relu')(quadruple_dense_layer)

    # 결국 동일 아웃풋 덴스 붙이는 게 최선의 성능
    transformed_model = Model(inputs=input_layer, outputs=dense_layer)

    return transformed_model
