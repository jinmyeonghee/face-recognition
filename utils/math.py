import tensorflow as tf


def get_contrastive_loss(margin=1):
    def contrastive_loss(y_true, y_pred):
      '''
      함수 목적 : 대조 손실 계산 함수
      인풋 : 
        - y_true : 실제 값 list
        - y_pred : 예측 값 list
      아웃풋 : float type의 대조손실값으로 이루어진 tensor
      '''
      y_true = tf.cast(y_true, tf.float32)
      square_pred = tf.math.square(y_pred)
      margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
      return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss
# -----------------------------------


class distanceLayer(tf.keras.layers.Layer):
  def call(self, vectors):
    '''
    함수 목적 : 두 벡터의 유클리디안 거리 계산
    인풋 : Vector([vector,vector] 형식)
    아웃풋 : 유클리디안 거리 값
    '''
    self.vectors = vectors
    self.featsA, self.featsB = self.vectors
    sumSquared = tf.math.reduce_sum(tf.math.square(self.featsA - self.featsB), axis = 1, keepdims = True)
    return tf.math.sqrt(tf.math.maximum(sumSquared, tf.keras.backend.epsilon()))