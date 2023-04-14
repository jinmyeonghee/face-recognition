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

class L2EuclideanDistanceLayer(tf.keras.layers.Layer):
  def call(self, vectors):
    '''
    함수 목적: 두 벡터의 L2 유클리디안 거리 계산
    인풋: Vector([vector, vector] 형식)
    아웃풋: L2 유클리디안 거리 값
    '''
    featsA, featsB = vectors
    sum_squared = tf.math.reduce_sum(tf.math.square(featsA - featsB), axis=1, keepdims=True)
    l2_distance = tf.math.sqrt(tf.math.maximum(sum_squared, tf.keras.backend.epsilon()))
    return l2_distance

class CosineSimilarityLayer(tf.keras.layers.Layer):
  def call(self, feature_vectors):
    '''
    함수 목적 : 두 벡터의 코사인 거리 계산
    인풋 : Vector([vector,vector] 형식)
    아웃풋 : 코사인 거리 값
    '''
    feat_vector1, feat_vector2 = feature_vectors
    normalized_feat_vector1 = tf.nn.l2_normalize(feat_vector1, axis=1)
    normalized_feat_vector2 = tf.nn.l2_normalize(feat_vector2, axis=1)
    cosine_similarity = tf.reduce_sum(tf.multiply(normalized_feat_vector1, normalized_feat_vector2), axis=1, keepdims=True)
    return cosine_similarity
  
def get_layer(features_img1, features_img2, distance_metric = 'cosine'):
  if distance_metric == 'cosine':
    return CosineSimilarityLayer()([features_img1, features_img2])
  elif distance_metric == 'euclidean':
    return distanceLayer()([features_img1, features_img2])
  elif distance_metric == 'euclidean_l2':
    return L2EuclideanDistanceLayer()([features_img1, features_img2])