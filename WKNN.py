import tensorflow as tf
K = tf.keras.backend


class WKNN(object):
    """
    # weighted KNN algorithm refers to the paper published by Klaus Hechenbichler and Klaus Schliep in 2004
        - paper link: https://epub.ub.uni-muenchen.de/1769/1/paper_399.pdf
        - code is developed with Tensorflow GPU 2.2.0 and Keras 2.3.0-tf
        - the paper provides 3 similarity calculations and 8 weight transfer kernels, but in this model only basics are implemented.
        - similarity calculation: Euclidean Distance
        - weight transfer calculation: Inversion Kernel

    # class arguments:
        - train_data: numpy array, tensor data,
        - train_labels: numpy array, tensor data,
        - k_value: must be an integer

    # predict method:
        - inputs: single data vector or a batch of data

    # return:
        - if the input is a single data, it will return a classification
        - if the input is a batch of data, it will return a classification list
    # initial unit test members work with test_WKNN.py
    """

    def __init__(self, train_data=None, train_labels=None, k_value=None, **kwargs):
        # super(WKNN, self).__init__(self, **kwargs)
        self._K = k_value
        self._data = tf.convert_to_tensor(train_data, dtype=tf.float32) if train_data.any() else None
        self._label = tf.convert_to_tensor(train_labels) if train_labels.any() else None
        self._inputs = None
        self._clf_res = []

    def __clf_predict(self, weights):
        # extract top kth weights and indices
        top_k_weights, top_k_indicies = tf.math.top_k(weights, self._K)
        # extract top kth labels
        top_k_labels = tf.gather(self._label, top_k_indicies)
        # cast labels to float type and concatenate to the weights
        top_k_labels = tf.cast(tf.expand_dims(top_k_labels, axis=1), tf.float32)
        weight_label_matrix = tf.concat([tf.expand_dims(top_k_weights, axis=1), top_k_labels], axis=1)
        # sort weight_label_matrix in ascending way based on the label order
        ascending_index = tf.argsort(weight_label_matrix[:, 1], direction='ASCENDING')
        sorted_weight_label_matrix = tf.gather(weight_label_matrix, ascending_index)
        # get segment for labels and weights
        segment = tf.cast(sorted_weight_label_matrix[:, 1], tf.int32)
        segment_weight = tf.expand_dims(sorted_weight_label_matrix[:, 0], axis=1)
        # sum all weights in each label segment
        weight_sum = tf.math.segment_sum(segment_weight, segment)
        # max weight is the predict classification type
        return tf.argmax(weight_sum).numpy()[0]

    def __calc_similarity(self, transition_matrix):
        similarity = tf.reduce_sum(tf.math.squared_difference(transition_matrix, self._data, name='similarity'), axis=1)
        return similarity

    def __inversion_kernel(self, similarity):
        zero_replacements = tf.math.multiply(tf.ones(shape=(similarity.shape)), 0.1)
        similarity = tf.where(tf.math.not_equal(similarity, 0), similarity, zero_replacements)
        return tf.math.divide_no_nan(1.0, similarity)

    def __wknn(self, row):
        # get a transistion matrix to expend 1*n to m*n, m is rows of training data
        trans_row = tf.ones(shape=(self._data.shape[0], 1))
        row = tf.expand_dims(row, axis=0)
        transition_matrix = tf.matmul(trans_row, row)
        # get similarity between input data and the train data
        similarity = self.__calc_similarity(transition_matrix)
        # transit similarity to weights
        weights = self.__inversion_kernel(similarity)
        # predict the classification
        self._clf_res.append(self.__clf_predict(weights))

    def __body(self, i):
        row = self._inputs[i]
        self.__wknn(row)
        i = tf.add(i,1)
        return (i,)

    def predict(self, inputs):
        # initialization test data
        self._inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # call algorithm in a loop
        index = tf.constant(0)
        c = lambda i: tf.less(i, self._inputs.shape[0])
        _ = tf.while_loop(c, self.__body, [index])
        # return result
        return self._clf_res

    '''
    - internal unit tests 
    '''
    def __method_for_calc_similarity_unit_test(self, row=[1., 1., 1.]):
        self._data = tf.Variable([[1, 2, 3],
                                  [1, 2, 3],
                                  [1, 2, 3]], dtype=tf.float32)
        trans_row = tf.ones(shape=(self._data.shape[0], 1))
        row = tf.expand_dims(row, axis=0)
        transition_matrix = tf.matmul(trans_row, row)
        res = self.__calc_similarity(transition_matrix)
        target = tf.convert_to_tensor([5, 5, 5], dtype=tf.float32)
        return True if tf.equal(res, target).numpy().all() else False

    def __method_for_inversion_kernel_unit_test(self,
                                                similarity=tf.Variable([0., 0., 0.], dtype=tf.float32)):
        res = self.__inversion_kernel(similarity)
        target = tf.Variable([10, 10, 10], dtype=tf.float32)
        return tf.equal(res, target).numpy().all()

    def __method_for_clf_predict_unit_test(self, weights=tf.Variable([0.1, 0.5, 0.9, 0.2,
                                                                      0.4, 0.6, 0.8, 0.7,
                                                                      0.5, 0.3, 0.6, 0.2,
                                                                      0.1], dtype=tf.float32)):
        self._K = 8
        self._label = tf.constant([0, 1, 2, 3, 1, 1, 2, 1, 2, 1, 1, 3, 3])
        res = self.__clf_predict(weights)
        return res == 1

    def __method_for_wknn_unit_test(self, row=[1., 1., 1.]):
        self._data = tf.Variable([[1, 2, 3],
                                  [2, 2, 2],
                                  [3, 3, 3],
                                  [4, 4, 5],], dtype=tf.float32)
        self._label = tf.constant([0, 1, 0, 1])
        self._K = 4
        self.__wknn(row)
        target = [1]
        return self._clf_res == target

    def __method_for_predict_unit_test(self, inputs=[[1., 1., 1.],
                                                  [5., 5., 5.]]):
        self._data = tf.Variable([[1, 2, 3],
                                  [2, 2, 2],
                                  [3, 3, 3],
                                  [4, 4, 5],
                                  [5, 5, 5],
                                  [6, 6, 6],
                                  [1, 1, 1],], dtype=tf.float32)
        self._label = tf.constant([0, 1, 0, 1, 1, 0, 0])
        self._K=7
        self.call(inputs)
        target =[0,1]
        return self._clf_res == target















