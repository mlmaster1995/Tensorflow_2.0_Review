import tensorflow as tf


def gpu_cpu_id():
    print(f'Tensorflow GPU Version: {tf.__version__}')
    print(f'Eager Execution is: {tf.executing_eagerly()}')
    print(f'Keras Version: {tf.keras.__version__}')

    var = tf.Variable([3,3])
    if tf.test.is_gpu_available():
        print('Running on GPU')
    else:
        print('Runing on CPU')


gpu_cpu_id()