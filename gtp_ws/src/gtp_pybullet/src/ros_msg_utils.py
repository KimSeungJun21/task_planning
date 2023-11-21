from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import numpy as np

def numpy_to_float64_multiarray(numpy_array):
    # NumPy 배열을 Float64MultiArray 메시지로 변환합니다.
    msg = Float64MultiArray()
    # 다차원 배열의 shape을 메시지에 설정합니다.
    msg.layout.dim.append(MultiArrayDimension(label='rows', size=numpy_array.shape[0], stride=numpy_array.shape[0] * numpy_array.shape[1]))
    msg.layout.dim.append(MultiArrayDimension(label='cols', size=numpy_array.shape[1], stride=numpy_array.shape[1]))
    # 데이터를 메시지에 추가합니다.
    msg.data = numpy_array.flatten().tolist()
    return msg

def float64_multiarray_to_numpy(msg):
    # Float64MultiArray 메시지에서 데이터를 가져와서 NumPy 배열로 변환합니다.
    flattened_data = np.array(msg.data, dtype=np.float64)
    # NumPy 배열을 원래 shape으로 reshape합니다.
    numpy_array = flattened_data.reshape([dim.size for dim in msg.layout.dim])
    return numpy_array

