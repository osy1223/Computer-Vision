# example of loading the keras facenet model
from keras.models import load_model

# load the model
model = load_model('facenet_keras.h5')

# summarize input and output shape
print(model.inputs)
print(model.outputs)
'''
[<KerasTensor: shape=(None, 160, 160, 3) dtype=float32 (created by layer 'input_1')>]
[<KerasTensor: shape=(None, 128) dtype=float32 (created by layer 'Bottleneck_BatchNorm')>]
'''

# confirm mtcnn was installed correctly
import mtcnn
# print version
print(mtcnn.__version__) # 0.1.0
