import tensorflow as tf
import numpy as np
import soundfile as sf
import sounddevice as sd

import resample

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="soundclassifier_with_metadata.tflite")
interpreter.allocate_tensors()

# Get input and output details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("input details: ", input_details)
print()
print("output details: ", output_details)
print()
default_output_device_info = sd.query_devices()
print("devices info: ", default_output_device_info)
