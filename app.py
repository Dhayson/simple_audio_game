# Import libraries
from time import sleep
import numpy as np

# Import tensorflow and keras libraries
import tensorflow as tf
from keras.models import load_model  # TensorFlow is required for Keras to work
import soundfile as sf
import sounddevice as sd
from scipy.io.wavfile import write
import pyaudio
import wave
import resample

def get_microphone_sample_rate(device_index=None):
    """
    Retrieves the default sample rate for a given microphone device.
    If no device_index is provided, it attempts to get the default input device's sample rate.
    """
    p = pyaudio.PyAudio()
    try:
        if device_index is None:
            # Try to get the default input device info
            info = p.get_default_input_device_info()
            device_index = info['index']
        else:
            info = p.get_device_info_by_index(device_index)

        # The 'defaultSampleRate' key contains the default sample rate
        sample_rate = int(info['defaultSampleRate'])
        print("sample rate is ", sample_rate)
        return sample_rate
    except Exception as e:
        print(f"Error getting microphone sample rate: {e}")
        return None
    finally:
        p.terminate()

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="soundclassifier_with_metadata.tflite")
interpreter.allocate_tensors()

# Get input and output details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the labels.
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()
   
def classify_sample(audio_data, samplerate):
    # Ensure the audio is mono and resample if necessary
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0] # Take one channel
    if samplerate != 44032: # Assuming shape[1] is the expected sample rate
        # You might need to resample the audio here using libraries like librosa
        audio_data = resample.resample(audio_data, samplerate, 44032)

    # Pad or truncate the audio to the expected length (e.g., 1 second)
    expected_length = 44032
    if len(audio_data) < expected_length:
        audio_data = np.pad(audio_data, (0, expected_length - len(audio_data)))
    elif len(audio_data) > expected_length:
        audio_data = audio_data[:expected_length]

    # Normalize audio data if required by your model
    # Teachable Machine models often expect float32 input in a specific range (e.g., -1 to 1)
    input_data = audio_data.astype(np.float32)
    input_data = (input_data - 0.5)*2
    input_data = np.expand_dims(input_data, axis=0) # Add batch dimension

    # Set the tensor.
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference.
    interpreter.invoke()

    # Get the output tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process the output (e.g., get the most likely class).
    predicted_class_index = np.argmax(output_data)
    predicted_label = labels[predicted_class_index]
    confidence = output_data[0][predicted_class_index]

    print(f"Predicted class: {predicted_label} with confidence: {confidence:.2f}")

audio_rate = get_microphone_sample_rate()
seconds = 1

# print(sd.query_devices())
sd.default.device = 9

def listen_audio():
    # Record audio from the default microphone
    my_recording = sd.rec(int(seconds * audio_rate), samplerate=audio_rate, channels=1, dtype='float32')
    return my_recording

voice = listen_audio()
sleep(1)
while True:
    classify_sample(voice, audio_rate)
    sf.write("my_recording5.wav", voice, audio_rate)
    voice = listen_audio()
    sleep(1)

print(voice)