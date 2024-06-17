#
# pip install onnx onnxruntime onnxruntime-tools
#
import onnxruntime as ort
import numpy as np

DATA_NUM = 10_000
QTY_RANGE = 10
LEARNING_RATE = 0.001
EPOCHS = 25
BATCH_SIZE = 32

# Define the cost of each item
watermelon_cost = 100
apple_cost = 10
grape_cost = 1

# Load the ONNX model
ort_session = ort.InferenceSession('onnx/model.onnx')
'''
Ensure that the input name 'input' matches the input node name defined in your ONNX model. 
You can inspect the model's input and output names using the following code:
If the input name is different, replace 'input' in the run method with the correct input name.
'''
# Inspect input and output names
input_name = [input.name for input in ort_session.get_inputs()][0]
print("Input names: ", [input.name for input in ort_session.get_inputs()])
print("Output names: ", [output.name for output in ort_session.get_outputs()])

# input
import random
watermelon_qty = random.randint(1, QTY_RANGE)
apple_qty = random.randint(1, QTY_RANGE)
grape_qty = random.randint(1, QTY_RANGE)
exact = (watermelon_qty * watermelon_cost) + (apple_qty * apple_cost) + (grape_qty * grape_cost)


# Example input data
input_data = np.array([[watermelon_qty, apple_qty, grape_qty]], dtype=np.float32)
# Perform inference
#outputs = ort_session.run(None, {'input': input_data})
outputs = ort_session.run(None, {input_name: input_data})
# Print the outputs
print(f"{watermelon_qty} {apple_qty} {grape_qty} = {exact}, predict:{outputs[0][0].item():.0f}")

