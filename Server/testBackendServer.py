import requests
import base64

file_path_to_img = "/data/lam138/DogIdentification/DogIdentification_S24/CNN/DogDataset/testing_data/golden_retriever/gr77.jpg" # Change this to the file path of the image you want to upload to the server

with open(file_path_to_img, "rb") as image_file:
    # Read the binary data from the image file
    binary_data = image_file.read()
    
    # Encode the binary data to Base64
    base64_data = base64.b64encode(binary_data)

    # Convert the bytes-like object to a string for printing or further processing
    base64_string = base64_data.decode('utf-8')

## API Endpoint to Upload/Preprocess Image to the server ###
url = 'https://bluepill.ecn.purdue.edu/~lam138/uploadImage.php'
# Create the payload with the Base64 encoded image
payload = {'image': base64_string}

# Send the POST request with multipart/form-data
response = requests.post(url, payload)
# Print the response text
print(response.text)
#################################################################
    
### API Endpoint to make a prediction on said image ###
url = 'https://bluepill.ecn.purdue.edu/~lam138/executePy.php'
response = requests.post(url)
# Print the response text
print(response.text)
##################################################################
