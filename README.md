# furniture-classification

##### Data processing
Sorted all images from 3 classes, in order of decreasing height. Batch images (of similarly sizes) together with batch size of 10 , along with the targets. Padded every image of the same batch to the largest width and length in said batch. All images were converted to grayscale and resized to 1/4 resolution to preserve memory and training time. The code for this part is in `dataset.ipynb` and the processed dataset is saved as `dataset_resized.pkl`

##### Model training
Split a subset of each batch (batches have varied sizes) into train and validation splits. Used a CNN model with GlobalMaxPooling since we have varied batch sizes. Also used Dropout and L1 L2 regularization to avoid overfitting and speed up training. The model achieved 79% accuracy on the validation set. This part was done on Google Colab for GPU training. The code can be found in `model.ipynb`, and the best model is saved as `best_model.hdf5`

##### API endpoint
Set up a Flask API with WSGI server for inference. The API takes a URL of an image, which then gets classified by the model to be either 'Bed', 'Sofa', or 'Chair', and the result is returned. The WSGI server is set up with 3 concurrent workers. The API is hosted on localhost at port 8081. The code for this part can be found in `model.py`, `app.py`, and `api.py`.

##### Docker
Created a Dockerfile to create a container for model inference and API endpoint. Docker container runs on ubuntu:22.04 and is exposed at port 8081. When ran, the container will start the WSGI server and start the inference service. The Dockerfile can be found at `Dockerfile`.

##### GitHub Actions CI/CD
Created a simple PyTest to test the input and output of the model. This code is in `model_test.py`. Created 2 GitHub workflow actions; the first is responsible for testing all unit tests in the repository, the second is to automatically build the Docker image. Both of these actions are triggered whenever there is a push or pull request to the main branch. These can be found at `.github/workflows/test.yml` and `.github/workflows/docker-image.yml`