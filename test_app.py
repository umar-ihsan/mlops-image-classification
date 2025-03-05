import pytest
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from unittest.mock import MagicMock, patch


# Mock model loading to avoid loading the actual model
@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.random.rand(1, 36)  # Mock prediction for 36 classes
    return model


# Test if model loads correctly
@patch("tensorflow.keras.models.load_model")
def test_model_loading(mock_load_model):
    mock_load_model.return_value = MagicMock()
    model = load_model("Image_classify_2.keras")
    assert model is not None, "Model should load successfully"


# Test image preprocessing
def test_image_preprocessing():
    img_height, img_width = 180, 180
    image = tf.keras.utils.img_to_array(
        tf.keras.utils.load_img("test.jpg", target_size=(img_height, img_width))
    )
    assert image.shape == (img_height, img_width, 3), "Image shape should match input size"


# Test prediction function
@patch("tensorflow.keras.models.load_model")
def test_prediction(mock_load_model, mock_model):
    mock_load_model.return_value = mock_model
    model = load_model("Image_classify_2.keras")

    img_height, img_width = 180, 180
    image = tf.keras.utils.img_to_array(
        tf.keras.utils.load_img("test.jpg", target_size=(img_height, img_width))
    )
    img_bat = tf.expand_dims(image, 0)

    prediction = model.predict(img_bat)
    assert prediction.shape == (1, 36), "Prediction output shape should match number of classes"
    assert np.argmax(prediction) in range(36), "Predicted index should be valid"

    score = tf.nn.softmax(prediction)
    assert 0 <= np.max(score) <= 1, "Softmax confidence should be between 0 and 1"
