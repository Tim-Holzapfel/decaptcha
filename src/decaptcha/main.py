"""Train a neural network to solve ReCaptchas."""


# Package Library
from decaptcha.models.train_model import TrainModel


if __name__ == "__main__":
    train_model: TrainModel = TrainModel(
        model_name="model_checkpoint", epochs=10, continue_loop=True
    )
    train_model()
