from pytorch_lightning import Trainer
from argparse import ArgumentParser
from research_seed.digit_recognition.mnist import MNISTRecognizer


def main():
    model = MNISTRecognizer()
    trainer = Trainer(
        max_nb_epochs=50,
        gpus=0
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()