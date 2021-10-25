from mahapala import MahaPala
from mahapala.fruits_classification import FruitsClassification


def test_fruits_classification():
    fc = FruitsClassification()
    fc.fit()
    r = fc.predict()
    print(r)


def test_image_sampling():
    mp = MahaPala()
    mp.input()
    mp.sample()
    mp.process()
    mp.plot_output()


if __name__ == "__main__":
    test_image_sampling()
    test_fruits_classification()
