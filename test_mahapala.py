from mahapala import MahaPala, FruitsClassification


def test_fruits_classification():
    fc = FruitsClassification()
    fc.epochs = 1
    fc.fit()

    assert fc.history is not None
    assert "accuracy" in fc.history.history

    r = fc.predict()

    assert len(r) == 1
    assert "class" in r[0]
    assert "confidence" in r[0]


def test_image_sampling():
    mp = MahaPala()
    mp.input()
    mp.sample()
    mp.process()

    assert len(mp.plot_output_list) == 1
    res, img, method = mp.plot_output_list[0]
    assert res is not None
    assert img is not None


if __name__ == "__main__":
    test_image_sampling()
    test_fruits_classification()

    mp = MahaPala()
    mp.input()
    mp.sample()
    mp.process()
    mp.plot_output()
