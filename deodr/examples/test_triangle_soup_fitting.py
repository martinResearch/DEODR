from deodr.examples.triangle_soup_fitting import run


def test_soup():
    final_loss = run(nb_max_iter=50, display=False)
    assert final_loss[False] == 1299.3337915732968
    assert final_loss[True] == 1406.0760211256033
