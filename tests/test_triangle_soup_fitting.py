"""Test using triangle soup rendering."""

from deodr.examples.triangle_soup_fitting import run


def test_soup():
    final_loss = run(nb_max_iter=50, display=False)
    assert abs(final_loss[False] - 1331.141082325624) < 10
    assert abs(final_loss[True] - 1457.4429407318612) < 20


if __name__ == "__main__":
    test_soup()
