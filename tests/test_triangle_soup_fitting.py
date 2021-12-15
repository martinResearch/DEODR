"""Test using triangle soup rendering."""

from deodr.examples.triangle_soup_fitting import run


def test_triangle_soup_fitting(check_hashes=True):

    losses, hashes = run(
        nb_max_iter=50, display=False, clockwise=False, antialiase_error=False
    )
    assert abs(losses[-1] - 1331.141082325624) < 10

    if check_hashes:
        assert (
            hashes[0]
            == "38b6f6954374230aeb1ce5d804308522f6b4c58a6736a040aeef7f2176a20b28"
        )
        assert (
            hashes[1]
            == "0434ea722edb9e3364da9b0e8564c3002b9aa3b12791ba8f089689beecd3c4e9"
        )

    losses, hashes = run(
        nb_max_iter=50, display=False, clockwise=False, antialiase_error=True
    )
    assert abs(losses[-1] - 1457.4429407318612) < 20
    if check_hashes:
        assert (
            hashes[0]
            == "82a7b73fde3615ef7c70008965f4bfda8610b9001c20dd435a880bf45a31d3d6"
        )
        assert (
            hashes[1]
            == "0de2e8b80730cfc444d0552cd81e5071897a525ec6495e643ca17fb0792496c0"
        )

    losses, hashes = run(
        nb_max_iter=50, display=False, clockwise=True, antialiase_error=False
    )
    assert abs(losses[-1] - 1331.141082325624) < 10
    if check_hashes:
        assert (
            hashes[0]
            == "eb9f335acdeb2abc3e029826a56fd77bd7a4bb39c2794d9972735a9f388ddeba"
        )
        assert (
            hashes[1]
            == "6b4cc11e56dfebe2b8235485ea1bb8230a5765e7f0eeb2ac671d9cd4e3311f74"
        )

    losses, hashes = run(
        nb_max_iter=50, display=False, clockwise=True, antialiase_error=True
    )
    assert abs(losses[-1] - 1457.4429407318612) < 20
    if check_hashes:
        assert (
            hashes[0]
            == "d2113e271f29afbdac393767297b923caaa4fe19af4bdd16d3b649c2f5ba3103"
        )
        assert (
            hashes[1]
            == "8820a5f90b4c0496b88542f42ed773afcd7cc0e0e3edc7a3da6427e2ab881d44"
        )


if __name__ == "__main__":
    test_triangle_soup_fitting()
