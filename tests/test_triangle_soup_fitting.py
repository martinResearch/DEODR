"""Test using triangle soup rendering."""

from deodr.examples.triangle_soup_fitting import run


def check_results(lkg_results, losses, hashes, check_hashes):
    platform = "windows"

    for key, value in lkg_results[platform]["losses"].items():
        assert losses[key] == value
    if check_hashes:
        for key, value in lkg_results[platform]["hashes"].items():
            assert hashes[key] == value


def test_triangle_soup_fitting(check_hashes=True):

    losses, hashes = run(
        nb_max_iter=50, display=False, clockwise=False, antialiase_error=False
    )

    lkg_results = {
        "windows": {
            "losses": {-1: 1331.3578738815468},
            "hashes": {
                0: "38b6f6954374230aeb1ce5d804308522f6b4c58a6736a040aeef7f2176a20b28",
                1: "0434ea722edb9e3364da9b0e8564c3002b9aa3b12791ba8f089689beecd3c4e9",
            },
        }
    }
    check_results(lkg_results, losses, hashes, check_hashes)

    losses, hashes = run(
        nb_max_iter=50, display=False, clockwise=False, antialiase_error=True
    )

    lkg_results = {
        "windows": {
            "losses": {-1: 1457.8585914203582},
            "hashes": {
                0: "82a7b73fde3615ef7c70008965f4bfda8610b9001c20dd435a880bf45a31d3d6",
                1: "0de2e8b80730cfc444d0552cd81e5071897a525ec6495e643ca17fb0792496c0",
            },
        }
    }
    check_results(lkg_results, losses, hashes, check_hashes)

    losses, hashes = run(
        nb_max_iter=50, display=False, clockwise=True, antialiase_error=False
    )

    lkg_results = {
        "windows": {
            "losses": {-1: 1331.357873881545},
            "hashes": {
                0: "eb9f335acdeb2abc3e029826a56fd77bd7a4bb39c2794d9972735a9f388ddeba",
                1: "6b4cc11e56dfebe2b8235485ea1bb8230a5765e7f0eeb2ac671d9cd4e3311f74",
            },
        }
    }

    check_results(lkg_results, losses, hashes, check_hashes)
    losses, hashes = run(
        nb_max_iter=50, display=False, clockwise=True, antialiase_error=True
    )

    lkg_results = {
        "windows": {
            "losses": {-1: 1457.8585914203607},
            "hashes": {
                0: "d2113e271f29afbdac393767297b923caaa4fe19af4bdd16d3b649c2f5ba3103",
                1: "8820a5f90b4c0496b88542f42ed773afcd7cc0e0e3edc7a3da6427e2ab881d44",
            },
        }
    }

    check_results(lkg_results, losses, hashes, check_hashes)


if __name__ == "__main__":
    test_triangle_soup_fitting()
