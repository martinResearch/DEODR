"""Test using triangle soup rendering."""

from typing import Any, Dict, List

from deodr.examples.triangle_soup_fitting import run


def check_results(
    all_lkg_results: List[Dict[str, Any]],
    losses: List[float],
    hashes: List[str],
    check_hashes: bool,
) -> None:
    for lkg_results in all_lkg_results:
        valid = all(losses[key] == value for key, value in lkg_results["losses"].items())

        if check_hashes:
            for key, value in lkg_results["hashes"].items():
                if hashes[key] != value:
                    valid = False
        if valid:
            return
    raise BaseException("None of the results matches")


def test_triangle_soup_fitting(check_hashes: bool = True) -> None:
    losses, hashes = run(nb_max_iter=50, display=False, clockwise=False, antialiase_error=False)

    lkg_results = [
        {  # windows
            "losses": {-1: 1331.3578738815468},
            "hashes": {
                0: "38b6f6954374230aeb1ce5d804308522f6b4c58a6736a040aeef7f2176a20b28",
                1: "0434ea722edb9e3364da9b0e8564c3002b9aa3b12791ba8f089689beecd3c4e9",
            },
        },
        {  # google colab Intel(R) Xeon(R) CPU @ 2.20GHz: 251.3164914350016
            "losses": {-1: 1328.235645237829},
            "hashes": {
                0: "f0571dcd10df3c81c902b703462b3c94139eab3d88bc8c081c1dca1c2533fc4c",
                1: "fe8416554d92528a13ebed3cb0d9d376abfbce6171c18d831ed765b0bbc7198a",
            },
        },
    ]

    check_results(lkg_results, losses, hashes, check_hashes)

    losses, hashes = run(nb_max_iter=50, display=False, clockwise=False, antialiase_error=True)

    lkg_results = [
        {
            # windows
            "losses": {-1: 1457.8585914203582},
            "hashes": {
                0: "82a7b73fde3615ef7c70008965f4bfda8610b9001c20dd435a880bf45a31d3d6",
                1: "0de2e8b80730cfc444d0552cd81e5071897a525ec6495e643ca17fb0792496c0",
            },
        },
        {  # google colab Intel(R) Xeon(R) CPU @ 2.20GHz: 251.3164914350016
            "losses": {-1: 1456.4362294737318},
            "hashes": {
                0: "55b545a8ae70b51ed0ad762210daf7b3bf568c786f4ddd5716ca6ec6479951ba",
                1: "e8c598184d89fe966ffe3efdfc2ca2500191f92b1e40d71f9213847c90982a1f",
            },
        },
    ]

    check_results(lkg_results, losses, hashes, check_hashes)

    losses, hashes = run(nb_max_iter=50, display=False, clockwise=True, antialiase_error=False)

    lkg_results = [
        {  # windows
            "losses": {-1: 1331.357873881545},
            "hashes": {
                0: "eb9f335acdeb2abc3e029826a56fd77bd7a4bb39c2794d9972735a9f388ddeba",
                1: "6b4cc11e56dfebe2b8235485ea1bb8230a5765e7f0eeb2ac671d9cd4e3311f74",
            },
        },
        {  # google colab Intel(R) Xeon(R) CPU @ 2.20GHz: 251.3164914350016
            "losses": {-1: 1328.2356452378326},
            "hashes": {
                0: "427c434a1e60537c57aa2d6a5a990e425c064fa9690955014f2dd7c087b30cfc",
                1: "6882ed3113f6b33505deeb5366d75ec48ffece0870af9ab73df74e9aa5c8f870",
            },
        },
    ]

    check_results(lkg_results, losses, hashes, check_hashes)
    losses, hashes = run(nb_max_iter=50, display=False, clockwise=True, antialiase_error=True)

    lkg_results = [
        {
            "losses": {-1: 1457.8585914203607},
            "hashes": {
                0: "d2113e271f29afbdac393767297b923caaa4fe19af4bdd16d3b649c2f5ba3103",
                1: "8820a5f90b4c0496b88542f42ed773afcd7cc0e0e3edc7a3da6427e2ab881d44",
            },
        },
        {  # google colab Intel(R) Xeon(R) CPU @ 2.20GHz: 251.3164914350016
            "losses": {-1: 1456.4362294737307},
            "hashes": {
                0: "0b41270c59ddf37d8fd7e9ff0f0723042bd6ee0546ac071dadb306f65aa143ad",
                1: "11e225715108f8fa4a307bd5c8ed2d1891abaec83290a8fcb3677e58d258ccfd",
            },
        },
    ]

    check_results(lkg_results, losses, hashes, check_hashes)


if __name__ == "__main__":
    test_triangle_soup_fitting()
