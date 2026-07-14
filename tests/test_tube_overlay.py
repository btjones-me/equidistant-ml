from scripts.export_tube_overlay import decode_line_paths


def test_decode_line_paths_converts_tfl_coordinates_for_leaflet() -> None:
    encoded = "[[[-0.100606,51.494536],[-0.112315,51.498808]]]"

    assert decode_line_paths(encoded) == [
        [[51.494536, -0.100606], [51.498808, -0.112315]]
    ]
