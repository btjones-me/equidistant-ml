from scripts.export_tube_overlay import decode_line_paths, smooth_path


def test_decode_line_paths_converts_tfl_coordinates_for_leaflet() -> None:
    encoded = "[[[-0.100606,51.494536],[-0.112315,51.498808]]]"

    assert decode_line_paths(encoded) == [
        [[51.494536, -0.100606], [51.498808, -0.112315]]
    ]


def test_smooth_path_preserves_endpoints_and_rounds_corners() -> None:
    path = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]

    smoothed = smooth_path(path, iterations=1)

    assert smoothed[0] == path[0]
    assert smoothed[-1] == path[-1]
    assert smoothed[1:-1] == [
        [0.25, 0.0],
        [0.75, 0.0],
        [1.0, 0.25],
        [1.0, 0.75],
    ]
