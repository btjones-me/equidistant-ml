from scripts.export_tube_overlay import (
    canonicalize_line_paths,
    decode_line_paths,
    smooth_path,
)


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


def test_canonicalize_line_paths_collapses_reversed_duplicates() -> None:
    path = [[51.5, -0.2], [51.51, -0.19], [51.52, -0.18]]

    canonical = canonicalize_line_paths(
        [path, list(reversed(path))], smoothing_iterations=0
    )

    assert canonical == [path] or canonical == [list(reversed(path))]


def test_canonicalize_line_paths_prefers_detailed_nearby_corridor() -> None:
    west = [51.56, -0.30]
    junction = [51.55, -0.25]
    intermediate = [51.556, -0.20]
    rejoin = [51.55, -0.15]
    east = [51.53, -0.10]
    direct = [junction, rejoin]
    detailed = [junction, intermediate, rejoin]

    canonical = canonicalize_line_paths(
        [
            [west, *detailed, east],
            [[51.57, -0.31], *direct, east],
            [[51.54, -0.32], *direct, east],
        ],
        smoothing_iterations=0,
    )
    edges = {
        (tuple(start), tuple(end))
        for path in canonical
        for start, end in zip(path, path[1:])
    }

    assert (tuple(junction), tuple(rejoin)) not in edges
    assert any(intermediate in path for path in canonical)
    assert {tuple(path[0]) for path in canonical} | {
        tuple(path[-1]) for path in canonical
    } >= {tuple(west), (51.57, -0.31), (51.54, -0.32), tuple(east)}


def test_canonicalize_line_paths_keeps_materially_separate_branches() -> None:
    junction = [51.55, -0.20]
    rejoin = [51.50, -0.10]
    north_branch = [junction, [51.56, -0.14], rejoin]
    south_branch = [junction, [51.49, -0.17], rejoin]

    canonical = canonicalize_line_paths(
        [north_branch, south_branch], smoothing_iterations=0
    )

    assert len(canonical) == 2
    assert any([51.56, -0.14] in path for path in canonical)
    assert any([51.49, -0.17] in path for path in canonical)


def test_canonicalize_line_paths_removes_shared_stop_shortcut() -> None:
    junction = [51.54, -0.20]
    intermediate = [51.545, -0.18]
    rejoin = [51.54, -0.16]
    direct = [junction, rejoin]
    detailed = [junction, intermediate, rejoin]

    canonical = canonicalize_line_paths(
        [
            [[51.55, -0.22], *detailed, [51.53, -0.14]],
            [[51.56, -0.23], *detailed, [51.52, -0.13]],
            [[51.57, -0.24], *direct, [51.51, -0.12]],
        ],
        smoothing_iterations=0,
    )
    edges = {
        (tuple(start), tuple(end))
        for path in canonical
        for start, end in zip(path, path[1:])
    }

    assert (tuple(junction), tuple(rejoin)) not in edges
    assert any(intermediate in path for path in canonical)
