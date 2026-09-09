"""Bounded, paired weekday departure probe; never trains or changes the atlas.

Run from the repository root with uv run python scripts/probe_evening.py.
Add --execute to make at most 24 TravelTime searches (six requests), paced
at four searches per minute. Existing results are reused, never overwritten.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import dotenv_values

AREAS = [
    ("West Hampstead", 51.5518, -0.1956),
    ("De Beauvoir", 51.5364, -0.0750),
    ("Putney", 51.4701, -0.2106),
    ("Waterloo", 51.5033, -0.1195),
    ("Brixton", 51.4628, -0.1140),
    ("Stratford", 51.5413, -0.0033),
    ("Ealing", 51.5149, -0.3020),
    ("Wembley", 51.5520, -0.2960),
    ("Greenwich", 51.4780, -0.0148),
    ("Richmond", 51.4613, -0.3037),
    ("Eltham", 51.4556, 0.0525),
    ("Hackney", 51.5471, -0.0554),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-09-09")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    day = datetime.strptime(args.date, "%Y-%m-%d")
    if day.weekday() >= 5:
        raise SystemExit("Choose a weekday.")
    times = {
        label: day.replace(
            hour=hour, minute=minute, tzinfo=ZoneInfo("Europe/London")
        ).isoformat()
        for label, hour, minute in [("morning", 8, 30), ("evening", 18, 0)]
    }
    source = Path("data/runs/harrow_sidcup_v1")
    origins = pd.read_parquet(source / "origins.parquet")
    destinations = pd.read_parquet(source / "destinations.parquet")
    selected = []
    for name, lat, lng in AREAS:
        distances = (origins.lat - lat) ** 2 + ((origins.lng - lng) * 0.6225) ** 2
        row = origins.loc[distances.idxmin()]
        selected.append(
            {
                "id": str(row.origin_id),
                "name": name,
                "lat": float(row.lat),
                "lng": float(row.lng),
            }
        )
    assert len({row["id"] for row in selected}) == 12
    sampled = pd.concat(
        [
            destinations[destinations.coverage_region == region].sample(
                n=count, random_state=20260908
            )
            for region, count in [("original", 64), ("outer", 32)]
        ]
    )
    arrivals = [str(value) for value in sampled.destination_id]
    locations = [
        {"id": row["id"], "coords": {"lat": row["lat"], "lng": row["lng"]}}
        for row in selected
    ]
    locations += [
        {
            "id": str(row.destination_id),
            "coords": {"lat": float(row.lat), "lng": float(row.lng)},
        }
        for row in sampled.itertuples()
    ]
    plan = {
        "date": args.date,
        "times": times,
        "origins": selected,
        "destinations": locations[12:],
        "search_limit": 24,
        "request_limit": 6,
        "travel_time_seconds": 10800,
    }
    out = Path("artifacts/runs") / f"evening_preliminary_{args.date.replace('-', '')}"
    out.mkdir(parents=True, exist_ok=True)
    plan_file = out / "plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise SystemExit("The existing plan differs; refusing to mix results.")
    plan_file.write_text(json.dumps(plan, indent=2) + "\n")
    if not args.execute:
        print(json.dumps({"plan": str(plan_file), "searches": 24, "route_pairs": 1152}))
        return
    if abs((day.date() - datetime.now(ZoneInfo("Europe/London")).date()).days) > 13:
        raise SystemExit("Choose a date within the timetable window.")
    values = dotenv_values(".env")
    if not values.get("TRAVELTIME_APP_ID") or not values.get("TRAVELTIME_API_KEY"):
        raise SystemExit("TravelTime credentials are missing.")
    headers = {
        "Content-Type": "application/json",
        "X-Application-Id": values["TRAVELTIME_APP_ID"],
        "X-Api-Key": values["TRAVELTIME_API_KEY"],
    }
    rows = []
    last_request = 0.0
    for batch in range(6):
        searches = [
            {
                "id": f"{origin['id']}__{period}",
                "departure_location_id": origin["id"],
                "arrival_location_ids": arrivals,
                "departure_time": timestamp,
                "travel_time": 10800,
                "properties": ["travel_time"],
                "transportation": {"type": "public_transport"},
            }
            for origin in selected[batch * 2 : batch * 2 + 2]
            for period, timestamp in times.items()
        ]
        result_file = out / f"batch_{batch}.json"
        if result_file.exists():
            data = json.loads(result_file.read_text())
        else:
            delay = max(0.0, 60.5 - (time.monotonic() - last_request))
            while delay > 0:
                pause = min(delay, 30.0)
                time.sleep(pause)
                delay -= pause
            last_request = time.monotonic()
            response = requests.post(
                "https://api.traveltimeapp.com/v4/time-filter",
                headers=headers,
                json={"locations": locations, "departure_searches": searches},
                timeout=50,
            )
            if not response.ok:
                # Do not print provider bodies or headers, which may contain credentials.
                raise SystemExit(
                    f"TravelTime returned HTTP {response.status_code}; stopped without retrying."
                )
            data = response.json()
            result_file.write_text(json.dumps(data, indent=2) + "\n")
        results = {result["search_id"]: result for result in data["results"]}
        if set(results) != {search["id"] for search in searches}:
            raise SystemExit("Incomplete response: search IDs differ.")
        for search in searches:
            result = results[search["id"]]
            reached = {
                location["id"]: location["properties"][0]["travel_time"]
                for location in result["locations"]
            }
            unreachable = set(result["unreachable"])
            if set(reached) & unreachable or set(reached) | unreachable != set(
                arrivals
            ):
                raise SystemExit(
                    "Incomplete response: destinations missing or duplicated."
                )
            origin_id, period = search["id"].rsplit("__", 1)
            rows += [
                {
                    "origin_id": origin_id,
                    "destination_id": dest,
                    "period": period,
                    "minutes": reached[dest] / 60 if dest in reached else None,
                }
                for dest in arrivals
            ]
        print(
            f"Completed {batch + 1}/6 batches ({len(rows)} travel times).", flush=True
        )
    long = pd.DataFrame(rows)
    pairs = long.pivot(
        index=["origin_id", "destination_id"], columns="period", values="minutes"
    ).reset_index()
    pairs["difference_minutes"] = pairs.evening - pairs.morning
    pairs.to_csv(out / "paired_routes.csv", index=False)
    both = pairs.dropna(subset=["morning", "evening"])
    absolute = both.difference_minutes.abs()
    summary = {
        "date": args.date,
        "route_pairs": len(pairs),
        "reachable_both": len(both),
        "unreachable_both": int((pairs.morning.isna() & pairs.evening.isna()).sum()),
        "morning_only": int((pairs.morning.notna() & pairs.evening.isna()).sum()),
        "evening_only": int((pairs.morning.isna() & pairs.evening.notna()).sum()),
        "mean_evening_minus_morning_minutes": float(both.difference_minutes.mean()),
        "median_absolute_difference_minutes": float(absolute.median()),
        "p90_absolute_difference_minutes": float(absolute.quantile(0.9)),
        "pct_difference_over_5_minutes": float((absolute > 5).mean() * 100),
        "pct_difference_over_10_minutes": float((absolute > 10).mean() * 100),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
