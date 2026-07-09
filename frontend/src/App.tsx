import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import MapView from "./MapView";
import type { CombineMode, Friend, SurfaceCell, SurfaceResponse } from "./types";

const sampleFriends: Friend[] = [
  { name: "Alex", lat: 51.5154, lng: -0.141 },
  { name: "Sam", lat: 51.5364, lng: -0.075 },
  { name: "Priya", lat: 51.4701, lng: -0.2106 },
  { name: "Maya", lat: 51.548, lng: -0.191 },
  { name: "Tom", lat: 51.5033, lng: -0.1195 },
  { name: "Nina", lat: 51.4628, lng: -0.114 }
];

const combineModes: CombineMode[] = ["balanced", "max", "mean", "fairness"];

function emptyResponse(): SurfaceResponse {
  return { lats: [], lngs: [], Z: [], cells: [] };
}

function formatMinutes(value?: number): string {
  if (value === undefined || !Number.isFinite(value)) {
    return "-";
  }
  return `${value.toFixed(1)} min`;
}

export default function App() {
  const [friendCount, setFriendCount] = useState(3);
  const [friends, setFriends] = useState<Friend[]>(sampleFriends.slice(0, 3));
  const [combine, setCombine] = useState<CombineMode>("balanced");
  const [gridSize, setGridSize] = useState(30);
  const [surface, setSurface] = useState<SurfaceResponse>(emptyResponse);
  const [selectedCell, setSelectedCell] = useState<SurfaceCell | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedScore = selectedCell?.score_minutes ?? selectedCell?.travel_time_minutes;
  const bestCells = useMemo(
    () =>
      surface.cells
        .slice()
        .sort(
          (left, right) =>
            (left.score_minutes ?? Number.POSITIVE_INFINITY) -
            (right.score_minutes ?? Number.POSITIVE_INFINITY)
        )
        .slice(0, 5),
    [surface.cells]
  );

  const requestSurface = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/group-surface", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          friends,
          combine,
          x_size: gridSize,
          y_size: gridSize
        })
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const body = (await response.json()) as SurfaceResponse;
      setSurface(body);
      setSelectedCell(
        body.cells
          .slice()
          .sort(
            (left, right) =>
              (left.score_minutes ?? Number.POSITIVE_INFINITY) -
              (right.score_minutes ?? Number.POSITIVE_INFINITY)
          )[0] ?? null
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Surface request failed");
    } finally {
      setLoading(false);
    }
  }, [combine, friends, gridSize]);

  useEffect(() => {
    requestSurface();
  }, [requestSurface]);

  function updateFriend(index: number, field: keyof Friend, value: string) {
    setFriends((current) => {
      const next = current.slice();
      const parsed = field === "name" ? value : Number(value);
      next[index] = { ...next[index], [field]: parsed };
      return next;
    });
  }

  function changeFriendCount(value: number) {
    const nextCount = Math.max(2, Math.min(6, value));
    setFriendCount(nextCount);
    setFriends((current) => {
      if (nextCount <= current.length) {
        return current.slice(0, nextCount);
      }
      return [...current, ...sampleFriends.slice(current.length, nextCount)];
    });
  }

  return (
    <main className="app-shell">
      <section className="control-panel" aria-label="Surface controls">
        <div className="brand-row">
          <div>
            <p className="eyebrow">Equidistant</p>
            <h1>Travel-time surfaces</h1>
          </div>
          <button className="icon-button" type="button" onClick={requestSurface} disabled={loading} title="Refresh surface">
            <RefreshCw size={18} aria-hidden="true" />
          </button>
        </div>

        <div className="field-row">
          <label>
            Friends
            <input
              type="number"
              min={2}
              max={6}
              value={friendCount}
              onChange={(event) => changeFriendCount(Number(event.target.value))}
            />
          </label>
          <label>
            Grid
            <input
              type="number"
              min={10}
              max={80}
              step={5}
              value={gridSize}
              onChange={(event) => setGridSize(Number(event.target.value))}
            />
          </label>
        </div>

        <div className="segmented" aria-label="Combine mode">
          {combineModes.map((mode) => (
            <button
              key={mode}
              className={combine === mode ? "active" : ""}
              type="button"
              onClick={() => setCombine(mode)}
            >
              {mode}
            </button>
          ))}
        </div>

        <div className="friends-list">
          {friends.map((friend, index) => (
            <div className="friend-editor" key={`${friend.name}-${index}`}>
              <input
                aria-label={`Friend ${index + 1} name`}
                value={friend.name}
                onChange={(event) => updateFriend(index, "name", event.target.value)}
              />
              <input
                aria-label={`Friend ${index + 1} latitude`}
                type="number"
                step="0.0001"
                value={friend.lat}
                onChange={(event) => updateFriend(index, "lat", event.target.value)}
              />
              <input
                aria-label={`Friend ${index + 1} longitude`}
                type="number"
                step="0.0001"
                value={friend.lng}
                onChange={(event) => updateFriend(index, "lng", event.target.value)}
              />
            </div>
          ))}
        </div>

        {error ? <div className="error-banner">{error}</div> : null}

        <section className="summary-panel" aria-label="Selected destination">
          <div>
            <p className="eyebrow">Selected cell</p>
            <h2>{formatMinutes(selectedScore)}</h2>
          </div>
          <dl>
            <div>
              <dt>Lat</dt>
              <dd>{selectedCell?.lat.toFixed(5) ?? "-"}</dd>
            </div>
            <div>
              <dt>Lng</dt>
              <dd>{selectedCell?.lng.toFixed(5) ?? "-"}</dd>
            </div>
            <div>
              <dt>Fairness</dt>
              <dd>{selectedCell?.fairness_seconds ? `${(selectedCell.fairness_seconds / 60).toFixed(1)} min` : "-"}</dd>
            </div>
          </dl>
        </section>

        <section className="ranked-list" aria-label="Best cells">
          <p className="eyebrow">Lowest scores</p>
          {bestCells.map((cell) => (
            <button
              key={cell.destination_id}
              type="button"
              className={selectedCell?.destination_id === cell.destination_id ? "ranked-item active" : "ranked-item"}
              onClick={() => setSelectedCell(cell)}
            >
              <span>{formatMinutes(cell.score_minutes)}</span>
              <span>{cell.lat.toFixed(3)}, {cell.lng.toFixed(3)}</span>
            </button>
          ))}
        </section>
      </section>

      <section className="visual-panel" aria-label="Travel-time map">
        <MapView friends={friends} cells={surface.cells} selectedCell={selectedCell} onSelectCell={setSelectedCell} />
      </section>
    </main>
  );
}
