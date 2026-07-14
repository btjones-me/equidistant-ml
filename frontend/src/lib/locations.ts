import type { Friend, SurfaceCell } from "../types";

function containsPoint(boundary: [number, number][], lat: number, lng: number): boolean {
  let inside = false;
  for (let index = 0, previous = boundary.length - 1; index < boundary.length; previous = index, index += 1) {
    const [currentLat, currentLng] = boundary[index];
    const [previousLat, previousLng] = boundary[previous];
    const crossesLatitude = currentLat > lat !== previousLat > lat;
    if (!crossesLatitude) {
      continue;
    }
    const crossingLng = ((previousLng - currentLng) * (lat - currentLat)) / (previousLat - currentLat) + currentLng;
    if (lng < crossingLng) {
      inside = !inside;
    }
  }
  return inside;
}

function nearestCell(friend: Friend, cells: SurfaceCell[]): SurfaceCell | undefined {
  const containingCell = cells.find(
    (cell) => Array.isArray(cell.boundary) && cell.boundary.length >= 3 && containsPoint(cell.boundary, friend.lat, friend.lng)
  );
  if (containingCell) {
    return containingCell;
  }
  return cells.reduce<SurfaceCell | undefined>((nearest, cell) => {
    if (!nearest) {
      return cell;
    }
    const latitudeScale = Math.cos((friend.lat * Math.PI) / 180);
    const cellDistance = (cell.lat - friend.lat) ** 2 + ((cell.lng - friend.lng) * latitudeScale) ** 2;
    const nearestDistance = (nearest.lat - friend.lat) ** 2 + ((nearest.lng - friend.lng) * latitudeScale) ** 2;
    return cellDistance < nearestDistance ? cell : nearest;
  }, undefined);
}

export function locationLabelForFriend(friend: Friend, cells: SurfaceCell[]): string {
  const cell = nearestCell(friend, cells);
  const cellName = cell?.nearest_station_name;
  if (typeof cellName === "string" && cellName.trim()) {
    return cellName.trim();
  }
  return friend.locationLabel?.trim() || "Set location";
}
