import { describe, expect, it } from 'vitest';
import { asMapVenue, distanceMetres, rankPubs, validatePubDataset, type FiftyPub } from './fiftyPubs';
import type { CoverageBounds, SurfaceCell } from '../types';

const pub = (id: string, rank: number, lat = 51.51, lng = -0.12): FiftyPub => ({
  id, rank, name: `Pub ${id}`, neighbourhood: 'Example district', address: `${rank} Example Road, London`, postcode: 'EX1 1AA', lat, lng,
  summary: 'Original test summary.', sourceUrl: 'https://editor.example/list', addressSourceUrl: 'https://venue.example/contact',
  coordinateSourceUrl: 'https://maps.example/place', images: [], tags: ['Historic']
});
const cell = (id: string, lat: number, lng: number, values: (number | undefined)[]): SurfaceCell => ({
  destination_id: id, lat, lng, x_index: 0, y_index: 0,
  ...Object.fromEntries(values.map((value, i) => [`friend_${i}_model_minutes`, value]))
});
const bounds: CoverageBounds = { south: 51.4, north: 51.6, west: -0.3, east: 0 };

describe('50 pub travel ranking', () => {
  it('uses only included participants for group scoring but retains individual journey estimates', () => {
    const pubs = [pub('a', 1), pub('b', 2, 51.55)];
    const cells = [cell('a', 51.51, -0.12, [10, 80]), cell('b', 51.55, -0.12, [25, 25])];
    expect(rankPubs(pubs, cells, 2, [0, 1], 'balanced')[0].id).toBe('b');
    const ranked = rankPubs(pubs, cells, 2, [0], 'balanced');
    expect(ranked[0]).toMatchObject({ id: 'a', journeys: [10, 80], score: 10, average: 10, longest: 10 });
  });

  it('changes the recommendation when the group strategy changes', () => {
    const pubs = [pub('short-average', 2), pub('short-longest', 1, 51.55)];
    const cells = [cell('a', 51.51, -0.12, [5, 35]), cell('b', 51.55, -0.12, [25, 25])];
    expect(rankPubs(pubs, cells, 2, [0, 1], 'mean')[0].id).toBe('short-average');
    expect(rankPubs(pubs, cells, 2, [0, 1], 'max')[0].id).toBe('short-longest');
    expect(rankPubs(pubs, cells, 2, [0, 1], 'balanced')[0].id).toBe('short-longest');
    expect(rankPubs(pubs, cells, 2, [0, 1], 'fairness')[0].id).toBe('short-longest');
  });

  it('breaks equal travel scores by editorial rank without mutating the input', () => {
    const pubs = [pub('later', 42), pub('earlier', 3)];
    const result = rankPubs(pubs, [cell('a', 51.51, -0.12, [20])], 1, [0], 'mean');
    expect(result.map(p => p.rank)).toEqual([3, 42]);
    expect(pubs.map(p => p.rank)).toEqual([42, 3]);
  });

  it('never extrapolates a venue outside declared model bounds even from a nearby edge cell', () => {
    const outside = pub('outside', 1, 51.6001, -0.12);
    const ranked = rankPubs([outside], [cell('edge', 51.6, -0.12, [15, 20])], 2, [0, 1], 'balanced', bounds);
    expect(ranked[0]).toMatchObject({ journeys: [null, null], score: null, average: null, longest: null, sampledDistanceMetres: null });
    expect(ranked[0].lat).toBe(outside.lat);
  });

  it('marks remote venues and empty model surfaces uncovered rather than inventing times', () => {
    for (const cells of [[], [cell('remote', 51.55, -0.12, [12])]]) {
      const [ranked] = rankPubs([pub('a', 1)], cells, 1, [0], 'mean', bounds);
      expect(ranked.score).toBeNull();
      expect(ranked.journeys).toEqual([null]);
    }
  });

  it('does not blend distant cells into an otherwise covered local venue', () => {
    const nearby = cell('near', 51.5105, -0.12, [12]);
    const distant = cell('far', 51.57, -0.12, [120]);
    const [ranked] = rankPubs([pub('a', 1)], [nearby, distant], 1, [0], 'mean', bounds);
    expect(ranked.journeys[0]).toBeCloseTo(12, 8);
  });

  it('requires finite estimates for included participants, not excluded participants', () => {
    const cells = [cell('a', 51.51, -0.12, [10, undefined, Number.NaN])];
    expect(rankPubs([pub('a', 1)], cells, 3, [0], 'mean')[0].score).toBe(10);
    expect(rankPubs([pub('a', 1)], cells, 3, [0, 1], 'mean')[0].score).toBeNull();
    expect(rankPubs([pub('a', 1)], cells, 3, [0, 2], 'mean')[0].score).toBeNull();
    expect(rankPubs([pub('a', 1)], cells, 3, [], 'mean')[0].score).toBeNull();
  });

  it('preserves exact verified venue coordinates and editorial marker rank', () => {
    const original = pub('specific-venue', 47, 51.512345, -0.123456);
    const venue = asMapVenue(original);
    expect(venue).toMatchObject({ place_id: original.id, name: '#47 · Pub specific-venue', lat: original.lat, lng: original.lng, marker_label: 47 });
    expect(venue.google_maps_url).toContain('51.512345,-0.123456');
    expect(venue.photo_url).toBeNull();
    expect(distanceMetres(original, original)).toBe(0);
  });
});

describe('50 pub dataset validation', () => {
  const collection = () => ({ article: { title: 'Editorial list', url: 'https://editor.example/list', checkedAt: '2026-10-30' }, pubs: Array.from({ length: 50 }, (_, i) => pub(`pub-${i + 1}`, i + 1)) });
  it('accepts exactly fifty distinct namespaced venues and complete editorial ranks', () => {
    const dataset = collection();
    expect(validatePubDataset(dataset)).toBe(dataset);
  });
  it('rejects missing venues, duplicate ranks or IDs, and invalid coordinates', () => {
    const short = collection(); short.pubs.pop();
    expect(() => validatePubDataset(short)).toThrow();
    for (const mutation of [{ rank: 1 }, { id: 'pub-1' }, { lat: Number.NaN }, { lng: 5 }, { rank: 51 }]) {
      const data = collection(); Object.assign(data.pubs[1], mutation);
      expect(() => validatePubDataset(data)).toThrow();
    }
  });
});
