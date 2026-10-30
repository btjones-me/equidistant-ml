import { describe, expect, it } from 'vitest';
import { combineValues } from './atlas';
import { deriveGroupTravel } from './groupTravel';
import { rankPubs, type FiftyPub } from './fiftyPubs';

const origins = [{ id: 'a', name: 'Household A', lat: 51.51, lng: -.1 }, { id: 'b', name: 'Household B', lat: 51.52, lng: -.2 }];
const people = [{ id: 'one', name: 'One', originId: 'a' }, { id: 'two', name: 'Two', originId: 'a' }, { id: 'three', name: 'Three', originId: 'b' }];
describe('shared group travel selection', () => {
  it('routes each origin once and weights every individual once', () => {
    const travel = deriveGroupTravel({ origins, freePeople: people });
    expect(travel.friends).toHaveLength(2);
    expect(travel.includedIndexes).toEqual([0, 0, 1]);
    expect(combineValues(travel.includedIndexes.map(i => [10, 40][i]), 'mean')).toBe(20);
    const smaller = deriveGroupTravel({ origins, freePeople: [people[0], people[2]] });
    expect(combineValues(smaller.includedIndexes.map(i => [10, 40][i]), 'mean')).toBe(25);
  });
  it('makes pub scores and averages person-weighted, not household-weighted', () => {
    const travel = deriveGroupTravel({ origins, freePeople: people });
    const pub = { id: 'pub', rank: 1, lat: 51.51, lng: -.1 } as FiftyPub;
    const cells = [{ destination_id: 'cell', lat: 51.51, lng: -.1, x_index: 0, y_index: 0, friend_0_model_minutes: 10, friend_1_model_minutes: 40 }];
    const ranked = rankPubs([pub], cells, 2, travel.includedIndexes, 'mean');
    expect(ranked[0].score).toBe(20);
    expect(ranked[0].average).toBe(20);
    expect(ranked[0].longest).toBe(40);
  });
  it('does not invent an origin for unlinked or zero attendees', () => {
    const travel = deriveGroupTravel({ origins, freePeople: [{ id: 'new', name: 'New person', originId: null }] });
    expect(travel.friends).toEqual([]);
    expect(travel.includedIndexes).toEqual([]);
    expect(travel.missingLocationNames).toEqual(['New person']);
    expect(travel.availableCount).toBe(1);
    expect(travel.coveredCount).toBe(0);
    expect(deriveGroupTravel({ origins, freePeople: [] }).includedIndexes).toEqual([]);
  });
  it('uses stable IDs rather than matching names and leaves source snapshots untouched', () => {
    const before = JSON.stringify(origins);
    const travel = deriveGroupTravel({ origins, freePeople: [...people, people[0], { id: 'other', name: 'One', originId: 'missing' }] });
    expect(travel.includedIndexes).toEqual([0, 0, 1]);
    expect(travel.missingLocationNames).toEqual(['One']);
    expect(JSON.stringify(origins)).toBe(before);
  });
});
