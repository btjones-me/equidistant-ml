import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { GroupTravelSelection } from './lib/groupAvailability';
import type { SurfaceResponse } from './types';
import { AppStateProvider, workspaceStorageKey } from './state/AppStateContext';
import ProductMode from './ProductMode';

const mocks = vi.hoisted(() => ({ surface: vi.fn() }));
vi.mock('./lib/atlas', async () => ({ ...await vi.importActual('./lib/atlas'), getAtlasSurface: mocks.surface, preloadAtlas: () => Promise.resolve() }));
vi.mock('./MapView', () => ({ default: (props: { friends: unknown; includedFriendIndexes: number[]; cells: unknown; onMoveFriend?: unknown }) => <output data-testid="map">{JSON.stringify({ friends: props.friends, indexes: props.includedFriendIndexes, cells: props.cells, draggable: !!props.onMoveFriend })}</output> }));
vi.mock('./VenueRecommendations', () => ({ default: () => null }));
vi.mock('./lib/fiftyPubs', async () => ({ ...await vi.importActual('./lib/fiftyPubs'), validatePubDataset: (value: unknown) => value }));
vi.mock('./FiftyPubs', () => ({
  FiftyPubsRecommendations: ({ pubs, error }: { pubs: Array<{ average: number | null }>; error: string | null }) => <div><output data-testid="scores">{JSON.stringify(pubs.map(pub => pub.average))}</output>{error && <p role="alert">{error}</p>}</div>,
  FiftyPubsDrawer: () => null
}));
vi.mock('./GroupAvailability', () => ({ default: ({ onSelection }: { onSelection: (value: GroupTravelSelection) => void }) => <div>
  <button onClick={() => onSelection(selection)}>Select shared date</button>
  <button onClick={() => onSelection({ ...selection, freePeople: [] })}>Select empty date</button>
  <button onClick={() => onSelection({ ...selection, freePeople: [...selection.freePeople, { id: 'unlinked', name: 'No location', originId: null }] })}>Select unlinked date</button>
</div> }));

const personalFriends = [{ id: 'personal', name: 'Personal starting point', lat: 51.54, lng: -.14, locationLabel: 'Personal station' }];
const origins = [{ id: 'shared-a', name: 'Shared A', lat: 51.51, lng: -.1, locationLabel: 'Shared station A' }, { id: 'shared-b', name: 'Shared B', lat: 51.52, lng: -.2, locationLabel: 'Shared station B' }];
const people = [{ id: 'one', name: 'Person one', originId: 'shared-a' }, { id: 'two', name: 'Person two', originId: 'shared-a' }, { id: 'three', name: 'Person three', originId: 'shared-b' }];
const selection: GroupTravelSelection = { groupId: 'group', groupName: 'Shared group', date: '2026-11-05', origins, people, freePeople: people };
const response = (id: string): SurfaceResponse => ({ lats: [51.51], lngs: [-.1], Z: [[20]], cells: [{ destination_id: id, lat: 51.51, lng: -.1, x_index: 0, y_index: 0, model_score_minutes: 20, mean_minutes: 20, friend_0_model_minutes: 10, friend_1_model_minutes: 40 }] });
const map = () => JSON.parse(screen.getByTestId('map').textContent || '{}');
const persistedFriends = () => JSON.parse(localStorage.getItem(workspaceStorageKey('test-account', '50pubs')) || '{}').friends;
function mount() { return render(<AppStateProvider accountId="test-account" workspace="50pubs" initialFriends={personalFriends}><ProductMode fiftyPubsMode accountEmail="example@example.test" onSignOut={() => {}} /></AppStateProvider>); }
function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>(done => { resolve = done; }); return { promise, resolve }; }
beforeEach(() => {
  localStorage.clear();
  mocks.surface.mockReset().mockResolvedValue(response('current'));
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json({ article: { title: 'Fixture', url: 'https://example.test' }, pubs: [{ id: 'pub', rank: 1, name: 'Fixture pub', lat: 51.51, lng: -.1, images: [] }] })));
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it('uses independent shared origins and person weights, then restores the untouched personal planner', async () => {
  mount();
  await waitFor(() => expect(mocks.surface).toHaveBeenCalledTimes(1));
  expect(map().friends).toEqual(personalFriends);
  fireEvent.click(screen.getByRole('button', { name: 'Select shared date' }));
  await waitFor(() => expect(mocks.surface).toHaveBeenCalledTimes(2));
  expect(mocks.surface.mock.calls[1][0]).toMatchObject({ friends: [{ ...origins[0], name: 'Person one & Person two' }, { ...origins[1], name: 'Person three' }], includedFriendIndexes: [0, 0, 1] });
  await waitFor(() => expect(screen.getByTestId('scores')).toHaveTextContent('[20]'));
  expect(map().indexes).toEqual([0, 0, 1]);
  expect(map().draggable).toBe(false);
  expect(persistedFriends()).toEqual(personalFriends);
  fireEvent.click(screen.getByRole('button', { name: 'Return to personal planner' }));
  await waitFor(() => expect(mocks.surface).toHaveBeenCalledTimes(3));
  expect(map().friends).toEqual(personalFriends);
  expect(map().indexes).toEqual([0]);
  expect(map().draggable).toBe(true);
  expect(persistedFriends()).toEqual(personalFriends);
});

it.each(['Select empty date', 'Select unlinked date'])('clears old map and pub scores for %s without requesting a misleading replacement', async name => {
  mount();
  await waitFor(() => expect(map().cells).toHaveLength(1));
  expect(mocks.surface).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name }));
  expect(map().cells).toEqual([]);
  expect(screen.getByTestId('scores')).toHaveTextContent('[null]');
  expect(screen.getByRole('alert')).toHaveTextContent('link all their starting locations');
  await act(async () => { await new Promise(resolve => setTimeout(resolve, 120)); });
  expect(mocks.surface).toHaveBeenCalledTimes(1);
});

it('rejects a late personal atlas response after a newer shared selection has rendered', async () => {
  const old = deferred<SurfaceResponse>();
  mocks.surface.mockReturnValueOnce(old.promise).mockResolvedValueOnce(response('new-shared'));
  mount();
  await waitFor(() => expect(mocks.surface).toHaveBeenCalledTimes(1));
  fireEvent.click(screen.getByRole('button', { name: 'Select shared date' }));
  await waitFor(() => expect(map().cells[0]?.destination_id).toBe('new-shared'));
  await act(async () => { old.resolve(response('old-personal')); await old.promise; });
  expect(map().cells[0]?.destination_id).toBe('new-shared');
  expect(map().indexes).toEqual([0, 0, 1]);
});
