import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { useAppState } from './state/AppStateContext';
import { activeAccountId } from './lib/account';

vi.mock('./ProductMode', () => ({ default: ({ accountEmail, fiftyPubsMode }: { accountEmail: string; fiftyPubsMode: boolean }) => {
  const { friends } = useAppState();
  return <div><p>{accountEmail}</p><p>{fiftyPubsMode ? 'Pub collection workspace' : 'Normal workspace'}</p>
    {friends.map(friend => <p key={friend.id}>{friend.name}</p>)}</div>;
} }));
vi.mock('./DeveloperMode', () => ({ default: () => <p>Restricted diagnostics</p> }));
window.history.replaceState({}, '', '/50pubs');
const { default: App } = await import('./App');
const session = (id: string) => ({ user: { id, email: `${id}@example.test`, name: id }, permissions: { debug: false, liveTravelTime: false, unlimitedRecommendations: false } });
const preset = (label: string) => ({ friends: [{ id: 'example-pair', name: label, lat: 51.53, lng: -0.13, locationLabel: 'Example station' }] });
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(done => { resolve = done; });
  return { promise, resolve };
}
beforeEach(() => { window.localStorage.clear(); window.history.replaceState({}, '', '/50pubs'); });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it('waits for a verified session before requesting an account-bound preset and rendering its participants', async () => {
  const sessionResponse = deferred<Response>(), presetResponse = deferred<Response>();
  const fetch = vi.fn((url: string) => url === '/api/session' ? sessionResponse.promise : presetResponse.promise);
  vi.stubGlobal('fetch', fetch);
  render(<App />);
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(fetch).toHaveBeenNthCalledWith(1, '/api/session', { cache: 'no-store' });
  expect(screen.queryByText('Private first pair')).not.toBeInTheDocument();
  sessionResponse.resolve(Response.json(session('owner')));
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
  expect(fetch).toHaveBeenNthCalledWith(2, '/api/50pubs/preset', { cache: 'no-store', headers: { 'X-Equidistant-Account': 'owner' } });
  expect(screen.queryByText('owner@example.test')).not.toBeInTheDocument();
  presetResponse.resolve(Response.json(preset('Private first pair')));
  expect(await screen.findByText('Private first pair')).toBeInTheDocument();
  expect(screen.getByText('Pub collection workspace')).toBeInTheDocument();
  expect(activeAccountId()).toBe('owner');
});

it('discards a previous account’s in-flight preset after focus starts a new session refresh', async () => {
  const ownerPreset = deferred<Response>();
  let currentAccount = 'owner';
  const fetch = vi.fn((url: string, init?: RequestInit) => {
    if (url === '/api/session') return Promise.resolve(Response.json(session(currentAccount)));
    const account = new Headers(init?.headers).get('X-Equidistant-Account');
    return account === 'owner' ? ownerPreset.promise : Promise.resolve(Response.json(preset('Different account pair')));
  });
  vi.stubGlobal('fetch', fetch);
  render(<App />);
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
  currentAccount = 'different';
  fireEvent.focus(window);
  expect(await screen.findByText('Different account pair')).toBeInTheDocument();
  await act(async () => {
    ownerPreset.resolve(Response.json(preset('Private owner pair')));
    await ownerPreset.promise;
  });
  expect(activeAccountId()).toBe('different');
  expect(screen.queryByText('Private owner pair')).not.toBeInTheDocument();
  expect(screen.queryByText('owner@example.test')).not.toBeInTheDocument();
  expect(window.localStorage.getItem('equidistant:workspace:v3:different:50pubs')).not.toContain('Private owner pair');
});

it('hides an existing owner workspace immediately on focus and does not reuse its preset for another account', async () => {
  const nextSession = deferred<Response>();
  let changed = false;
  const fetch = vi.fn((url: string) => {
    if (url === '/api/session') return changed ? nextSession.promise : Promise.resolve(Response.json(session('owner')));
    return Promise.resolve(Response.json(changed ? { friends: [] } : preset('Private owner pair')));
  });
  vi.stubGlobal('fetch', fetch);
  render(<App />);
  await screen.findByText('Private owner pair');
  changed = true;
  fireEvent.focus(window);
  expect(screen.queryByText('Private owner pair')).not.toBeInTheDocument();
  expect(activeAccountId()).toBe('');
  nextSession.resolve(Response.json(session('ordinary')));
  await screen.findByText('ordinary@example.test');
  expect(screen.queryByText('Private owner pair')).not.toBeInTheDocument();
  expect(activeAccountId()).toBe('ordinary');
});

it('fails closed if account changes between session verification and the preset request', async () => {
  vi.stubGlobal('fetch', vi.fn((url: string) => Promise.resolve(url === '/api/session'
    ? Response.json(session('owner')) : Response.json({ detail: 'Your signed-in account changed.' }, { status: 409 }))));
  render(<App />);
  expect(await screen.findByRole('alert')).toHaveTextContent('We couldn’t check your sign-in. Please reload to try again.');
  expect(screen.queryByText('owner@example.test')).not.toBeInTheDocument();
  expect(screen.queryByText('Pub collection workspace')).not.toBeInTheDocument();
  expect(activeAccountId()).toBe('');
});
