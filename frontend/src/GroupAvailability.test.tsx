import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useState } from 'react';
import GroupAvailability from './GroupAvailability';
import { groupRequest, londonToday, dateLabel, PENDING_INVITE_KEY, type GroupMonth, type GroupTravelSelection } from './lib/groupAvailability';
vi.mock('./lib/groupAvailability', async () => ({ ...await vi.importActual('./lib/groupAvailability'), groupRequest: vi.fn() }));
const month = londonToday().slice(0, 7), date = `${month}-01`;
const groupId = `grp_${'a'.repeat(32)}`;
const friends = [{ id: 'original', name: 'A & B', locationLabel: 'Station', lat: 51.5, lng: -0.1 }];
const base: GroupMonth = { group: { id: groupId, name: 'Our group', createdAt: 1, memberCount: 2, origins: [{ id: 'origin', name: 'Household', locationLabel: 'Station', lat: 51.5, lng: -0.1 }], people: [{ id: 'person-a', name: 'A', originId: 'origin' }, { id: 'person-b', name: 'B', originId: 'origin' }] }, month, availability: [], serverTime: 1 };
let snapshot: GroupMonth;
const mock = vi.mocked(groupRequest);
const select = vi.fn();
function Harness() {
  const [active, setActive] = useState<GroupTravelSelection | null>(null);
  return <><button onClick={() => setActive(null)}>Clear map date</button><GroupAvailability friends={friends} accountId="account" activeSelection={active} onSelection={value => { select(value); setActive(value); }} /></>;
}
beforeEach(() => {
  snapshot = structuredClone(base); mock.mockReset(); select.mockReset(); sessionStorage.clear();
  mock.mockImplementation(async (url, init) => {
    if (url === '/api/groups' && !init?.method) return { groups: [{ id: groupId, name: 'Our group', memberCount: 2, createdAt: 1 }] } as never;
    if (url.includes('?month=')) return structuredClone(snapshot) as never;
    if (url.endsWith('/availability')) { const body = JSON.parse(init!.body as string); const cell = { ...body, updatedAt: 2 }; snapshot.availability = [cell]; return cell as never; }
    if (url === '/api/groups' && init?.method === 'POST') return { groupId } as never;
    if (url.endsWith('/join')) return { groupId } as never;
    return {} as never;
  });
});
afterEach(() => { cleanup(); vi.useRealTimers(); });
async function open() { fireEvent.click(screen.getByRole('button', { name: /Group availability/ })); await screen.findByRole('button', { name: 'A' }); }
describe('shared group panel', () => {
  it('loads a monthly grid and saves only the clicked cell', async () => {
    render(<Harness />); await open();
    fireEvent.click(screen.getByRole('button', { name: `A, ${dateLabel(date)}: unanswered. Set free` }));
    await screen.findByRole('button', { name: `A, ${dateLabel(date)}: free. Set unavailable` });
    expect(mock).toHaveBeenCalledWith(`/api/groups/${groupId}/availability`, expect.objectContaining({ method: 'PATCH', body: JSON.stringify({ personId: 'person-a', date, status: 'free' }) }));
    expect(screen.getByRole('button', { name: `B, ${dateLabel(date)}: unanswered. Set free` })).toBeInTheDocument();
    expect(screen.getByText('Saved')).toBeInTheDocument();
  });
  it('reviews and confirms split people before creating a snapshot', async () => {
    render(<Harness />); await open(); fireEvent.click(screen.getByRole('button', { name: 'Start a group session' }));
    const inputs = screen.getAllByRole('textbox', { name: 'Person at A & B' });
    expect(inputs.map(input => (input as HTMLInputElement).value)).toEqual(['A', 'B']);
    fireEvent.change(inputs[1], { target: { value: 'Bee' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create shared group' }));
    await waitFor(() => expect(mock.mock.calls.some(([url, init]) => url === '/api/groups' && init?.method === 'POST')).toBe(true));
    const body = JSON.parse(mock.mock.calls.find(([url, init]) => url === '/api/groups' && init?.method === 'POST')![1]!.body as string);
    expect(body.people.map((person: {name:string}) => person.name)).toEqual(['A', 'Bee']);
    expect(body.origins[0].locationLabel).toBe('Station');
    expect(body.origins[0].id).not.toBe('original');
  });
  it('applies available people only and closing does not discard the selection', async () => {
    snapshot.availability = [{ personId: 'person-a', date, status: 'free', updatedAt: 1 }];
    render(<Harness />); await open(); fireEvent.click(screen.getByRole('button', { name: `Select ${dateLabel(date)}` }));
    fireEvent.click(screen.getByRole('button', { name: 'Find pubs for 1 person' }));
    expect(select).toHaveBeenLastCalledWith(expect.objectContaining({ date, freePeople: [base.group.people[0]] }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });
  it('blocks pub recommendations when a free person has no linked location', async () => {
    snapshot.group.people[0].originId = null; snapshot.availability = [{ personId: 'person-a', date, status: 'free', updatedAt: 1 }];
    render(<Harness />); await open(); fireEvent.click(screen.getByRole('button', { name: `Select ${dateLabel(date)}` }));
    expect(screen.getByRole('button', { name: 'Find pubs for 1 person' })).toBeDisabled();
    expect(screen.getByText('Link a starting location for A before finding pubs.')).toBeInTheDocument();
  });
  it('opens an invitation without joining until explicitly requested', async () => {
    sessionStorage.setItem(PENDING_INVITE_KEY, JSON.stringify({ token: 'a'.repeat(64), expiresAt: Date.now() + 60000 }));
    render(<Harness />); await screen.findByRole('button', { name: 'Join this group' });
    expect(mock.mock.calls.some(([url]) => url.endsWith('/join'))).toBe(false);
    fireEvent.click(screen.getByRole('button', { name: 'Join this group' }));
    await waitFor(() => expect(sessionStorage.getItem(PENDING_INVITE_KEY)).toBeNull());
    expect(mock).toHaveBeenCalledWith('/api/groups/join', expect.objectContaining({ method: 'POST' }));
  });
  it('returns focus to the launch button on Escape', async () => {
    render(<Harness />); await open(); fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Group availability/ })).toHaveFocus();
  });
  it('does not restore a cleared parent map selection after refreshing the panel', async () => {
    snapshot.availability = [{ personId: 'person-a', date, status: 'free', updatedAt: 1 }];
    render(<Harness />); await open(); fireEvent.click(screen.getByRole('button', { name: `Select ${dateLabel(date)}` }));
    fireEvent.click(screen.getByRole('button', { name: 'Find pubs for 1 person' }));
    fireEvent.click(screen.getByRole('button', { name: 'Clear map date' })); select.mockClear();
    await open(); await act(async () => { await Promise.resolve(); });
    expect(select).not.toHaveBeenCalled();
  });
  it('keeps an unanswered cell on a failed save and shows an error', async () => {
    const original = mock.getMockImplementation()!;
    mock.mockImplementation((url, init) => url.endsWith('/availability') ? Promise.reject(new Error('Connection lost')) : original(url, init));
    render(<Harness />); await open(); fireEvent.click(screen.getByRole('button', { name: `A, ${dateLabel(date)}: unanswered. Set free` }));
    await screen.findByText('Connection lost');
    expect(screen.getByRole('button', { name: `A, ${dateLabel(date)}: unanswered. Set free` })).toBeEnabled();
  });
  it('polls applied attendance while the dialog is closed and retains an empty selection', async () => {
    snapshot.availability = [{ personId: 'person-a', date, status: 'free', updatedAt: 1 }];
    render(<Harness />); await open();
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole('button', { name: `Select ${dateLabel(date)}` }));
    fireEvent.click(screen.getByRole('button', { name: 'Find pubs for 1 person' }));
    await act(async () => { await Promise.resolve(); });
    snapshot.availability = [];
    await act(async () => { await vi.advanceTimersByTimeAsync(10_001); });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(select).toHaveBeenLastCalledWith(expect.objectContaining({ groupId, date, freePeople: [] }));
  });
  it('sends only the edited field when renaming a person', async () => {
    render(<Harness />); await open();
    fireEvent.click(screen.getByRole('button', { name: 'A' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Name' }), { target: { value: 'Alice' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save person' }));
    await waitFor(() => expect(mock).toHaveBeenCalledWith(`/api/groups/${groupId}/people/person-a`, expect.objectContaining({ method: 'PATCH', body: JSON.stringify({ name: 'Alice' }) })));
  });
  it('exposes a failed background refresh without reverting to personal results', async () => {
    snapshot.availability = [{ personId: 'person-a', date, status: 'free', updatedAt: 1 }];
    render(<Harness />); await open(); vi.useFakeTimers();
    fireEvent.click(screen.getByRole('button', { name: `Select ${dateLabel(date)}` }));
    fireEvent.click(screen.getByRole('button', { name: 'Find pubs for 1 person' }));
    await act(async () => { await Promise.resolve(); });
    const original = mock.getMockImplementation()!;
    mock.mockImplementation((url, init) => url.includes('?month=') ? Promise.reject(new Error('Unable to refresh group')) : original(url, init));
    await act(async () => { await vi.advanceTimersByTimeAsync(10_001); });
    expect(select).toHaveBeenLastCalledWith(expect.objectContaining({ groupId, freePeople: [], error: 'Unable to refresh group' }));
  });

});
