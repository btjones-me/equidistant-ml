import type { Friend } from '../types';
import { accountFetch } from './account';

export type AvailabilityStatus = 'unknown' | 'free' | 'unavailable';
export type GroupOrigin = { id: string; name: string; locationLabel: string; lat: number; lng: number };
export type GroupPerson = { id: string; name: string; originId: string | null };
export type AvailabilityCell = { personId: string; date: string; status: AvailabilityStatus; updatedAt: number };
export type AvailabilityGroup = { id: string; name: string; createdAt: number; memberCount: number; origins: GroupOrigin[]; people: GroupPerson[] };
export type GroupSummary = Pick<AvailabilityGroup, 'id' | 'name' | 'createdAt' | 'memberCount'>;
export type GroupMonth = { group: AvailabilityGroup; month: string; availability: AvailabilityCell[]; serverTime: number };
export type GroupTravelSelection = { error?: string; groupId: string; groupName: string; date: string; origins: GroupOrigin[]; people: GroupPerson[]; freePeople: GroupPerson[] };
export type SavedGroupSelection = { groupId: string | null; month: string; date: string | null };
export type GroupDraft = { name: string; origins: GroupOrigin[]; people: GroupPerson[] };

export const PENDING_INVITE_KEY = 'equidistant:pending-group-invite';
export const groupSelectionKey = (accountId: string) => `equidistant:group-selection:v1:${accountId}`;
export const cellKey = (personId: string, date: string) => `${personId}:${date}`;
const validId = (value: unknown): value is string => typeof value === 'string' && /^[A-Za-z0-9_-]{1,80}$/.test(value);
export const isMonth = (value: unknown): value is string => typeof value === 'string' && /^20\d{2}-(0[1-9]|1[0-2])$/.test(value);
export function londonToday(now = new Date()): string {
  const parts = new Intl.DateTimeFormat('en-GB', { timeZone: 'Europe/London', year: 'numeric', month: '2-digit', day: '2-digit' }).formatToParts(now);
  const part = (type: string) => parts.find(p => p.type === type)?.value;
  return `${part('year')}-${part('month')}-${part('day')}`;
}
export function monthDates(month: string): string[] {
  if (!isMonth(month)) return [];
  const [year, number] = month.split('-').map(Number);
  return Array.from({ length: new Date(Date.UTC(year, number, 0)).getUTCDate() }, (_, index) => `${month}-${String(index + 1).padStart(2, '0')}`);
}
export function isDate(value: unknown): value is string {
  return typeof value === 'string' && /^20\d{2}-\d{2}-\d{2}$/.test(value) && monthDates(value.slice(0, 7)).includes(value);
}
export function shiftMonth(month: string, direction: number): string {
  const [year, number] = month.split('-').map(Number);
  const date = new Date(Date.UTC(year, number - 1 + direction, 1));
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, '0')}`;
}
export const dateLabel = (date: string, options: Intl.DateTimeFormatOptions = { weekday: 'short', day: 'numeric', month: 'short' }) => new Intl.DateTimeFormat('en-GB', { ...options, timeZone: 'UTC' }).format(new Date(`${date}T12:00:00Z`));
export const monthLabel = (month: string) => dateLabel(`${month}-01`, { month: 'long', year: 'numeric' });
export function suggestedPeople(name: string): string[] {
  const names = name.split(/\s*(?:&|\+)\s*|\s+and\s+/i).map(value => value.trim()).filter(Boolean);
  return names.length === 2 ? names : [name.trim() || 'Person'];
}
export function makeGroupDraft(friends: Friend[]): GroupDraft {
  const origins = friends.slice(0, 6).map(friend => ({ id: crypto.randomUUID(), name: friend.name.trim() || 'Household', locationLabel: friend.locationLabel?.trim() || 'Map location', lat: friend.lat, lng: friend.lng }));
  return { name: 'Our pub plans', origins, people: origins.flatMap(origin => suggestedPeople(origin.name).map(name => ({ id: crypto.randomUUID(), name, originId: origin.id }))) };
}
export function statusMap(cells: AvailabilityCell[]): Record<string, AvailabilityStatus> {
  return Object.fromEntries(cells.map(cell => [cellKey(cell.personId, cell.date), cell.status]));
}
export function attendanceForDate(people: GroupPerson[], date: string, statuses: Record<string, AvailabilityStatus>) {
  const freePeople = people.filter(person => statuses[cellKey(person.id, date)] === 'free');
  const unavailable = people.filter(person => statuses[cellKey(person.id, date)] === 'unavailable').length;
  return { freePeople, free: freePeople.length, unavailable, unknown: people.length - freePeople.length - unavailable, everyoneFree: people.length > 0 && freePeople.length === people.length };
}
export function nextStatus(status: AvailabilityStatus): AvailabilityStatus {
  return status === 'unknown' ? 'free' : status === 'free' ? 'unavailable' : 'unknown';
}
export function readGroupSelection(accountId: string): SavedGroupSelection {
  const fallback = { groupId: null, month: londonToday().slice(0, 7), date: null };
  try {
    const value = JSON.parse(localStorage.getItem(groupSelectionKey(accountId)) || 'null');
    if (!value || !validId(value.groupId) || !isMonth(value.month)) return fallback;
    return { groupId: value.groupId, month: value.month, date: isDate(value.date) ? value.date : null };
  } catch { return fallback; }
}
export function readPendingInvite(now = Date.now()): string | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(PENDING_INVITE_KEY) || 'null');
    if (typeof value?.token === 'string' && /^[a-f0-9]{64}$/.test(value.token) && Number.isFinite(value.expiresAt) && value.expiresAt > now && value.expiresAt <= now + 10 * 60_000) return value.token;
    sessionStorage.removeItem(PENDING_INVITE_KEY);
  } catch { /* A blocked or corrupted browser store must not break the planner. */ }
  return null;
}
export class GroupApiError extends Error {
  status: number;
  constructor(message: string, status: number) { super(message); this.status = status; }
}
export async function groupRequest<T>(url: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body) headers.set('Content-Type', 'application/json');
  const response = await accountFetch(url, { ...init, headers, cache: 'no-store' });
  const value = await response.json().catch(() => null);
  if (!response.ok) throw new GroupApiError(typeof value?.detail === 'string' ? value.detail : 'Could not save your group. Please try again.', response.status);
  if (value === null) throw new GroupApiError('The group response could not be read. Please try again.', 502);
  return value as T;
}
export function validateGroupMonth(value: GroupMonth, expectedGroup: string, expectedMonth: string): GroupMonth {
  const group = value?.group;
  if (!group || group.id !== expectedGroup || value.month !== expectedMonth || typeof group.name !== 'string' || !Array.isArray(group.origins) || !Array.isArray(group.people) || !Array.isArray(value.availability) || group.origins.length > 6 || group.people.length > 12) throw new GroupApiError('This group could not be loaded safely. Please try again.', 502);
  const origins = new Set<string>();
  for (const origin of group.origins) {
    if (!validId(origin.id) || origins.has(origin.id) || typeof origin.name !== 'string' || typeof origin.locationLabel !== 'string' || !Number.isFinite(origin.lat) || !Number.isFinite(origin.lng)) throw new GroupApiError('This group has an invalid location.', 502);
    origins.add(origin.id);
  }
  const people = new Set<string>();
  for (const person of group.people) {
    if (!validId(person.id) || people.has(person.id) || typeof person.name !== 'string' || (person.originId !== null && !origins.has(person.originId))) throw new GroupApiError('This group has an invalid person.', 502);
    people.add(person.id);
  }
  for (const cell of value.availability) if (!people.has(cell.personId) || !isDate(cell.date) || !cell.date.startsWith(expectedMonth) || !['free', 'unavailable', 'unknown'].includes(cell.status)) throw new GroupApiError('This group has an invalid availability entry.', 502);
  return value;
}
