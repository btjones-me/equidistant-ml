import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { CalendarDays, ChevronLeft, ChevronRight, MapPin, Plus, X } from 'lucide-react';
import type { Friend } from './types';
import { attendanceForDate, cellKey, dateLabel, groupRequest, londonToday, makeGroupDraft, monthDates, monthLabel, nextStatus, PENDING_INVITE_KEY, readPendingInvite, shiftMonth, statusMap, validateGroupMonth, type AvailabilityCell, type GroupDraft, type GroupMonth, type GroupPerson, type GroupSummary, type GroupTravelSelection } from './lib/groupAvailability';

type Props = { friends: Friend[]; accountId: string; activeSelection: GroupTravelSelection | null; onSelection: (selection: GroupTravelSelection | null) => void };
const message = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong. Please try again.';
const symbols = { unknown: '–', free: '✓', unavailable: '×' };
export default function GroupAvailability(props: Props) { return <AvailabilityPanel key={props.accountId} {...props} />; }
function AvailabilityPanel({ friends, activeSelection, onSelection }: Props) {
  const [pendingInvite, setPendingInvite] = useState(readPendingInvite);
  const [open, setOpen] = useState(() => !!readPendingInvite());
  const [groups, setGroups] = useState<GroupSummary[]>([]);
  const [groupId, setGroupId] = useState('');
  const [month, setMonth] = useState(() => londonToday().slice(0, 7));
  const [data, setData] = useState<GroupMonth | null>(null);
  const [draft, setDraft] = useState<GroupDraft | null>(null);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [appliedDate, setAppliedDate] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [savingCells, setSavingCells] = useState<Set<string>>(new Set());
  const [saved, setSaved] = useState(false);
  const [invite, setInvite] = useState('');
  const [copied, setCopied] = useState(false);
  const [editing, setEditing] = useState<GroupPerson | null>(null);
  const [adding, setAdding] = useState(false);
  const [newName, setNewName] = useState('');
  const [newOrigin, setNewOrigin] = useState('');
  const [locationInfo, setLocationInfo] = useState<string | null>(null);
  const editBaseline = useRef<GroupPerson | null>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const launch = useRef<HTMLButtonElement>(null);
  const alive = useRef(true), revision = useRef(0), operation = useRef(false), writes = useRef(new Set<string>());
  const context = useRef({ groupId, month }); context.current = { groupId, month };
  const selectionCallback = useRef(onSelection); selectionCallback.current = onSelection;
  const selectionRef = useRef(activeSelection); selectionRef.current = activeSelection;
  useEffect(() => { alive.current = true; return () => { alive.current = false; revision.current++; }; }, []);
  const loadGroups = useCallback(async () => {
    const result = await groupRequest<{ groups: GroupSummary[] }>('/api/groups');
    if (alive.current) { setGroups(result.groups); setGroupId(current => current || result.groups[0]?.id || ''); }
    return result.groups;
  }, []);
  useEffect(() => {
    if (!open) return;
    let active = true;
    setLoading(true);
    void loadGroups().catch(reason => { if (active) setError(message(reason)); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [open, loadGroups]);
  const loadMonth = useCallback(async (quiet = false) => {
    if (!groupId || writes.current.size || operation.current) return;
    const ticket = ++revision.current;
    if (!quiet) setLoading(true);
    try {
      const result = validateGroupMonth(await groupRequest<GroupMonth>(`/api/groups/${groupId}?month=${month}`), groupId, month);
      if (alive.current && ticket === revision.current && !writes.current.size) { setData(result); if (!quiet) setError(''); }
    } catch (reason) { if (alive.current && ticket === revision.current) { setError(message(reason)); setData(null); if (selectionRef.current?.groupId === groupId) selectionCallback.current({ ...selectionRef.current, freePeople: [], error: message(reason) }); } }
    finally { if (alive.current && ticket === revision.current) setLoading(false); }
  }, [groupId, month]);
  useEffect(() => {
    if ((!open && activeSelection?.groupId !== groupId) || !groupId) return;
    void loadMonth();
    const timer = window.setInterval(() => { if (document.visibilityState !== 'hidden') void loadMonth(true); }, 10_000);
    return () => { revision.current++; window.clearInterval(timer); };
  }, [open, groupId, loadMonth, activeSelection?.groupId]);
  useEffect(() => {
    if (!open) return;
    const oldOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialog.current?.focus();
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); setOpen(false); }
      if (event.key !== 'Tab') return;
      const controls = [...(dialog.current?.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex="0"]') || [])];
      const first = controls[0], last = controls[controls.length - 1];
      if (!first) { event.preventDefault(); return; }
      if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    window.addEventListener('keydown', keydown);
    return () => { document.body.style.overflow = oldOverflow; window.removeEventListener('keydown', keydown); launch.current?.focus(); };
  }, [open]);
  const current = data?.group.id === groupId && data.month === month ? data : null;
  const statuses = useMemo(() => statusMap(current?.availability || []), [current]);
  const attendance = current && selectedDate ? attendanceForDate(current.group.people, selectedDate, statuses) : null;
  const missing = attendance?.freePeople.filter(person => !person.originId).map(person => person.name) || [];
  const apply = useCallback((value: GroupMonth, date: string) => {
    const freePeople = attendanceForDate(value.group.people, date, statusMap(value.availability)).freePeople;
    selectionCallback.current({ groupId: value.group.id, groupName: value.group.name, date, origins: value.group.origins, people: value.group.people, freePeople });
  }, []);
  useEffect(() => { if (current && appliedDate && appliedDate.startsWith(month) && activeSelection?.groupId === current.group.id && activeSelection.date === appliedDate) apply(current, appliedDate); }, [current, appliedDate, month, apply, activeSelection?.groupId, activeSelection?.date]);
  function chooseGroup(id: string) {
    revision.current++; setGroupId(id); setData(null); setSelectedDate(null); setAppliedDate(null); setInvite(''); setEditing(null); setAdding(false); setError(''); selectionCallback.current(null);
  }
  function changeMonth(direction: number) {
    revision.current++; setMonth(value => shiftMonth(value, direction)); setSelectedDate(null); setAppliedDate(null); setData(null); selectionCallback.current(null);
  }
  async function mutate<T>(url: string, init: RequestInit): Promise<T | null> {
    if (operation.current) return null;
    operation.current = true; revision.current++; setBusy(true); setError('');
    try { return await groupRequest<T>(url, init); }
    catch (reason) { if (alive.current) setError(message(reason)); return null; }
    finally { operation.current = false; if (alive.current) { setBusy(false); setLoading(false); } }
  }
  async function createGroup(event: FormEvent) {
    event.preventDefault();
    if (!draft) return;
    const result = await mutate<{ groupId: string }>('/api/groups', { method: 'POST', body: JSON.stringify(draft) });
    if (result && alive.current) { setDraft(null); chooseGroup(result.groupId); void loadGroups().catch(reason => setError(message(reason))); }
  }
  async function joinGroup() {
    if (!pendingInvite) return;
    const result = await mutate<{ groupId: string }>('/api/groups/join', { method: 'POST', body: JSON.stringify({ token: pendingInvite }) });
    if (result && alive.current) {
      try { sessionStorage.removeItem(PENDING_INVITE_KEY); } catch { /* optional storage */ }
      setPendingInvite(null); chooseGroup(result.groupId); void loadGroups().catch(reason => setError(message(reason)));
    }
  }
  async function saveCell(personId: string, date: string) {
    const key = cellKey(personId, date);
    if (!current || writes.current.has(key) || operation.current) return;
    const targetGroup = groupId, targetMonth = month;
    writes.current.add(key); revision.current++; setSavingCells(new Set(writes.current)); setSaved(false); setError('');
    try {
      const cell = await groupRequest<AvailabilityCell>(`/api/groups/${targetGroup}/availability`, { method: 'PATCH', body: JSON.stringify({ personId, date, status: nextStatus(statuses[key] || 'unknown') }) });
      if (alive.current && context.current.groupId === targetGroup && context.current.month === targetMonth) {
        setData(previous => previous && previous.group.id === targetGroup && previous.month === targetMonth ? { ...previous, availability: [...previous.availability.filter(existing => cellKey(existing.personId, existing.date) !== key), cell] } : previous);
        setSaved(true);
      }
    } catch (reason) { if (alive.current && context.current.groupId === targetGroup && context.current.month === targetMonth) setError(message(reason)); }
    finally { writes.current.delete(key); if (alive.current) { setSavingCells(new Set(writes.current)); setLoading(false); } }
  }
  async function savePerson(event: FormEvent) {
    event.preventDefault();
    if (!current) return;
    const target = groupId;
    const changes = editing ? { ...(editing.name !== editBaseline.current?.name ? { name: editing.name } : {}), ...(editing.originId !== editBaseline.current?.originId ? { originId: editing.originId } : {}) } : { name: newName, originId: newOrigin || null };
    if (!Object.keys(changes).length) { setEditing(null); return; }
    const result = await mutate(`/api/groups/${target}/people${editing ? `/${editing.id}` : ''}`, { method: editing ? 'PATCH' : 'POST', body: JSON.stringify(changes) });
    if (result && alive.current && context.current.groupId === target) { setEditing(null); setAdding(false); setNewName(''); setNewOrigin(''); void loadMonth(); }
  }
  async function removePerson() {
    if (!editing) return;
    const result = await mutate(`/api/groups/${groupId}/people/${editing.id}`, { method: 'DELETE' });
    if (result && alive.current) { setEditing(null); void loadMonth(); }
  }
  async function makeInvite() {
    const result = await mutate<{ url: string }>(`/api/groups/${groupId}/invite`, { method: 'POST' });
    if (result && alive.current) { setInvite(result.url); setCopied(false); }
  }
  async function copyInvite() {
    try { await navigator.clipboard.writeText(invite); setCopied(true); }
    catch { setError('Copying is unavailable in this browser. Select and copy the invitation link below.'); }
  }
  const people = current?.group.people || [];
  const columns = current ? [...current.group.origins.map(origin => ({ id: origin.id, label: origin.name, people: people.filter(person => person.originId === origin.id) })), { id: 'unlinked', label: 'No location linked', people: people.filter(person => !person.originId) }].filter(column => column.people.length) : [];
  return <>
    <button ref={launch} type="button" className="group-launch" onClick={() => setOpen(true)}><CalendarDays size={19} /><span><strong>Group availability</strong><small>Find a day. Find your pub.</small></span><span aria-hidden="true">→</span></button>
    {open && <div className="group-backdrop"><div ref={dialog} className="group-panel" role="dialog" aria-modal="true" aria-labelledby="group-title" tabIndex={-1}>
      <header className="group-panel-header"><div><span className="fifty-eyebrow">MAKE IT HAPPEN</span><h2 id="group-title">Group availability</h2></div><button type="button" className="small-icon-button" aria-label="Close group availability" onClick={() => setOpen(false)}><X size={20} /></button></header>
      <div className="group-body">
        {error && <p role="alert" className="group-error">{error} <button type="button" onClick={() => { setError(''); void loadGroups().then(() => loadMonth()).catch(reason => setError(message(reason))); }}>Retry loading</button></p>}
        {pendingInvite && <section className="group-invitation"><h3>You’ve been invited</h3><p>Joining shares your signed-in account with this group. Everyone in the group can view and edit everyone’s availability and starting locations.</p><button type="button" className="group-primary" onClick={() => void joinGroup()} disabled={busy}>Join this group</button><button type="button" className="group-text-button" disabled={busy} onClick={() => { try { sessionStorage.removeItem(PENDING_INVITE_KEY); } catch { /* optional */ } setPendingInvite(null); }}>Decline invitation</button></section>}
        {!draft && <div className="group-toolbar">{groups.length > 0 && <label>Shared group<select aria-label="Shared group" value={groupId} onChange={event => chooseGroup(event.target.value)} disabled={busy || savingCells.size > 0}>{groups.map(group => <option key={group.id} value={group.id}>{group.name}</option>)}</select></label>}<button type="button" className="group-outline" disabled={busy || savingCells.size > 0 || !friends.length} onClick={() => { setDraft(makeGroupDraft(friends)); setError(''); }}><Plus size={16} /> Start a group session</button></div>}
        {!draft && !groups.length && !loading && <div className="group-empty"><CalendarDays size={32} /><h3>A good night starts with a date.</h3><p>Start a group with your current starting locations, or open an invitation from a friend.</p></div>}
        {draft && <form className="group-draft" onSubmit={createGroup}><h3>Start a group session?</h3><p>You’ll be able to select your availability. Pub recommendations will change depending on what’s equidistant for those available.</p><label>Group name<input required maxLength={80} value={draft.name} onChange={event => setDraft({ ...draft, name: event.target.value })} /></label><p className="group-muted">Review the people at each starting location. Suggested names are editable. This creates a separate shared copy of your planner.</p>{draft.origins.map(origin => {
          const members = draft.people.filter(person => person.originId === origin.id);
          return <fieldset key={origin.id}><legend>{origin.name}</legend><small><MapPin size={13} /> {origin.locationLabel}</small><div className="group-draft-people">{members.map(person => <div key={person.id}><input aria-label={`Person at ${origin.name}`} required maxLength={120} value={person.name} onChange={event => setDraft({ ...draft, people: draft.people.map(value => value.id === person.id ? { ...value, name: event.target.value } : value) })} />{members.length > 1 && <button type="button" aria-label={`Remove ${person.name} from draft`} onClick={() => setDraft({ ...draft, people: draft.people.filter(value => value.id !== person.id) })}><X size={16} /></button>}</div>)}</div>{members.length < 2 && <button type="button" className="group-text-button" onClick={() => setDraft({ ...draft, people: [...draft.people, { id: crypto.randomUUID(), name: '', originId: origin.id }] })}>+ Add second person</button>}</fieldset>;
        })}<div className="group-actions"><button className="group-primary" disabled={busy} type="submit">{busy ? 'Creating…' : 'Create shared group'}</button><button type="button" className="group-outline" disabled={busy} onClick={() => setDraft(null)}>Cancel</button></div></form>}
        {loading && !current && !draft && <p role="status">Loading your shared availability…</p>}
        {current && !draft && <>
          <div className="group-sharing"><p><strong>{current.group.name}</strong><small>{current.group.memberCount} signed-in {current.group.memberCount === 1 ? 'member' : 'members'} · everyone can edit</small></p><button type="button" className="group-outline" disabled={busy} onClick={() => void makeInvite()}>Invite friends</button></div>
          {invite && <div className="group-invite-link"><label>Shared editing invitation<input aria-label="Invitation link" readOnly value={invite} onFocus={event => event.target.select()} /></label><button type="button" className="group-outline" onClick={() => void copyInvite()}>{copied ? 'Copied' : 'Copy link'}</button><small>Valid for 7 days. Anyone with this link can sign in and join. Share only with your group.</small></div>}
          <div className="group-month"><button type="button" aria-label="Previous month" disabled={busy || savingCells.size > 0 || month <= '2000-01'} onClick={() => changeMonth(-1)}><ChevronLeft size={20} /></button><h3>{monthLabel(month)}</h3><button type="button" aria-label="Next month" disabled={busy || savingCells.size > 0 || month >= '2099-12'} onClick={() => changeMonth(1)}><ChevronRight size={20} /></button></div>
          <div className="group-legend"><span><b className="free">✓</b> Free</span><span><b className="unavailable">×</b> Unavailable</span><span><b>–</b> Unanswered</span><small role="status">{savingCells.size ? 'Saving…' : saved ? 'Saved' : 'Changes save automatically'}</small></div>
          <p className="group-muted">Tap a cell to cycle through the three answers. Dates use London time. Tap a name to edit or link a location.</p>
          {!people.length ? <p>No people yet. Add someone below to start sharing availability.</p> : <div className="group-grid-scroll" tabIndex={0} aria-label="Monthly availability, scroll horizontally for more people"><table className="group-grid"><thead><tr><th rowSpan={2} scope="col">Date</th>{columns.map(column => <th scope="colgroup" colSpan={column.people.length} key={column.id}>{column.label}</th>)}<th rowSpan={2} scope="col">Who’s free?</th></tr><tr>{columns.flatMap(column => column.people).map(person => <th scope="col" key={person.id}><button type="button" className="group-person" onClick={() => { editBaseline.current = { ...person }; setEditing({ ...person }); setAdding(false); }}>{person.name}</button><button type="button" className={person.originId ? 'group-pin linked' : 'group-pin'} aria-label={`Location for ${person.name}`} onClick={() => setLocationInfo(locationInfo === person.id ? null : person.id)}>{person.originId ? <MapPin size={14} /> : '❓'}</button>{locationInfo === person.id && <span className="group-location-info">{person.originId ? `Location linked: ${current.group.origins.find(origin => origin.id === person.originId)?.locationLabel}` : 'No location linked. Tap the name to choose one.'}</span>}</th>)}</tr></thead><tbody>{monthDates(month).map(date => {
            const summary = attendanceForDate(people, date, statuses);
            return <tr key={date} className={`${summary.everyoneFree ? 'everyone-free ' : ''}${selectedDate === date ? 'selected-date' : ''}`}><th scope="row"><button type="button" aria-label={`Select ${dateLabel(date)}`} aria-pressed={selectedDate === date} onClick={() => setSelectedDate(date)}>{dateLabel(date, { weekday: 'short', day: 'numeric' })}</button></th>{columns.flatMap(column => column.people).map(person => {
              const key = cellKey(person.id, date), status = statuses[key] || 'unknown';
              return <td key={person.id}><button className={`group-cell ${status}`} type="button" aria-label={`${person.name}, ${dateLabel(date)}: ${status === 'unknown' ? 'unanswered' : status}. Set ${nextStatus(status) === 'unknown' ? 'unanswered' : nextStatus(status)}`} disabled={savingCells.has(key) || busy} onClick={() => void saveCell(person.id, date)}>{savingCells.has(key) ? '…' : symbols[status]}</button></td>;
            })}<td><button type="button" className="group-attendance" onClick={() => setSelectedDate(date)}><strong>{summary.everyoneFree ? 'Everyone free' : `${summary.free} free`}</strong><small>{summary.unavailable} unavailable · {summary.unknown} unanswered</small></button></td></tr>;
          })}</tbody></table></div>}
          <button type="button" className="group-text-button" disabled={people.length >= 12 || busy} onClick={() => { setAdding(true); setEditing(null); }}>+ Add a person</button>
          {(editing || adding) && <form className="group-person-editor" onSubmit={savePerson}><h3>{editing ? 'Edit person' : 'Add person'}</h3><label>Name<input required maxLength={120} value={editing?.name ?? newName} onChange={event => editing ? setEditing({ ...editing, name: event.target.value }) : setNewName(event.target.value)} /></label><label>Starting location<select value={editing ? editing.originId || '' : newOrigin} onChange={event => editing ? setEditing({ ...editing, originId: event.target.value || null }) : setNewOrigin(event.target.value)}><option value="">No location linked</option>{current.group.origins.map(origin => <option key={origin.id} value={origin.id} disabled={people.filter(person => person.originId === origin.id && person.id !== editing?.id).length >= 2}>{origin.name} · {origin.locationLabel}</option>)}</select></label><p className="group-muted">Up to two people can share each starting location. A name change keeps their saved answers.</p><div className="group-actions"><button type="submit" className="group-primary" disabled={busy}>Save person</button><button type="button" className="group-outline" disabled={busy} onClick={() => { setEditing(null); setAdding(false); }}>Cancel</button>{editing && <button type="button" className="group-remove" disabled={busy} onClick={() => void removePerson()}>Remove from group</button>}</div></form>}
          <section className="group-date-choice">{selectedDate && attendance ? <><h3>{dateLabel(selectedDate)}</h3><p>{attendance.freePeople.length ? attendance.freePeople.map(person => person.name).join(', ') : 'Nobody has confirmed they are free.'}</p>{missing.length > 0 && <p className="group-error">Link a starting location for {missing.join(', ')} before finding pubs.</p>}<button className="group-primary" type="button" disabled={!attendance.free || missing.length > 0 || savingCells.size > 0 || busy} onClick={() => { setAppliedDate(selectedDate); apply(current, selectedDate); setOpen(false); }}>Find pubs for {attendance.free} {attendance.free === 1 ? 'person' : 'people'}</button></> : <p>Select a date to find pubs for the people who are free.</p>}<small>The date chooses attendees, not live timetables or pub opening hours.</small></section>
        </>}
      </div>
    </div></div>}
  </>;
}
