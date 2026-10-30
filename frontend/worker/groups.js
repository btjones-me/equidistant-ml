import { digest } from './auth.js';

const ID = /^[a-zA-Z0-9_-]{1,80}$/;
const GROUP_ID = /^grp_[a-f0-9]{32}$/;
const TOKEN = /^[a-f0-9]{64}$/;
const MAX_PEOPLE = 12;
const MAX_MEMBERS = 24;
const INVITE_LIFETIME = 7 * 24 * 60 * 60 * 1000;

class GroupError extends Error {
  constructor(status, message, retryAfter) { super(message); this.status = status; this.retryAfter = retryAfter; }
}
const reject = (status, message) => { throw new GroupError(status, message); };
const randomHex = (size = 16) => [...crypto.getRandomValues(new Uint8Array(size))].map(v => v.toString(16).padStart(2, '0')).join('');
const result = (value, status = 200) => Response.json(value, { status, headers: { 'Cache-Control': 'private, no-store', Vary: 'Cookie' } });
const nameValue = (value, max = 120) => {
  if (typeof value !== 'string' || !value.trim() || value.length > max || /[\u0000-\u001f\u007f]/.test(value)) reject(400, 'Enter a valid name.');
  return value.trim();
};
const idValue = value => { if (typeof value !== 'string' || !ID.test(value)) reject(400, 'Invalid record identifier.'); return value; };
const monthValue = value => {
  if (typeof value !== 'string' || !/^(20\d{2})-(0[1-9]|1[0-2])$/.test(value)) reject(400, 'Use a month in YYYY-MM format.');
  return value;
};
const londonMonth = now => {
  const parts = new Intl.DateTimeFormat('en-GB', { timeZone: 'Europe/London', year: 'numeric', month: '2-digit' }).formatToParts(new Date(now));
  return `${parts.find(p => p.type === 'year').value}-${parts.find(p => p.type === 'month').value}`;
};
const dateValue = value => {
  if (typeof value !== 'string' || !/^20\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$/.test(value) || new Date(`${value}T12:00:00Z`).toISOString().slice(0, 10) !== value) reject(400, 'Use a valid date in YYYY-MM-DD format.');
  return value;
};

async function readJson(request) {
  if (!request.headers.get('Content-Type')?.toLowerCase().startsWith('application/json')) reject(415, 'Send a JSON request.');
  if (Number(request.headers.get('Content-Length')) > 32768) reject(413, 'This request is too large.');
  if (!request.body) reject(400, 'A JSON request body is required.');
  const reader = request.body.getReader();
  const chunks = []; let size = 0;
  while (true) {
    const { done, value } = await reader.read(); if (done) break;
    size += value.byteLength;
    if (size > 32768) { await reader.cancel(); reject(413, 'This request is too large.'); }
    chunks.push(value);
  }
  const bytes = new Uint8Array(size); let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  try {
    const data = JSON.parse(new TextDecoder().decode(bytes));
    if (!data || typeof data !== 'object' || Array.isArray(data)) reject(400, 'Send a JSON object.');
    return data;
  } catch (error) { if (error instanceof GroupError) throw error; reject(400, 'The JSON request could not be read.'); }
}

async function rate(db, scope, limit, seconds, now) {
  const window = Math.floor(now / (seconds * 1000)) * seconds * 1000;
  const row = await db.prepare(`INSERT INTO group_rate_limits (scope, window_started_at, requests) VALUES (?1, ?2, 1)
    ON CONFLICT(scope) DO UPDATE SET
      requests = CASE WHEN window_started_at <> ?2 THEN 1 ELSE requests + 1 END,
      window_started_at = ?2
    WHERE window_started_at <> ?2 OR requests < ?3
    RETURNING requests`).bind(scope, window, limit).first();
  if (!row) throw new GroupError(429, 'Too many group requests. Please try again shortly.', Math.max(1, Math.ceil((window + seconds * 1000 - now) / 1000)));
}
async function requireMember(db, groupId, userId) {
  if (!GROUP_ID.test(groupId)) reject(404, 'Group not found.');
  const group = await db.prepare(`SELECT g.group_id AS id, g.name, g.created_at AS createdAt
    FROM planning_groups g JOIN group_members m ON m.group_id = g.group_id
    WHERE g.group_id = ?1 AND m.user_id = ?2`).bind(groupId, userId).first();
  if (!group) reject(404, 'Group not found.');
  return group;
}
async function snapshot(db, group) {
  const origins = (await db.prepare('SELECT origin_id AS id, name, location_label AS locationLabel, lat, lng FROM group_origins WHERE group_id = ?1 ORDER BY position').bind(group.id).all()).results;
  const people = (await db.prepare('SELECT person_id AS id, name, origin_id AS originId FROM group_people WHERE group_id = ?1 AND deleted_at IS NULL ORDER BY position, person_id').bind(group.id).all()).results;
  const members = await db.prepare('SELECT COUNT(*) AS count FROM group_members WHERE group_id = ?1').bind(group.id).first();
  return { ...group, origins, people, memberCount: members.count };
}
function initialSnapshot(body) {
  const name = nameValue(body.name, 80);
  if (!Array.isArray(body.origins) || body.origins.length < 1 || body.origins.length > 6 || !Array.isArray(body.people) || body.people.length < 1 || body.people.length > MAX_PEOPLE) reject(400, 'Use 1–6 starting locations and 1–12 people.');
  const originsById = new Map();
  const origins = body.origins.map(origin => {
    if (!origin || typeof origin !== 'object') reject(400, 'Invalid starting location.');
    const id = idValue(origin.id);
    if (originsById.has(id)) reject(400, 'Starting location IDs must be unique.');
    if (!Number.isFinite(origin.lat) || !Number.isFinite(origin.lng) || origin.lat < 51.2 || origin.lat > 51.75 || origin.lng < -0.6 || origin.lng > 0.35) reject(400, 'Starting locations must be within London.');
    const value = { id, name: nameValue(origin.name), locationLabel: nameValue(origin.locationLabel, 240), lat: origin.lat, lng: origin.lng };
    originsById.set(id, 0); return value;
  });
  const ids = new Set();
  const people = body.people.map(person => {
    if (!person || typeof person !== 'object') reject(400, 'Invalid person.');
    const id = idValue(person.id), originId = idValue(person.originId);
    if (ids.has(id)) reject(400, 'Person IDs must be unique.');
    ids.add(id);
    if (!originsById.has(originId)) reject(400, 'Each person needs one of this group’s starting locations.');
    originsById.set(originId, originsById.get(originId) + 1);
    return { id, originId, name: nameValue(person.name) };
  });
  if ([...originsById.values()].some(count => count < 1 || count > 2)) reject(400, 'Each initial starting location needs one or two people.');
  return { name, origins, people };
}
async function validateOrigin(db, groupId, originId) {
  if (originId === null) return null;
  idValue(originId);
  if (!await db.prepare('SELECT origin_id FROM group_origins WHERE group_id = ?1 AND origin_id = ?2').bind(groupId, originId).first()) reject(400, 'Choose a starting location from this group, or leave it unlinked.');
  return originId;
}

export async function groupRoute(request, env, user) {
  const url = new URL(request.url);
  if (url.pathname !== '/api/groups' && !url.pathname.startsWith('/api/groups/')) return null;
  try {
    // The outer worker has verified the opaque session. Require an exact account
    // binding as well, so stale tabs cannot use a newly signed-in user's groups.
    if (!user?.id || request.headers.get('X-Equidistant-Account') !== user.id) reject(409, 'Your signed-in account changed. Reload to continue.');
    const db = env.DB, now = Date.now();
    const readOnly = request.method === 'GET';
    const actor = await digest(`groups:${user.id}`);
    await rate(db, `${readOnly ? 'read' : 'write'}:${actor}`, 120, 60, now);
    if (!readOnly) await rate(db, `write-hour:${actor}`, 1200, 3600, now);

    if (url.pathname === '/api/groups') {
      if (request.method === 'GET') {
        const groups = (await db.prepare(`SELECT g.group_id AS id, g.name, g.created_at AS createdAt,
          (SELECT COUNT(*) FROM group_members all_members WHERE all_members.group_id = g.group_id) AS memberCount
          FROM planning_groups g JOIN group_members m ON m.group_id = g.group_id WHERE m.user_id = ?1 ORDER BY g.created_at DESC, g.group_id LIMIT 100`).bind(user.id).all()).results;
        return result({ groups });
      }
      if (request.method !== 'POST') reject(405, 'Method not allowed.');
      const body = initialSnapshot(await readJson(request));
      await rate(db, `create:${actor}`, 5, 86400, now);
      await rate(db, 'create-global', 100, 86400, now);
      const id = `grp_${randomHex()}`;
      const statements = [db.prepare(`INSERT INTO planning_groups (group_id, name, created_by, created_at)
        SELECT ?1, ?2, ?3, ?4 WHERE (SELECT COUNT(*) FROM planning_groups WHERE created_by = ?3) < 20`).bind(id, body.name, user.id, now),
        db.prepare('INSERT INTO group_members (group_id, user_id, joined_at) SELECT group_id, ?2, ?3 FROM planning_groups WHERE group_id = ?1').bind(id, user.id, now)];
      for (const [position, origin] of body.origins.entries()) statements.push(db.prepare(`INSERT INTO group_origins (group_id, origin_id, name, location_label, lat, lng, position)
        SELECT group_id, ?2, ?3, ?4, ?5, ?6, ?7 FROM planning_groups WHERE group_id = ?1`).bind(id, origin.id, origin.name, origin.locationLabel, origin.lat, origin.lng, position));
      for (const [position, person] of body.people.entries()) statements.push(db.prepare(`INSERT INTO group_people (group_id, person_id, name, origin_id, position)
        SELECT group_id, ?2, ?3, ?4, ?5 FROM planning_groups WHERE group_id = ?1`).bind(id, person.id, person.name, person.originId, position));
      await db.batch(statements);
      const group = await requireMember(db, id, user.id);
      return result({ groupId: id, group: await snapshot(db, group) }, 201);
    }

    if (url.pathname === '/api/groups/join') {
      if (request.method !== 'POST') reject(405, 'Method not allowed.');
      await rate(db, `join:${actor}`, 20, 3600, now);
      const { token } = await readJson(request);
      if (typeof token !== 'string' || !TOKEN.test(token)) reject(404, 'This invitation is unavailable or expired.');
      const hash = await digest(token);
      // Token bytes never enter SQL, logs, URLs or analytics; only this digest is stored.
      const invitation = await db.prepare('SELECT group_id AS groupId FROM group_invites WHERE token_hash = ?1 AND expires_at > ?2').bind(hash, now).first();
      if (!invitation) reject(404, 'This invitation is unavailable or expired.');
      const joined = await db.prepare(`INSERT INTO group_members (group_id, user_id, joined_at)
        SELECT i.group_id, ?2, ?3 FROM group_invites i WHERE i.token_hash = ?1 AND i.expires_at > ?3
        AND ((SELECT COUNT(*) FROM group_members WHERE group_id = i.group_id) < ?4
          OR EXISTS(SELECT 1 FROM group_members WHERE group_id = i.group_id AND user_id = ?2))
        ON CONFLICT(group_id, user_id) DO UPDATE SET joined_at = group_members.joined_at
        RETURNING group_id AS groupId`).bind(hash, user.id, now, MAX_MEMBERS).first();
      if (!joined) reject(409, 'This group has reached its membership limit.');
      return result({ groupId: joined.groupId });
    }

    const route = url.pathname.match(/^\/api\/groups\/([^/]+)(?:\/(availability|invite|people)(?:\/([^/]+))?)?$/);
    if (!route) reject(404, 'Group route not found.');
    const [, groupId, action, personId] = route;
    const group = await requireMember(db, groupId, user.id);
    if (!action) {
      if (request.method !== 'GET') reject(405, 'Method not allowed.');
      const month = monthValue(url.searchParams.get('month') || londonMonth(now));
      const availability = (await db.prepare(`SELECT a.person_id AS personId, a.date, a.status, a.updated_at AS updatedAt
        FROM group_availability a JOIN group_people p ON p.group_id = a.group_id AND p.person_id = a.person_id
        WHERE a.group_id = ?1 AND a.date >= ?2 AND a.date <= ?3 AND p.deleted_at IS NULL ORDER BY a.date, p.position`).bind(groupId, `${month}-01`, `${month}-31`).all()).results;
      return result({ group: await snapshot(db, group), month, availability, serverTime: now });
    }
    if (action === 'availability' && !personId) {
      if (request.method !== 'PATCH') reject(405, 'Method not allowed.');
      const body = await readJson(request);
      const target = idValue(body.personId), date = dateValue(body.date), status = body.status;
      if (!['unknown', 'free', 'unavailable'].includes(status)) reject(400, 'Choose unknown, free or unavailable.');
      const changed = await db.prepare(`INSERT INTO group_availability (group_id, person_id, date, status, updated_at, updated_by)
        SELECT group_id, person_id, ?3, ?4, ?5, ?6 FROM group_people WHERE group_id = ?1 AND person_id = ?2 AND deleted_at IS NULL
        ON CONFLICT(group_id, person_id, date) DO UPDATE SET status = excluded.status, updated_at = excluded.updated_at, updated_by = excluded.updated_by
        RETURNING person_id AS personId, date, status, updated_at AS updatedAt`).bind(groupId, target, date, status, now, user.id).first();
      if (!changed) reject(404, 'Person not found.');
      return result(changed);
    }
    if (action === 'invite' && !personId) {
      if (request.method !== 'POST') reject(405, 'Method not allowed.');
      await rate(db, `invite:${actor}`, 10, 3600, now);
      const token = randomHex(32), expiresAt = now + INVITE_LIFETIME;
      await db.batch([
        db.prepare('DELETE FROM group_invites WHERE expires_at <= ?1').bind(now),
        db.prepare('INSERT INTO group_invites (token_hash, group_id, created_by, created_at, expires_at) VALUES (?1, ?2, ?3, ?4, ?5)').bind(await digest(token), groupId, user.id, now, expiresAt)
      ]);
      return result({ url: `${env.AUTH_ORIGIN}/50pubs#invite=${token}`, expiresAt });
    }
    if (action === 'people') {
      if (!personId && request.method === 'POST') {
        const body = await readJson(request), name = nameValue(body.name), originId = await validateOrigin(db, groupId, body.originId);
        const id = `person_${randomHex()}`;
        const person = await db.prepare(`INSERT INTO group_people (group_id, person_id, name, origin_id, position)
          SELECT ?1, ?2, ?3, ?4, COALESCE((SELECT MAX(position) + 1 FROM group_people WHERE group_id = ?1), 0)
          WHERE (SELECT COUNT(*) FROM group_people WHERE group_id = ?1 AND deleted_at IS NULL) < ?5
            AND (?4 IS NULL OR (SELECT COUNT(*) FROM group_people WHERE group_id = ?1 AND origin_id = ?4 AND deleted_at IS NULL) < 2)
          RETURNING person_id AS id, name, origin_id AS originId`).bind(groupId, id, name, originId, MAX_PEOPLE).first();
        if (!person) reject(409, 'Use at most 12 people, with at most two linked to each starting location.');
        return result({ person }, 201);
      }
      if (personId && request.method === 'PATCH') {
        idValue(personId);
        const body = await readJson(request);
        if (!Object.keys(body).length || Object.keys(body).some(key => !['name', 'originId'].includes(key))) reject(400, 'Change a name or starting-location link.');
        const hasName = Object.hasOwn(body, 'name'), hasOrigin = Object.hasOwn(body, 'originId');
        const name = hasName ? nameValue(body.name) : null;
        const originId = hasOrigin ? await validateOrigin(db, groupId, body.originId) : null;
        // Apply only supplied fields atomically. An older tab renaming someone
        // must not undo a concurrent location change (or vice versa).
        const person = await db.prepare(`UPDATE group_people SET
          name = CASE WHEN ?5 THEN ?3 ELSE name END,
          origin_id = CASE WHEN ?6 THEN ?4 ELSE origin_id END
          WHERE group_id = ?1 AND person_id = ?2 AND deleted_at IS NULL
          AND (?6 = 0 OR ?4 IS NULL OR (SELECT COUNT(*) FROM group_people WHERE group_id = ?1 AND origin_id = ?4 AND person_id <> ?2 AND deleted_at IS NULL) < 2)
          RETURNING person_id AS id, name, origin_id AS originId`).bind(groupId, personId, name, originId, Number(hasName), Number(hasOrigin)).first();
        if (!person) {
          const existing = await db.prepare('SELECT person_id FROM group_people WHERE group_id = ?1 AND person_id = ?2 AND deleted_at IS NULL').bind(groupId, personId).first();
          if (!existing) reject(404, 'Person not found.');
          reject(409, 'At most two people can be linked to each starting location.');
        }
        return result({ person });
      }
      if (personId && request.method === 'DELETE') {
        idValue(personId);
        await db.prepare('UPDATE group_people SET deleted_at = ?3 WHERE group_id = ?1 AND person_id = ?2 AND deleted_at IS NULL').bind(groupId, personId, now).run();
        return result({ personId, deleted: true });
      }
    }
    reject(405, 'Method not allowed.');
  } catch (error) {
    if (!(error instanceof GroupError)) throw error;
    const response = result({ detail: error.message }, error.status);
    if (error.retryAfter) response.headers.set('Retry-After', String(error.retryAfter));
    return response;
  }
}

// Synchronous first-party bootstrap: never fetch or join automatically. Tokens
// belong only in a short-lived tab-local buffer, never request URLs or analytics.
export const groupInviteScript = `(()=>{try{const p=new URLSearchParams(location.hash.slice(1));if(!p.has('invite'))return;const token=p.get('invite');history.replaceState(history.state,'',location.pathname+location.search);if(/^[a-f0-9]{64}$/.test(token||'')){try{sessionStorage.setItem('equidistant:pending-group-invite',JSON.stringify({token,expiresAt:Date.now()+600000}));}catch{}}}catch{}})();`;
