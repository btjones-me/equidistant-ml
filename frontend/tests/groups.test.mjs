import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import { createHash } from 'node:crypto';
import worker from '../worker/index.js';
import { groupInviteScript } from '../worker/groups.js';
import { createSession } from '../worker/auth.js';
import { testCookie, testEnvironment, testUser } from './db.mjs';

const draft = () => ({name: 'Example plans', origins: [{id:'home', name:'Example & Partner',locationLabel:'Example station',lat:51.55,lng:-0.19}],people:[{id:'p1',name:'Example',originId:'home'},{id:'p2',name:'Partner',originId:'home'}]});
function req(path, method='GET', body, user=testUser, cookie=testCookie, headers={}) {
  return new Request(`https://example.test${path}`, {method,headers:{Cookie:cookie,Origin:'https://example.test','X-Equidistant-Account':user.id,...(body === undefined ? {} : {'Content-Type':'application/json'}),...headers},...(body === undefined ? {} : {body:JSON.stringify(body)})});
}
async function setup() {
 const env=testEnvironment();
 const response=await worker.fetch(req('/api/groups','POST',draft()),env);
 assert.equal(response.status,201,await response.clone().text());
 const {groupId}=await response.json();
 const other={id:'other-user',email:'other@example.test',name:'Other'};
 const cookie=(await createSession(env.DB,other)).split(';')[0];
 return {env,groupId,other,cookie};
}

test('group routes require Google session, exact account binding and same origin writes',async()=>{
 const {env,groupId}=await setup();
 for(const path of ['/api/groups',`/api/groups/${groupId}?month=2026-10`]) {
  assert.equal((await worker.fetch(req(path,'GET',undefined,testUser,''),env)).status,401);
  assert.equal((await worker.fetch(req(path,'GET',undefined,testUser,testCookie,{'X-Equidistant-Account':''}),env)).status,409);
  assert.equal((await worker.fetch(req(path,'GET',undefined,testUser,testCookie,{'X-Equidistant-Account':'different'}),env)).status,409);
 }
 assert.equal((await worker.fetch(req('/api/groups','POST',draft(),testUser,testCookie,{Origin:'https://evil.example'}),env)).status,403);
 assert.equal(env.DB.sqlite.prepare('SELECT COUNT(*) n FROM planning_groups').get().n,1);
});

test('private group list, snapshot and all mutations reject unrelated signed-in accounts',async()=>{
 const {env,groupId,other,cookie}=await setup();
 assert.deepEqual(await (await worker.fetch(req('/api/groups','GET',undefined,other,cookie),env)).json(),{groups:[]});
 for(const [path,method,body] of [[`/api/groups/${groupId}?month=2026-10`,'GET'],[`/api/groups/${groupId}/invite`,'POST'],[`/api/groups/${groupId}/availability`,'PATCH',{personId:'p1',date:'2026-10-30',status:'free'}],[`/api/groups/${groupId}/people`,'POST',{name:'Intruder',originId:null}],[`/api/groups/${groupId}/people/p1`,'PATCH',{name:'Changed'}],[`/api/groups/${groupId}/people/p1`,'DELETE']]) {
  const response=await worker.fetch(req(path,method,body,other,cookie),env);
  assert.equal(response.status,404,`${method} ${path}`);
  assert.doesNotMatch(await response.text(),/Example station|Partner|51\.55/);
 }
});

test('invites contain random fragment tokens stored only hashed; explicit join grants group membership',async()=>{
 const {env,groupId,other,cookie}=await setup();
 const invite=await (await worker.fetch(req(`/api/groups/${groupId}/invite`,'POST'),env)).json();
 const url=new URL(invite.url),token=url.hash.slice('#invite='.length);
 assert.equal(url.origin,'https://example.test'); assert.equal(url.pathname,'/50pubs'); assert.equal(url.search,'');assert.match(token,/^[a-f0-9]{64}$/);
 const stored=env.DB.sqlite.prepare('SELECT * FROM group_invites').get();
 assert.equal(stored.token_hash,createHash('sha256').update(token).digest('hex'));assert.ok(!JSON.stringify(stored).includes(token));
 assert.equal((await worker.fetch(req(`/api/groups/${groupId}`,'GET',undefined,other,cookie),env)).status,404);
 const joined=await worker.fetch(req('/api/groups/join','POST',{token},other,cookie),env);assert.equal(joined.status,200);assert.deepEqual(await joined.json(),{groupId});
 assert.equal((await worker.fetch(req('/api/groups/join','POST',{token},other,cookie),env)).status,200);
 assert.equal(env.DB.sqlite.prepare('SELECT COUNT(*) n FROM group_members').get().n,2);
 const snapshot=await (await worker.fetch(req(`/api/groups/${groupId}`,'GET',undefined,other,cookie),env)).json();assert.match(snapshot.month,/^20\d\d-\d\d$/);assert.equal(snapshot.group.people.length,2);
 assert.equal((await worker.fetch(req(`/api/groups/${groupId}/availability`,'PATCH',{personId:'p1',date:'2026-10-30',status:'free'},other,cookie),env)).status,200);
 const loaded=await worker.fetch(req(`/api/groups/${groupId}?month=2026-10`),env);
 assert.match(loaded.headers.get('Cache-Control'),/private, no-store/);assert.equal(loaded.headers.get('Vary'),'Cookie');
 const data=await loaded.json();assert.equal(data.availability[0].status,'free');assert.equal(data.availability[0].personId,'p1');
 for(const row of data.group.people) assert.ok(!('email' in row));
 env.DB.sqlite.prepare('UPDATE group_invites SET expires_at=0').run();assert.equal((await worker.fetch(req('/api/groups/join','POST',{token},other,cookie),env)).status,404);
});

test('per-cell saves preserve other dates and people, month filtering and deleted-person protection',async()=>{
 const {env,groupId}=await setup();
 for(const [personId,date,status] of [['p1','2026-10-30','free'],['p2','2026-10-30','unavailable'],['p1','2026-11-01','free'],['p1','2026-10-30','unknown']]) assert.equal((await worker.fetch(req(`/api/groups/${groupId}/availability`,'PATCH',{personId,date,status}),env)).status,200);
 let data=await (await worker.fetch(req(`/api/groups/${groupId}?month=2026-10`),env)).json();assert.equal(data.availability.length,2);assert.equal(data.availability.find(c=>c.personId==='p1').status,'unknown');assert.equal(data.availability.find(c=>c.personId==='p2').status,'unavailable');
 assert.equal((await worker.fetch(req(`/api/groups/${groupId}/people/p1`,'PATCH',{name:'Renamed'}),env)).status,200);
 data=await (await worker.fetch(req(`/api/groups/${groupId}?month=2026-10`),env)).json();assert.equal(data.group.people[0].id,'p1');assert.equal(data.group.people[0].name,'Renamed');assert.equal(data.availability.length,2);
 assert.equal((await worker.fetch(req(`/api/groups/${groupId}/people/p1`,'DELETE'),env)).status,200);
 assert.equal((await worker.fetch(req(`/api/groups/${groupId}/availability`,'PATCH',{personId:'p1',date:'2026-10-30',status:'free'}),env)).status,404);
 data=await (await worker.fetch(req(`/api/groups/${groupId}?month=2026-10`),env)).json();assert.equal(data.group.people.length,1);assert.equal(data.availability.length,1);
});

test('people can be unlinked or added but cannot cross groups or overfill a shared location',async()=>{
 const {env,groupId}=await setup();
 assert.equal((await worker.fetch(req(`/api/groups/${groupId}/people`,'POST',{name:'Third',originId:'home'}),env)).status,409);
 const response=await worker.fetch(req(`/api/groups/${groupId}/people`,'POST',{name:'New',originId:null}),env);assert.equal(response.status,201);const {person}=await response.json();
 assert.equal(person.originId,null);
 assert.equal((await worker.fetch(req(`/api/groups/${groupId}/people/${person.id}`,'PATCH',{originId:'home'}),env)).status,409);
 assert.equal((await worker.fetch(req(`/api/groups/${groupId}/people/p2`,'PATCH',{originId:null}),env)).status,200);
 assert.equal((await worker.fetch(req(`/api/groups/${groupId}/people/${person.id}`,'PATCH',{originId:'home'}),env)).status,200);
 assert.equal((await worker.fetch(req(`/api/groups/${groupId}/people/${person.id}`,'PATCH',{originId:'outside-origin'}),env)).status,400);
 const other=await (await worker.fetch(req('/api/groups','POST',{...draft(),people:[{id:'foreign',name:'Foreign',originId:'home'}]}),env)).json();
 assert.notEqual(other.groupId,groupId);
 assert.equal((await worker.fetch(req(`/api/groups/${groupId}/availability`,'PATCH',{personId:'foreign',date:'2026-10-30',status:'free'}),env)).status,404);
});

test('invalid dates, statuses, names, coordinates and oversized requests fail without data mutation',async()=>{
 const {env,groupId}=await setup();
 for(const body of [{personId:'p1',date:'2026-02-30',status:'free'},{personId:'p1',date:'2026-13-01',status:'free'},{personId:'p1',date:'2026-10-30',status:'maybe'},{personId:'not-real',date:'2026-10-30',status:'free'}]) assert.ok([400,404].includes((await worker.fetch(req(`/api/groups/${groupId}/availability`,'PATCH',body),env)).status));
 assert.equal((await worker.fetch(req(`/api/groups/${groupId}?month=10/2026`),env)).status,400);
 for(const value of [{...draft(),name:''},{...draft(),origins:[{...draft().origins[0],lat:90}]},{...draft(),people:[...draft().people,draft().people[0]]}]) assert.equal((await worker.fetch(req('/api/groups','POST',value),env)).status,400);
 assert.equal((await worker.fetch(req('/api/groups','POST',{...draft(),name:'a'.repeat(40000)}),env)).status,413);
 assert.equal(env.DB.sqlite.prepare('SELECT COUNT(*) n FROM group_availability').get().n,0);
});

test('anonymous invite bootstrap scrubs the fragment before other scripts and never joins or fetches',async()=>{
 const env=testEnvironment();const response=await worker.fetch(req('/50pubs','GET',undefined,testUser,''),env);const html=await response.text();assert.ok(html.indexOf('/group-invite.js')<html.indexOf('/welcome-motion.js'));
 const script=await worker.fetch(req('/group-invite.js','GET',undefined,testUser,''),env);assert.equal(script.status,200);assert.equal(await script.text(),groupInviteScript);
 const token='b'.repeat(64),writes=[],locations=[];
 vm.runInNewContext(groupInviteScript,{URLSearchParams,Date,location:{hash:`#invite=${token}`,pathname:'/50pubs',search:''},history:{state:null,replaceState:(_s,_t,url)=>locations.push(url)},sessionStorage:{setItem:(k,v)=>writes.push([k,JSON.parse(v)])}});
 assert.deepEqual(locations,['/50pubs']);assert.equal(writes[0][1].token,token);assert.ok(writes[0][1].expiresAt>Date.now());assert.ok(writes[0][1].expiresAt<=Date.now()+600000);
 assert.doesNotMatch(groupInviteScript,/fetch\(|XMLHttpRequest|sendBeacon|localStorage/);
});

test('partial person updates preserve concurrently changed omitted fields',async()=>{
 const {env,groupId}=await setup();
 const original=env.DB.prepare.bind(env.DB);let change='origin';
 env.DB.prepare=sql=>{
  if(sql.startsWith('UPDATE group_people SET\n')) {
   if(change==='origin') env.DB.sqlite.prepare('UPDATE group_people SET origin_id=NULL WHERE person_id=?').run('p1');
   if(change==='name') env.DB.sqlite.prepare('UPDATE group_people SET name=? WHERE person_id=?').run('Concurrent name','p1');
   change=null;
  }
  return original(sql);
 };
 let response=await worker.fetch(req(`/api/groups/${groupId}/people/p1`,'PATCH',{name:'Renamed'}),env);
 assert.equal(response.status,200);let value=await response.json();assert.equal(value.person.name,'Renamed');assert.equal(value.person.originId,null);
 change='name';response=await worker.fetch(req(`/api/groups/${groupId}/people/p1`,'PATCH',{originId:'home'}),env);
 assert.equal(response.status,200);value=await response.json();assert.equal(value.person.name,'Concurrent name');assert.equal(value.person.originId,'home');
});

test('malformed bodies and unsupported methods are rejected safely',async()=>{
 const {env,groupId}=await setup();
 const headers={Cookie:testCookie,Origin:'https://example.test','X-Equidistant-Account':testUser.id,'Content-Type':'application/json'};
 for(const body of ['{','[]','null']) assert.equal((await worker.fetch(new Request('https://example.test/api/groups',{method:'POST',headers,body}),env)).status,400);
 assert.equal((await worker.fetch(req(`/api/groups/${groupId}/availability`,'PATCH',{personId:'p1',date:'2026-10-30',status:'free'},testUser,testCookie,{'Content-Type':'text/plain'}),env)).status,415);
 assert.equal((await worker.fetch(req('/api/groups','DELETE'),env)).status,405);
 assert.equal((await worker.fetch(req(`/api/groups/${groupId}/people/absent`,'PATCH',{name:'New'}),env)).status,404);
});

test('invitation creation and joining have bounded per-account rates',async()=>{
 const {env,groupId,other,cookie}=await setup();
 for(let i=0;i<10;i++) assert.equal((await worker.fetch(req(`/api/groups/${groupId}/invite`,'POST'),env)).status,200);
 const limited=await worker.fetch(req(`/api/groups/${groupId}/invite`,'POST'),env);assert.equal(limited.status,429);assert.ok(Number(limited.headers.get('Retry-After'))>0);
 for(let i=0;i<20;i++) assert.equal((await worker.fetch(req('/api/groups/join','POST',{token:'c'.repeat(64)},other,cookie),env)).status,404);
 assert.equal((await worker.fetch(req('/api/groups/join','POST',{token:'c'.repeat(64)},other,cookie),env)).status,429);
});
