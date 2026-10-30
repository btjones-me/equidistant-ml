import { useEffect, useRef, useState } from 'react';
import { ArrowLeft, ExternalLink, MapPin, Search, X } from 'lucide-react';
import type { Friend } from './types';
import type { PubDataset, PubImage, RankedPub } from './lib/fiftyPubs';

const minutes=(n:number|null)=>n===null?'Not covered':`~${Math.round(n)} min`;
export function FiftyPubsRecommendations({pubs,loading,error,onSelect,onBrowse}:{pubs:RankedPub[];loading:boolean;error:string|null;onSelect:(id:string)=>void;onBrowse:()=>void}){
  const choices=pubs.filter(p=>p.score!==null).slice(0,3);
  return <section className="result-card fifty-recommendations" aria-label="Recommended pubs" aria-busy={loading}>
    <div className="result-card-heading">Best pubs for your group</div>
    {error?<p role="alert" className="product-error">{error}</p>:choices.length?<div className="suggestion-list">{choices.map((pub,i)=><button type="button" key={pub.id} onClick={()=>onSelect(pub.id)}><span>{i+1}</span><span><strong>{pub.name}</strong><small>Time Out #{pub.rank} · {pub.neighbourhood}</small></span><b>{minutes(pub.average)}<small>average</small></b></button>)}</div>:<p className="fifty-note">{loading?'Comparing the 50 pubs…':'No covered pubs for these locations.'}</p>}
    <button type="button" className="venue-cta" onClick={onBrowse}>Explore all {pubs.length||50} pubs <span aria-hidden="true">→</span></button>
    <p className="fifty-note">{pubs.filter(p=>p.score!==null).length} of {pubs.length || 50} pubs within the model’s coverage. Sorted by travel strategy, not editorial rank. Estimates, not live routes.</p>
  </section>;
}

function PubPhoto({photo,name}:{photo:PubImage;name:string}){
 const [failed,setFailed]=useState(false);
 return <figure className="fifty-photo">{failed?<div className="fifty-photo-fallback"><MapPin size={24}/><span>Photo unavailable</span></div>:<img src={photo.url} alt={`${name} — pub photograph`} loading="lazy" onError={()=>setFailed(true)}/>}
 <figcaption><a href={photo.sourceUrl} target="_blank" rel="noreferrer">{photo.credit || 'Photo source'}</a>{photo.license ? <> · {photo.licenseUrl ? <a href={photo.licenseUrl} target="_blank" rel="noreferrer">{photo.license}</a> : photo.license}</> : null} · cropped to fit</figcaption></figure>;
}

export function FiftyPubsDrawer({open,pubs,dataset,activeId,friends,included,loading,onSelect,onClose}:{open:boolean;pubs:RankedPub[];dataset:PubDataset|null;activeId:string|null;friends:Friend[];included:boolean[];loading:boolean;onSelect:(id:string|null)=>void;onClose:()=>void}){
 const [query,setQuery]=useState('');
 const [sort,setSort]=useState<'fit'|'editorial'>('fit');
 const closeRef=useRef<HTMLButtonElement>(null);
 const active=pubs.find(p=>p.id===activeId);
 useEffect(()=>{if(!open)return; const previous=document.activeElement as HTMLElement|null;closeRef.current?.focus();const esc=(e:KeyboardEvent)=>{if(e.key==='Escape'){e.preventDefault();onClose();}};window.addEventListener('keydown',esc);return()=>{window.removeEventListener('keydown',esc);previous?.focus();};},[open,onClose]);
 if(!open)return null;
 const matches=pubs.filter(p=>`${p.name} ${p.neighbourhood} ${p.address}`.toLowerCase().includes(query.toLowerCase()));
 const list=sort==='editorial'?[...matches].sort((a,b)=>a.rank-b.rank):matches;
 return <aside className="venue-drawer fifty-drawer" aria-label={active?`${active.name} details`:'The 50 pubs'}>
 <header className="venue-drawer-header"><div><span className="fifty-eyebrow">EQUIDISTANT COLLECTION</span><h2>{active?'The pub':'50 pubs. One good plan.'}</h2></div><button ref={closeRef} type="button" className="small-icon-button" onClick={onClose} aria-label="Close pub details"><X size={19}/></button></header>
 <div className="fifty-content">{active?<>
 <button type="button" className="fifty-back" onClick={()=>onSelect(null)}><ArrowLeft size={15}/> All 50 pubs</button>
 <span className="fifty-rank">#{active.rank} in Time Out’s list</span><h2 className="fifty-pub-title">{active.name}</h2><p className="fifty-area">{active.neighbourhood}</p>
 {active.images.length?<div className={`fifty-gallery ${active.images.length===1?'single':''}`}>{active.images.slice(0,3).map((image,i)=><PubPhoto key={`${active.id}-${i}`} photo={image} name={active.name}/>)}</div>:<p className="fifty-note">No verified reusable photo is available.</p>}
 <p className="fifty-address"><MapPin size={16}/><span>{active.address}{active.postcode&&!active.address.includes(active.postcode)?`, ${active.postcode}`:''}</span></p>
 <p className="fifty-description">{active.summary}</p>
 {active.tags?.length?<div className="fifty-tags">{active.tags.map(tag=><span key={tag}>{tag}</span>)}</div>:null}
 <section className="fifty-journeys" aria-busy={loading}><h3>Your journeys</h3>{active.score===null?<p>Outside the current travel-time model coverage. The map pin still marks the venue’s location.</p>:<><div className="fifty-metrics"><span><strong>{minutes(active.average)}</strong> group average</span><span><strong>{minutes(active.longest)}</strong> longest trip</span></div>{friends.map((friend,i)=><div className={`fifty-journey ${included[i]?'':'excluded'}`} key={friend.id}><span>{friend.name}{included[i]?'':' · excluded'}</span><strong>{minutes(active.journeys[i])}</strong></div>)}<p className="fifty-note">Estimated from nearby cells in Equidistant’s offline model. These are not live itineraries or exact door-to-door timings.</p></>}</section>
 <div className="fifty-links"><a href={active.sourceUrl||dataset?.article.url} target="_blank" rel="noreferrer">Read the full Time Out review <ExternalLink size={14}/></a><a href={`https://www.google.com/maps/search/?api=1&query=${active.lat},${active.lng}`} target="_blank" rel="noreferrer">Open location in Maps <ExternalLink size={14}/></a><a href={active.addressSourceUrl} target="_blank" rel="noreferrer">Venue details source <ExternalLink size={14}/></a></div>
 <p className="fifty-note">Original concise summary; Time Out’s full description stays on its website. Photos may show earlier décor or signage. Pins identify the venue building, not a surveyed entrance. Check opening hours with the pub before travelling.</p>
 </>:<>
 <p className="fifty-description">Time Out’s London shortlist, placed on your map. Choose a pub to see its story, photos and your group’s estimated journeys.</p>
 <div className="fifty-search"><Search size={17}/><input aria-label="Search the 50 pubs" placeholder="Find a pub or neighbourhood" value={query} onChange={e=>setQuery(e.target.value)}/></div>
 <label className="fifty-sort">Sort by <select value={sort} onChange={e=>setSort(e.target.value as 'fit'|'editorial')}><option value="fit">Best fit for your group</option><option value="editorial">Time Out rank</option></select></label>
 <p className="fifty-note">{list.length} pubs · pin numbers are Time Out ranks</p>
 <div className="fifty-list">{list.map(pub=><button type="button" key={pub.id} onClick={()=>onSelect(pub.id)}><span className="fifty-list-rank">{pub.rank}</span><span><strong>{pub.name}</strong><small>{pub.neighbourhood}</small></span><b>{minutes(pub.average)}</b></button>)}</div>
 {!list.length?<p>No pubs match that search.</p>:null}
 </>}
 <footer className="fifty-source">Source: <a href={dataset?.article.url} target="_blank" rel="noreferrer">{dataset?.article.title||'Time Out — The 50 best pubs in London'}</a>. {dataset?.article.checkedAt?`Checked ${new Date(dataset.article.checkedAt).toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric'})}.`:''} An independent collection, not affiliated with Time Out.</footer>
 </div></aside>;
}
