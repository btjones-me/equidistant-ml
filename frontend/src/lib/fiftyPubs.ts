import { combineValues } from './atlas';
import type { CombineMode, CoverageBounds, SurfaceCell, VenueRecommendation } from '../types';

export type PubImage = { url: string; sourceUrl: string; credit: string; license: string; licenseUrl?: string; date?: string; caption?: string };
export type FiftyPub = { id: string; rank: number; name: string; neighbourhood: string; address: string; postcode: string; lat: number; lng: number; summary: string; sourceUrl: string; addressSourceUrl: string; coordinateSourceUrl: string; images: PubImage[]; tags: string[] };
export type PubDataset = { article: { title: string; url: string; checkedAt: string }; pubs: FiftyPub[] };
export type RankedPub = FiftyPub & { journeys: (number | null)[]; score: number | null; average: number | null; longest: number | null; sampledDistanceMetres: number | null };

export function distanceMetres(a: {lat:number;lng:number}, b: {lat:number;lng:number}) {
  const rad = Math.PI / 180;
  const dLat = (b.lat-a.lat)*rad, dLng=(b.lng-a.lng)*rad;
  const h = Math.sin(dLat/2)**2+Math.cos(a.lat*rad)*Math.cos(b.lat*rad)*Math.sin(dLng/2)**2;
  return 6371000*2*Math.atan2(Math.sqrt(h),Math.sqrt(1-h));
}

export function rankPubs(pubs: FiftyPub[], cells: SurfaceCell[], friendCount: number, included: number[], combine: CombineMode, bounds?: CoverageBounds): RankedPub[] {
  return pubs.map(pub => {
    const within = !bounds || (pub.lat>=bounds.south && pub.lat<=bounds.north && pub.lng>=bounds.west && pub.lng<=bounds.east);
    const nearest = within ? cells.map(cell=>({cell,distance:distanceMetres(pub,cell)})).sort((a,b)=>a.distance-b.distance).filter(x=>x.distance<=1500).slice(0,3) : [];
    // The marker is the verified venue coordinate; travel times remain atlas estimates.
    // Never extrapolate a remote venue from an unrelated edge cell.
    const samples = nearest[0]?.distance <= 1500 ? (nearest[0].distance<20 ? nearest.slice(0,1) : nearest) : [];
    const journeys = Array.from({length:friendCount},(_,i)=>{
      if(!samples.length) return null;
      const values=samples.map(({cell,distance})=>({value:cell[`friend_${i}_model_minutes`],weight:1/Math.max(distance,25)**2}));
      if(values.some(x=>typeof x.value!=='number'||!Number.isFinite(x.value))) return null;
      return values.reduce((s,x)=>s+(x.value as number)*x.weight,0)/values.reduce((s,x)=>s+x.weight,0);
    });
    const values=included.map(i=>journeys[i]);
    const valid=values.length>0&&values.every((v):v is number=>typeof v==='number'&&Number.isFinite(v));
    return {...pub,journeys,score:valid?combineValues(values,combine):null,average:valid?values.reduce((s,v)=>s+v,0)/values.length:null,longest:valid?Math.max(...values):null,sampledDistanceMetres:samples.length?samples[0].distance:null};
  }).sort((a,b)=>(a.score??Infinity)-(b.score??Infinity)||a.rank-b.rank);
}

export function asMapVenue(pub: FiftyPub): VenueRecommendation {
  return {place_id:pub.id,name:`#${pub.rank} · ${pub.name}`,address:pub.address,lat:pub.lat,lng:pub.lng,primary_type:'pub',rating:null,user_rating_count:null,price_level:null,open_now:null,opening_summary:null,website_url:null,google_maps_url:`https://www.google.com/maps/search/?api=1&query=${pub.lat},${pub.lng}`,photo_url:pub.images[0]?.url??null,photo_attribution:null,why:pub.summary,verified_details:[],source_urls:[pub.sourceUrl],marker_label:pub.rank};
}

export function validatePubDataset(data: unknown): PubDataset {
  const value = data as PubDataset;
  if(!value?.article?.url || !Array.isArray(value.pubs) || value.pubs.length!==50) throw new Error('The 50-pub collection is unavailable. Please try again.');
  const ids=new Set<string>(),ranks=new Set<number>();
  for(const pub of value.pubs){
    if(!pub.id||ids.has(pub.id)||!Number.isInteger(pub.rank)||pub.rank<1||pub.rank>50||ranks.has(pub.rank)||!pub.name||!pub.address||!Number.isFinite(pub.lat)||!Number.isFinite(pub.lng)||pub.lat<51.2||pub.lat>51.8||pub.lng<-.7||pub.lng>.4||!Array.isArray(pub.images)) throw new Error('The pub collection could not be verified.');
    ids.add(pub.id); ranks.add(pub.rank);
  }
  return value;
}
