import googleLogo from "./google-logo.js";
import { cells } from "./landing-geometry.js";

// This script is served on one explicit public route. It has no network or storage calls.
export const landingScript = `const cells=${JSON.stringify(cells)};
const canvas=document.getElementById('london');
const ctx=canvas.getContext('2d');
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
const places=[['Camden',-.1425,51.5393],['Soho',-.135,51.514],['Shoreditch',-.078,51.524],['Brixton',-.1149,51.4626],['Bermondsey',-.063,51.498],['Hammersmith',-.223,51.492],['Greenwich',-.01,51.481]];
const xy=(lng,lat)=>[(lng+.335)*2000,(51.59-lat)*3200];
let active=null,ripple=null,frame=0;
const paths=cells.map(cell=>{const p=new Path2D();cell.p.forEach((v,i)=>i?p.lineTo(...v):p.moveTo(...v));p.closePath();return p;});
function paint(now=0){
 frame=0; const ratio=Math.min(devicePixelRatio||1,2),w=canvas.clientWidth,h=canvas.clientHeight;
 if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}
 ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);
 const scale=Math.max(w/900,h/550),ox=(w-900*scale)/2,oy=(h-550*scale)/2;
 ctx.setTransform(ratio*scale,0,0,ratio*scale,ratio*ox,ratio*oy);
 const age=ripple?(now-ripple.start)/750:2;
 cells.forEach((cell,i)=>{const [x,y]=cell.c; const d=active?Math.hypot(x-active[0],y-active[1]):999;
 const ring=ripple&&age<1?Math.abs(Math.hypot(x-ripple.x,y-ripple.y)-age*260):999;
 const tone=(Math.sin(x*127.1+y*311.7)*43758.5453)%1;
 const base=Math.abs(tone);
 ctx.fillStyle=d<45?'#087f73':ring<15?'#d8a747':base>.85?'#bfd2c4':base>.45?'#d3dfd3':'#e0e7dc';
 ctx.strokeStyle='#f4f5ee';ctx.lineWidth=1.5;ctx.fill(paths[i]);ctx.stroke(paths[i]);});
 ctx.font='500 14px system-ui';ctx.textAlign='center';
 places.forEach(([name,lng,lat])=>{const [x,y]=xy(lng,lat);ctx.fillStyle='#f4f5ee';ctx.fillRect(x-ctx.measureText(name).width/2-8,y-14,ctx.measureText(name).width+16,24);ctx.fillStyle='#263f35';ctx.fillText(name,x,y+3);});
 if(ripple&&age<1&&!reduced)frame=requestAnimationFrame(paint);
}
function schedule(){if(!frame)frame=requestAnimationFrame(paint);}
function locate(event){const b=canvas.getBoundingClientRect();const scale=Math.max(b.width/900,b.height/550);return [(event.clientX-b.left-(b.width-900*scale)/2)/scale,(event.clientY-b.top-(b.height-550*scale)/2)/scale];}
canvas.addEventListener('pointermove',event=>{if(event.pointerType==='mouse'){active=locate(event);schedule();}});
canvas.addEventListener('pointerleave',()=>{active=null;schedule();});
canvas.addEventListener('click',event=>{active=locate(event);ripple=reduced?null:{x:active[0],y:active[1],start:performance.now()};schedule();});
document.querySelectorAll('[data-place]').forEach(button=>button.addEventListener('click',()=>{
 const [name,lng,lat]=places[Number(button.dataset.place)];active=xy(lng,lat);ripple=reduced?null:{x:active[0],y:active[1],start:performance.now()};
 document.getElementById('map-note').textContent=name;schedule();
}));
new ResizeObserver(schedule).observe(canvas);schedule();`;

export function landingPage({ error = false, unavailable = false } = {}) {
  return `<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Find a place to meet | Equidistant</title>
<meta name="description" content="Compare estimated public transport travel times for 2–6 people and find somewhere to meet. Currently in beta with coverage in central and inner London.">
<link rel="canonical" href="https://equidistant.me/">
<link rel="icon" type="image/png" href="/favicon.png"><meta property="og:image" content="https://equidistant.me/og.png"><meta name="twitter:card" content="summary_large_image"><meta name="twitter:image" content="https://equidistant.me/og.png"><meta name="theme-color" content="#f4f5ee"><meta property="og:type" content="website"><meta property="og:title" content="Find a place to meet | Equidistant"><meta property="og:description" content="Compare estimated public transport travel times and find somewhere to meet. Currently in beta in London."><meta property="og:url" content="https://equidistant.me/">
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#203b30;background:#f4f5ee;font-synthesis:none}*{box-sizing:border-box}body{margin:0}a{color:inherit}button,a{-webkit-tap-highlight-color:transparent}a:focus-visible,button:focus-visible{outline:3px solid #087f73;outline-offset:5px}.skip{position:absolute;left:20px;top:-100px}.skip:focus{top:16px;z-index:5;background:white;padding:12px}.wrap{max-width:1440px;margin:auto;padding:0 5vw}header{height:100px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #d6dfd5}.brand{font-size:23px;font-weight:650;text-decoration:none;letter-spacing:-1px}.brand span{color:#087f73;font-size:32px;vertical-align:-2px;margin-right:8px}nav{display:flex;gap:28px;align-items:center;font-size:14px}nav a{text-decoration:none}.hero{display:grid;grid-template-columns: .9fr 1.25fr;gap:20px;align-items:center;min-height:650px;padding:60px 0}.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:13px;font-weight:650;color:#537364}.hero h1{font-family:Georgia,serif;font-size:clamp(48px,5.2vw,76px);font-weight:400;letter-spacing:-3px;line-height:1.04;margin:24px 0}.hero h1 em{color:#087f73;font-weight:400}.intro{font-size:19px;line-height:1.65;max-width:430px;color:#51665a}.signin{display:inline-flex;align-items:center;justify-content:center;gap:12px;border:1px solid #747775;border-radius:5px;background:white;color:#1f1f1f;padding:15px 23px;font:500 15px Arial,sans-serif;text-decoration:none;margin-top:18px;min-height:48px}.signin img{width:20px;height:20px}.small{font-size:14px;line-height:1.6;color:#587064;max-width:420px}.map{min-width:0;position:relative}.map-top{display:flex;justify-content:space-between;font-size:12px;letter-spacing:1.8px;color:#587064;text-transform:uppercase;padding:0 20px}canvas{width:100%;aspect-ratio:900/550;display:block;cursor:crosshair}.map-footer{text-align:center}.map-footer p{font-size:14px;color:#587064;min-height:23px}.places{display:flex;flex-wrap:wrap;gap:8px;justify-content:center}.places button{background:transparent;border:1px solid #bdcfc3;border-radius:30px;padding:10px 15px;color:#315446;font:inherit;font-size:14px;cursor:pointer}.places button:hover{background:#dce8df}.caption{font-size:12px!important;color:#637b6d!important}.how{border-top:1px solid #d6dfd5;padding:55px 0}.section-heading{font-family:Georgia,serif;font-weight:400;font-size:36px;margin:0 0 32px;letter-spacing:-1px}.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:42px}.step-number{color:#087f73;font-size:14px}.steps h3{font-size:19px;font-weight:600}.steps p,.faq p{font-size:16px;color:#51665a;line-height:1.75;max-width:55ch}.faq{display:grid;grid-template-columns:1fr 1.5fr;gap:50px;padding:42px 0 70px;border-top:1px solid #d6dfd5}.faq details{border-bottom:1px solid #d6dfd5;padding:19px 0}.faq summary{cursor:pointer;font-size:17px;font-weight:550;line-height:1.5}.faq details:first-child{padding-top:0}footer{border-top:1px solid #d6dfd5;padding:26px 0 38px;display:flex;justify-content:space-between;gap:20px;font-size:14px;color:#587064}.error{color:#9c382c;font-size:16px}.hero-copy{position:relative;z-index:1}
@media(max-width:850px){.hero{grid-template-columns:1fr;padding:40px 0;gap:35px;min-height:0}.hero h1{font-size:60px;max-width:600px}.intro{max-width:540px}.map{max-width:700px;width:100%;margin:auto}.steps{gap:24px}.faq{grid-template-columns:1fr;gap:10px}header{height:80px}nav{gap:16px}.wrap{padding:0 6vw}}@media(max-width:520px){nav a:first-child{display:none}.hero h1{font-size:50px;letter-spacing:-2px}.intro{font-size:17px}.steps{grid-template-columns:1fr;gap:12px}.how{padding:36px 0}.map-top{font-size:10px;padding:0}.places button{font-size:13px;min-height:44px}.section-heading{font-size:30px}.map-footer p{font-size:13px}.brand{font-size:20px}.hero{gap:35px}footer{flex-wrap:wrap}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}

/* The map is an interactive backdrop; its mask leaves a quiet reading area. */
.hero{position:relative;isolation:isolate;display:flex;min-height:720px;padding:80px 0 150px;overflow:hidden;overflow:clip;margin:0 -5vw;padding-left:5vw;padding-right:5vw}
.hero-copy{max-width:520px;pointer-events:none}.hero-copy a,.hero-copy .error{pointer-events:auto}
.hero h1{font-size:clamp(60px,6.2vw,88px);max-width:520px}.intro{max-width:410px}
.map{position:absolute;inset:0;max-width:none;width:auto;margin:0;z-index:-1}
.map canvas{height:100%;aspect-ratio:auto;mask-image:linear-gradient(90deg,transparent 3%,rgba(0,0,0,.06) 25%,rgba(0,0,0,.4) 45%,#000 67%);-webkit-mask-image:linear-gradient(90deg,transparent 3%,rgba(0,0,0,.06) 25%,rgba(0,0,0,.4) 45%,#000 67%)}
.map-top{position:absolute;right:5vw;top:30px;z-index:1;display:block;padding:0;letter-spacing:.5px;text-transform:none}.map-top span:first-child{display:none}
.map-footer{position:absolute;right:5vw;bottom:24px;max-width:480px;z-index:1;background:linear-gradient(0deg,#f4f5ee 65%,transparent);padding:20px 12px 0}
.map-footer p{margin:8px 0}.map-footer #map-note:empty{display:none}.places button{background:#f4f5eedb}
.map noscript{position:absolute;right:5vw;top:55px;max-width:300px}
@media(max-width:850px){.hero{min-height:850px;margin:0 -6vw;padding:50px 6vw 400px;align-items:flex-start}.hero h1{font-size:64px;max-width:520px}.intro{max-width:480px}.map canvas{mask-image:linear-gradient(180deg,transparent 0%,rgba(0,0,0,.08) 35%,#000 68%);-webkit-mask-image:linear-gradient(180deg,transparent 0%,rgba(0,0,0,.08) 35%,#000 68%)}.map-top{top:auto;bottom:345px;right:6vw}.map-footer{right:6vw;left:6vw;bottom:20px;max-width:none}.map noscript{top:auto;bottom:260px}}
@media(max-width:520px){.hero{min-height:870px;padding-bottom:380px}.hero h1{font-size:58px}.intro{max-width:350px}.map-top{bottom:320px;font-size:11px}.map-footer{padding:16px 0 0}.places{gap:6px}}
</style><script src="/welcome-motion.js" defer></script></head><body>
<a class="skip" href="#main">Skip to content</a><div class="wrap"><header><a href="/" class="brand"><span aria-hidden="true">◎</span>Equidistant</a><nav aria-label="Main"><a href="#how-it-works">How it works</a><a href="#start">Sign in →</a></nav></header>
<main id="main"><section class="hero" aria-labelledby="headline"><div class="hero-copy"><p class="eyebrow">London beta</p><h1 id="headline">Find a place<br>to <em>meet.</em></h1><p class="intro">Add where everyone’s coming from. Equidistant compares estimated public transport travel times to help you choose an area, then find pubs, restaurants and things to do nearby.</p>
<div id="start">${unavailable ? '<p class="error" role="status">Sign-in is temporarily unavailable. Please try again later.</p>' : `<a class="signin" href="/auth/google/start"><img src="${googleLogo}" alt="" width="20" height="20"><span>Sign in with Google</span></a>`}</div>${error ? '<p class="error" role="alert">Sign-in did not finish. Please try again.</p>' : ''}
<p class="small">For groups of 2–6 people.</p></div>
<div class="map"><div class="map-top"><span>London</span><span>Hover or tap the hexagons</span></div><noscript><p class="small">Enable JavaScript to play with the London map. You can still read about Equidistant and sign in below.</p></noscript><canvas id="london" width="900" height="550" aria-label="Interactive hexagon map of London. Use the neighbourhood buttons below for keyboard interaction.">A decorative London map. Try a neighbourhood below.</canvas><div class="map-footer"><div class="places" aria-label="Explore London neighbourhoods"><button type="button" data-place="0">Camden</button><button type="button" data-place="1">Soho</button><button type="button" data-place="2">Shoreditch</button><button type="button" data-place="3">Brixton</button></div><p id="map-note" aria-live="polite"></p><p class="caption">This map is a demo. Sign in to plan a meeting.</p></div></div></section>
<section id="how-it-works" class="how"><h2 class="section-heading">How it works</h2><div class="steps"><div><span class="step-number">01</span><h3>Add everyone’s starting point</h3><p>Choose locations within the current beta coverage area.</p></div><div><span class="step-number">02</span><h3>Compare travel times</h3><p>Look for similar journey times for everyone, reduce the longest journey, or minimise the group’s total travel time.</p></div><div><span class="step-number">03</span><h3>Find somewhere to go</h3><p>Search for pubs, restaurants and things to do around the area you choose.</p></div></div></section>
<section class="faq"><h2 class="section-heading">Common questions</h2><div><details><summary>Where does Equidistant work?</summary><p>The beta currently covers central London and parts of inner London. Sign in to see the coverage boundary on the map. Other cities and the rest of Greater London are not currently covered.</p></details><details><summary>Is halfway the same as a fair journey?</summary><p>Not always. Stations, connections and the transport network can make two similar distances very different trips. Equidistant compares estimated public-transport travel times rather than straight-line distance.</p></details><details><summary>Are these live journey times?</summary><p>No. The map uses estimated travel times, without live service updates. Check current routes and venue opening times before travelling.</p></details><details><summary>Why do I need to sign in?</summary><p>You need a Google account to use the planner and search for venues. This helps us limit misuse during the beta. Your group is saved in this browser, separately for each account. We use your Google name and email, and do not request access to Gmail, contacts or files. Read our <a href="/privacy">privacy notice</a>.</p></details></div></section></main><footer><span>◎ Equidistant</span><a href="/privacy">Privacy</a></footer></div></body></html>`;
}
