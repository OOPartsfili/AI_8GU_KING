// Inspect every rendered formula, including actual scroll endpoints.
const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert');
const {pathToFileURL}=require('url');
let chromium;try{({chromium}=require('playwright'));}catch{({chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')));}
const root=path.resolve(__dirname,'..');
const output=process.env.KNOWLEDGE_QA_OUTPUT||path.join(os.tmpdir(),'llm-formula-qa');
fs.mkdirSync(output,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.KNOWLEDGE_BROWSER||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1440,height:1080}});
 const errors=[],network=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
 const url=pathToFileURL(path.join(root,'开始阅读.html')).href;
 await page.goto(url);
 const db=await page.evaluate(()=>DB);
 const entries=[...db.questions.map(q=>({...q,route:'q-'+q.id})),
                ...db.documents.map(d=>({...d,route:'doc-'+d.id}))].filter(x=>x.html.includes('<math'));
 const rows=[],gallery=[];
 for(const entry of entries){
  await page.evaluate(route=>location.hash=route,entry.route);
  await page.waitForFunction(route=>location.hash==='#'+route,entry.route);
  await page.waitForFunction(id=>document.querySelector('#q-'+id)||document.querySelector('.article'),entry.id);
  await page.selectOption('#depth','full');
  // Only direct article formulas for documents; their question formulas are tested by q routes.
  const selector=entry.route.startsWith('q-')?'#q-'+entry.id+' math':'.article math';
  await page.locator(selector).first().waitFor({state:'visible'});
  for(const width of [1440,768,390,320]){
   await page.setViewportSize({width,height:1080});
   const measured=await page.locator(selector).evaluateAll(nodes=>nodes.map((math,index)=>{
    const box=math.closest('.formula')||math.parentElement;
    const b=box.getBoundingClientRect(),style=getComputedStyle(box);
    const leaves=()=>[...math.querySelectorAll('mi,mn,mo,mtext')].map(e=>e.getBoundingClientRect()).filter(r=>r.width&&r.height);
    box.scrollLeft=0;const start=leaves();
    const left=Math.min(...start.map(r=>r.left));
    box.scrollLeft=box.scrollWidth-box.clientWidth;const end=leaves();
    const right=Math.max(...end.map(r=>r.right));
    const top=Math.min(...start.map(r=>r.top)),bottom=Math.max(...start.map(r=>r.bottom));
    const result={index,text:math.textContent,
      scrolls:box.scrollWidth>box.clientWidth+1,
      leftHidden:left<b.left+parseFloat(style.borderLeftWidth)-1.5,
      rightHidden:right>b.right-parseFloat(style.borderRightWidth)+1.5,
      verticallyHidden:top<b.top-1.5||bottom>b.bottom+1.5,
      unknownTokens:/\\[A-Za-z]+|MATHPLACEHOLDER|\ufffd|\$\$/.test(math.textContent),
      errorNodes:math.querySelectorAll('merror').length,
      pageOverflow:document.documentElement.scrollWidth>innerWidth+2};
    box.scrollLeft=0;return result;
   }));
   rows.push(...measured.map(m=>({entry:entry.id,width,...m})));
   if(width===1440){
    const formulas=await page.locator(selector).evaluateAll(nodes=>nodes.map(n=>n.outerHTML));
    gallery.push(...formulas.map((html,index)=>({label:entry.id+' · '+(index+1),html})));
   }
  }
 }
 assert.equal(gallery.length,db.stats.math_count,'all source formulas covered');
 const problems=rows.filter(r=>r.leftHidden||r.rightHidden||r.verticallyHidden||r.unknownTokens||r.errorNodes||r.pageOverflow);
 const style=await page.locator('style').textContent();
 // Gallery uses the same embedded CSS and native MathML renderer.
 for(let i=0;i<gallery.length;i+=10){
  await page.setViewportSize({width:1120,height:1000});
  await page.setContent('<!doctype html><meta charset="utf-8"><style>'+style+
   'body{padding:24px}h1{font-size:24px;margin:0}section{margin:9px 0}.formula{margin:4px 0;padding:14px 12px}.label{font-size:12px;color:#52646b}</style>'+
   '<h1>公式逐项审阅 '+(i+1)+'—'+Math.min(i+10,gallery.length)+'</h1>'+
   gallery.slice(i,i+10).map(x=>'<section><div class="label">'+x.label+'</div><div class="formula">'+x.html+'</div></section>').join(''));
  await page.screenshot({path:path.join(output,'formula-gallery-'+String(i/10+1).padStart(2,'0')+'.png'),fullPage:true});
 }
 await page.goto('about:blank');
 await page.goto(url+'#q-KD03');
 await page.selectOption('#depth','full');
 await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:path.join(output,'distillation-mobile.png'),fullPage:true});
 const report={version:db.version,formula_count:gallery.length,widths:[1440,768,390,320],render_checks:rows.length,
               problems,javascript_errors:errors,automatic_http_requests:network,
               scope:'Native MathML in headless Edge; all formulas, actual horizontal scroll endpoints, tokens and page bounds. Gallery images require visual review.',rows};
 fs.writeFileSync(path.join(root,'90_维护与来源/formula-qa.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify({formula_count:gallery.length,render_checks:rows.length,problems,javascript_errors:errors,automatic_http_requests:network},null,2));
 assert.deepEqual(problems,[]);assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
