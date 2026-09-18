const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert');
const {pathToFileURL}=require('url');
const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const root=path.resolve(__dirname,'..'),work=path.join(root,'work/transformer-visual-revision'),shots=path.join(work,'qa');
fs.mkdirSync(shots,{recursive:true});
const file=path.join(root,'开始阅读.html'),url=pathToFileURL(file).href;
const median=a=>[...a].sort((x,y)=>x-y)[Math.floor(a.length/2)];
const types=['projection','attention','heads','block','rope','cache','probability','lora'];
const near=(a,b,eps=.001)=>assert(Math.abs(a-b)<eps,`${a} != ${b}`);
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const report={version:'1.5',visuals:[],errors:[],network:[],widths:[1440,768,390,320],performance:{}};
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1080},acceptDownloads:true}),page=await context.newPage();
  page.on('pageerror',e=>report.errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))report.network.push(r.url())});
  await page.goto(url);await page.waitForSelector('.hero');
  assert.equal(await page.locator('[data-mounted]').count(),0);
  assert.equal(await page.evaluate(()=>codeAssets),null);
  await page.locator('#transformerBtn').click();await page.waitForSelector('#q-TH01');
  assert.equal(await page.locator('.question').count(),22);assert.equal(await page.locator('.qbody').count(),0);
  const download=page.waitForEvent('download');await page.locator('[data-code="transformer_handwrite.py"]').click();
  const d=await download,saved=path.join(work,'downloaded-transformer.py');await d.saveAs(saved);
  assert.equal(fs.readFileSync(saved,'utf8'),fs.readFileSync(path.join(root,'03_代码实验/transformer_handwrite.py'),'utf8'));
  report.download_matches_source=true;
  await page.goto(url+'#q-TH04');await page.waitForSelector('.viz-panel');
  assert.equal(await page.locator('[data-mounted]').count(),0);
  await page.locator('.viz-panel>summary').focus();await page.keyboard.press('Enter');
  await page.waitForSelector('[data-mounted="attention"]');
  assert.equal(await page.locator('.viz-host[data-mounted]').count(),1);
  report.lazy_and_keyboard=true;
  for(const width of report.widths){
   await page.setViewportSize({width,height:950});
   for(const type of types){
    await page.goto(url+'#viz-'+type);await page.waitForSelector('[data-mounted="'+type+'"]');
    assert.equal(await page.locator('[data-mounted]').count(),1);
    for(const control of await page.locator('.viz-controls input[type=range]').all()){
     await control.evaluate(el=>{el.value=el.max;el.dispatchEvent(new Event('input',{bubbles:true}));});
     const key=await control.getAttribute('data-key');
     assert.equal(await page.locator('output[data-output="'+key+'"]').textContent(),await control.inputValue());
    }
    for(const control of await page.locator('.viz-controls select').all()){
     const value=await control.locator('option').last().getAttribute('value');await control.selectOption(value);
    }
    for(const control of await page.locator('.viz-controls input[type=checkbox]').all()){await control.uncheck();await control.check();}
    if(await page.locator('[data-cell]').count()){await page.locator('[data-cell]').last().click();assert.equal(await page.locator('[data-cell][aria-pressed=true]').count(),1);}
    const problem=await page.evaluate(()=>({overflow:document.documentElement.scrollWidth>innerWidth+1,
     badText:/\bNaN\b|undefined|Infinity/.test(document.querySelector('.viz-host').innerText),
     controls:[...document.querySelectorAll('.viz-controls input,.viz-controls select')].some(el=>{const r=el.getBoundingClientRect();return r.right>innerWidth+1||r.left<0;})}));
    assert.deepEqual(problem,{overflow:false,badText:false,controls:false},type+' '+width);
    await page.locator('.viz-host').screenshot({path:path.join(shots,`${type}-${width}.png`)});
    report.visuals.push({type,width,checks:'controls, selection, finite output, overflow'});
   }
  }
  await page.setViewportSize({width:1440,height:1080});
  // Independent expected values, checked from rendered outputs.
  await page.goto(url+'#viz-projection');await page.waitForSelector('[data-cell="0,0"]');
  near(Number(await page.locator('[data-cell="0,0"]').textContent()),6);
  await page.locator('[data-cell="1,1"]').click();assert((await page.locator('.viz-detail').innerText()).includes('= 3'));
  await page.goto(url+'#viz-attention');await page.waitForSelector('[data-cell="1,0"]');
  near(Number(await page.locator('[data-cell="1,0"]').textContent()),1/(1+Math.exp(1/Math.sqrt(2))));
  assert.equal((await page.locator('[data-cell="0,1"]').textContent()),'遮住');
  await page.locator('[data-key="causal"]').uncheck();assert.notEqual(await page.locator('[data-cell="0,1"]').textContent(),'遮住');
  await page.goto(url+'#viz-probability');await page.waitForSelector('[data-key="q"]');
  await page.locator('[data-key="q"]').evaluate(el=>{el.value=80;el.dispatchEvent(new Event('input',{bubbles:true}));});
  const values=await page.locator('.viz-matrix tbody td').allTextContents();
  near(Number(values[0]),-0.8*Math.log(.8)-.2*Math.log(.2));near(Number(values[2]),0);
  await page.goto(url+'#viz-cache');await page.waitForSelector('.viz-detail');
  assert((await page.locator('.viz-detail').innerText()).includes('25 vs 13'));
  await page.goto(url+'#viz-rope');await page.waitForSelector('[data-key="m"]');
  await page.locator('[data-key="m"]').evaluate(el=>{el.value=3;el.dispatchEvent(new Event('input',{bubbles:true}));});
  assert((await page.locator('.viz-detail').innerText()).includes('= 1。'));
  await page.goto(url+'#viz-lora');await page.waitForSelector('[data-key="alpha"]');
  await page.locator('[data-key="alpha"]').evaluate(el=>{el.value=0;el.dispatchEvent(new Event('input',{bubbles:true}));});
  const rows=await page.locator('.viz-matrix').last().locator('tbody tr').all();
  for(const row of rows){const v=await row.locator('td').allTextContents();near(Number(v[1]),0);near(Number(v[0]),Number(v[2]));}
  report.numerical_interactions=true;
  // Check tutorial routes, actual code fences, deep reading and narrow screen.
  for(const id of ['TH01','TH05','TH06','TH14','TH17','TH21']){
   await page.goto(url+'#q-'+id);await page.waitForSelector('.qbody');
   await page.selectOption('#depth','full');await page.setViewportSize({width:390,height:844});
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
   await page.locator('.question').screenshot({path:path.join(shots,id+'-mobile.png')});
  }
  await context.close();
  // Compare against the optimized 1.4 snapshot, not the older unoptimized page.
  for(const [label,reader] of [['before',path.join(work,'before.html')],['after',file]]){
   const runs=[];
   for(let i=0;i<5;i++){
    const ctx=await browser.newContext({viewport:{width:1440,height:1080}}),p=await ctx.newPage(),cdp=await ctx.newCDPSession(p);
    await cdp.send('Emulation.setCPUThrottlingRate',{rate:4});await p.goto(pathToFileURL(reader).href);await p.waitForSelector('.hero');
    const sample={load_ms:await p.evaluate(()=>performance.getEntriesByType('navigation')[0].loadEventEnd)};
    for(const [name,hash,query] of [['all','search',''],['agent','agent',''],['core','core',''],['search','search','模型']]){
     sample[name]=await p.evaluate(async({hash,query})=>{search.value=query;history.replaceState(null,'','#'+hash);const start=performance.now();route();content.getBoundingClientRect();const sync=performance.now()-start;await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));return {sync_ms:sync,paint_ms:performance.now()-start,body_nodes:content.querySelectorAll('.qbody *').length,nodes:content.querySelectorAll('*').length};},{hash,query});
    }
    if(label==='after'){
     sample.interactions=[];
     for(const type of types){
      await p.evaluate(type=>{history.replaceState(null,'','#viz-'+type);route();},type);
      const timing=await p.evaluate(()=>{const ts=[];for(let n=0;n<10;n++){const el=document.querySelector('.viz-controls input,.viz-controls select,[data-cell]');if(el.type==='range')el.value=n%2?el.min:el.max;else if(el.type==='checkbox')el.checked=!el.checked;else if(el.tagName==='SELECT')el.selectedIndex=n%el.options.length;const start=performance.now();el.dispatchEvent(new Event(el.dataset.cell?'click':'input',{bubbles:true}));document.querySelector('.viz-result').getBoundingClientRect();ts.push(performance.now()-start);}return Math.max(...ts);});
      sample.interactions.push(timing);
     }
     sample.max_interaction_ms=Math.max(...sample.interactions);
     await p.evaluate(()=>go('home'));await p.waitForSelector('.hero');assert.equal(await p.locator('[data-mounted]').count(),0);
    }
    runs.push(sample);await ctx.close();
   }
   report.performance[label]={reader_bytes:fs.statSync(reader).size,load_ms:median(runs.map(x=>x.load_ms)),routes:{}};
   for(const route of ['all','agent','core','search'])report.performance[label].routes[route]=Object.fromEntries(Object.keys(runs[0][route]).map(k=>[k,median(runs.map(x=>x[route][k]))]));
   if(label==='after')report.performance.after.max_interaction_ms=Math.max(...runs.map(x=>x.max_interaction_ms));
  }
  report.performance.method='Headless Edge; 4× CPU slowdown; fresh browser context; median of 5 runs for routes; worst of 5×8×10 interaction samples. Not disk cold-cache or physical mobile hardware.';
  assert.equal(report.performance.after.routes.all.body_nodes,0);
  assert(report.performance.after.routes.all.sync_ms<150,'All-list synchronous work exceeds budget');
  assert(report.performance.after.max_interaction_ms<100,'Visual interaction exceeds 100ms at 4× CPU slowdown');
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.network,[]);
  fs.writeFileSync(path.join(root,'90_维护与来源/transformer-visual-qa.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify({version:report.version,visual_cases:report.visuals.length,performance:report.performance,errors:report.errors,network:report.network},null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
