// Compare offline reader rendering in a fresh, isolated Edge context.
const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert');
const {pathToFileURL}=require('url');
let chromium;try{({chromium}=require('playwright'));}catch{({chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')));}
const root=path.resolve(__dirname,'..');
const before=path.join(root,'work/reader-performance/before.html');
const after=path.join(root,'开始阅读.html');
const original=JSON.parse(fs.readFileSync(before,'utf8').match(/<script id="knowledge" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const median=values=>[...values].sort((a,b)=>a-b)[Math.floor(values.length/2)];
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.KNOWLEDGE_BROWSER||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 try{
  const report={method:'Headless Edge, 1440×1080, fresh context for each run, 4× CPU slowdown, median of 5 runs; route includes synchronous layout and two animation frames. Not a disk cold-cache measurement.',results:{}};
  for(const [label,file] of (process.argv.includes('--before-only')?[['before',before]]:[['before',before],['after',after]])){
   const runs=[];
   for(let run=0;run<5;run++){
    const context=await browser.newContext({viewport:{width:1440,height:1080}});
    const page=await context.newPage();const cdp=await context.newCDPSession(page);
    await cdp.send('Emulation.setCPUThrottlingRate',{rate:4});
    await page.goto(pathToFileURL(file).href);
    await page.waitForSelector('.hero');
    const sample={load_ms:await page.evaluate(()=>performance.getEntriesByType('navigation')[0].loadEventEnd)};
    for(const [name,hash,query] of [['all','search',''],['agent','agent',''],['core','core',''],['search','search','模型'],['question','q-M02','']]){
     sample[name]=await page.evaluate(async({hash,query})=>{
      document.getElementById('search').value=query;
      history.replaceState(null,'','#'+hash);
      const start=performance.now();route();
      document.getElementById('content').getBoundingClientRect();
      const sync=performance.now()-start;
      await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
      return {sync_ms:sync,paint_ms:performance.now()-start,nodes:document.querySelectorAll('#content *').length,
       questions:document.querySelectorAll('.question').length,body_nodes:document.querySelectorAll('.qbody *').length};
     },{hash,query});
    }
    if(label==='after'&&run===0){
     const questions=await page.evaluate(()=>DB.questions.map(({id,title,body,html})=>({id,title,body,html})));
     assert.deepEqual(questions,original.questions.map(({id,title,body,html})=>({id,title,body,html})),'Every question must retain its original content');
     report.question_content_identical=true;
    }
    runs.push(sample);await context.close();
   }
   const result={bytes:fs.statSync(file).size,load_ms:median(runs.map(x=>x.load_ms)),routes:{}};
   for(const name of ['all','agent','core','search','question']){
    result.routes[name]=Object.fromEntries(Object.keys(runs[0][name]).map(k=>[k,median(runs.map(x=>x[name][k]))]));
   }
   report.results[label]=result;
  }
  fs.writeFileSync(path.join(root,'90_维护与来源/performance-qa.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
  if(report.results.after){
   assert.equal(report.results.after.routes.all.questions,original.questions.length);
   assert.equal(report.results.after.routes.all.body_nodes,0,'Collapsed lists must not mount question bodies');
   assert(report.results.after.routes.all.nodes<report.results.before.routes.all.nodes/3);
  }
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
