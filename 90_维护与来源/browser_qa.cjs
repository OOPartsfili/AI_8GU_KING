const runtimeModules=require('path').join(require('os').homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
let chromium;try{({chromium}=require('playwright'));}catch{({chromium}=require(require('path').join(runtimeModules,'playwright')));}
const {pathToFileURL}=require('url');
const fs=require('fs'),path=require('path'),assert=require('assert');
const base=path.resolve(__dirname,'..');
const work=process.env.KNOWLEDGE_QA_OUTPUT||path.join(require('os').tmpdir(),'llm-interview-qa');fs.mkdirSync(work,{recursive:true});
const url=pathToFileURL(path.join(base,'开始阅读.html')).href;
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.KNOWLEDGE_BROWSER||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const context=await browser.newContext({viewport:{width:1440,height:1080},acceptDownloads:true});
 const page=await context.newPage();const errors=[],network=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
 await page.goto(url);await page.waitForSelector('.hero');
 // Home and collapsed lists must leave all question records unparsed.
 const initialRecords=await page.locator('script[id^="data-q-"]').count();
 assert(initialRecords>0);
 await page.locator('#coreBtn').click();await page.waitForSelector('#q-M01');
 assert.equal(await page.locator('script[id^="data-q-"]').count(),initialRecords);
 assert.equal(await page.locator('.qbody').count(),0);
 await page.locator('#q-M01>summary').focus();await page.keyboard.press('Enter');
 await page.waitForSelector('#q-M01 .qbody');
 assert.equal(await page.locator('script[id^="data-q-"]').count(),initialRecords-1);
 assert.equal(await page.locator('.qbody').count(),1);
 assert.equal(await page.locator('.deepbody>*').count(),0);
 await page.locator('#q-M01 .deep>summary').click();
 await page.waitForSelector('#q-M01 .deepbody p');
 await page.goto(url);await page.waitForSelector('.hero');
 const data=await page.evaluate(()=>DB);
 assert.equal(data.questions.length,data.stats.question_count);assert.equal(data.sources.length,data.stats.source_count);
 assert.equal(data.beginner.length,20);assert.equal(data.version,'1.5');
 await page.screenshot({path:path.join(work,'reader-desktop.png'),fullPage:true});
 await page.locator('#beginnerBtn').click();await page.waitForSelector('.question');
 assert.equal(await page.locator('.question').count(),20);
 assert.deepEqual(await page.locator('.question').evaluateAll(nodes=>nodes.map(n=>n.id.slice(2))),data.beginner);
 await page.goto(url+'#q-M02');await page.waitForSelector('#q-M02[open]');
 assert.equal(await page.locator('.deep[open]').count(),0);
 assert((await page.locator('#q-M02').innerText()).includes('红球'));
 assert((await page.locator('#q-M02').innerText()).includes('0.092'));
 await page.screenshot({path:path.join(work,'reader-ce-intro.png'),fullPage:false});
 await page.getByRole('heading',{name:'KL 散度：其中有多少分是可以避免的？',exact:true}).scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(work,'reader-kl-intro.png'),fullPage:false});
 await page.selectOption('#depth','full');assert.equal(await page.locator('.deep[open]').count(),1);
 await page.reload();assert.equal(await page.locator('#depth').inputValue(),'full');
 assert.equal(await page.locator('.deep[open]').count(),1);
 await page.selectOption('#depth','intro');assert.equal(await page.locator('.deep[open]').count(),0);
 await page.locator('.deep>summary').click();assert.equal(await page.locator('.deep[open]').count(),1);
 await page.locator('.deep>summary').click();
 const newDocs=data.documents.filter(d=>d.path.includes('新人从这里开始')||d.path.includes('术语翻译小词表'));
 assert.equal(newDocs.length,2);
 for(const d of newDocs){await page.locator('nav button[data-route="doc-'+d.id+'"]').click();await page.getByRole('heading',{name:d.title,exact:true}).waitFor();assert((await page.locator('.article').innerText()).includes(d.title));}
 await page.locator('#coreBtn').click();await page.waitForSelector('.question');
 assert.equal(await page.locator('.question').count(),data.core.length);
 await page.locator('#expandAll').click();
 await page.waitForFunction(()=>!document.getElementById('expandAll').disabled);
 assert.equal(await page.locator('details.question[open]').count(),data.core.length);
 assert.equal(await page.locator('details.deep[open]').count(),0);
 await page.locator('#collapseAll').click();
 assert.equal(await page.locator('details[open]').count(),0);
 await page.goto(url+'#q-P08');await page.waitForSelector('#q-P08[open]');
 await page.selectOption('#depth','full');
 assert((await page.locator('math').count())>0);
 assert((await page.locator('#q-P08').innerText()).includes('γ/β'));
 await page.screenshot({path:path.join(work,'reader-simpo.png'),fullPage:true});
 await page.selectOption('.study','done');await page.reload();
 assert.equal(await page.locator('.study').inputValue(),'done');
 const downloadPromise=page.waitForEvent('download');
 await page.locator('#exportBtn').click();
 const download=await downloadPromise;const saved=path.join(work,'qa-progress.json');await download.saveAs(saved);
 const exported=JSON.parse(fs.readFileSync(saved,'utf8'));assert.equal(exported.progress.P08.state,'done');
 await page.evaluate(()=>localStorage.clear());await page.reload();assert.equal(await page.locator('.study').inputValue(),'new');
 await page.locator('#importFile').setInputFiles(saved);
 await page.waitForFunction(()=>document.getElementById('notice').textContent.includes('已导入'));
 assert.equal(await page.locator('.study').inputValue(),'done');
 await page.selectOption('#depth','full');
 await page.locator('a[href="#source-S09"]').click();
 await page.waitForSelector('#source-S09');
 assert((await page.locator('#source-S09').innerText()).includes('SimPO'));
 await page.locator('#search').fill('KV cache');await page.locator('#searchBtn').click();
 await page.getByRole('heading',{name:'检索结果',exact:true}).waitFor();
 const searchCount=await page.locator('.question').count();assert(searchCount>=3);
 await page.locator('#clearBtn').click();await page.selectOption('#priority','P2');
 assert.equal(await page.locator('.question').count(),data.questions.filter(q=>q.priority==='P2').length);
 await page.locator('#clearBtn').click();
 await page.locator('#distillBtn').click();await page.waitForSelector('#q-KD01');
 await page.waitForFunction(()=>location.hash==='#distill'&&document.querySelectorAll('.question').length===34);
 assert.equal(await page.locator('.question').count(),34);
 assert.deepEqual(await page.locator('.question').evaluateAll(nodes=>nodes.map(n=>n.id.slice(2))),
                  Array.from({length:34},(_,i)=>'KD'+String(i+1).padStart(2,'0')));
 await page.screenshot({path:path.join(work,'reader-distillation.png'),fullPage:true});
 for(const [name,count,first] of [['opd',10,'KD25'],['posttrain',8,'HP01'],['pretrain',20,'PT01']]){
   await page.locator('#search').fill('NONMATCHING');await page.selectOption('#priority','P2');
   await page.locator('#'+name+'Btn').click();await page.waitForSelector('#q-'+first);
   assert.equal(await page.locator('.question').count(),count,name+' count / reset filters');
   assert.equal(await page.locator('#search').inputValue(),'');
   await page.screenshot({path:path.join(work,'reader-'+name+'.png'),fullPage:true});
 }
 for(const [term,id] of [['OPSD','KD27'],['MOPD','KD29'],['GSPO','HP02'],['MTP','PT15']]){
   await page.locator('#search').fill(term);await page.locator('#searchBtn').click();
   await page.waitForSelector('#q-'+id);
 }
 await page.locator('#clearBtn').click();
 await page.goto(url+'#q-PT03');await page.waitForSelector('#q-PT03[open]');
 await page.locator('a[href="#source-S103"]').click();await page.waitForSelector('#source-S103');
 assert((await page.locator('#source-S103').innerText()).includes('Fill in the Middle'));
 await page.goto(url+'#q-PT19');await page.waitForSelector('#q-PT19[open]');
 await page.locator('a[href="#source-S100"]').first().click();await page.waitForSelector('#source-S100');
 assert((await page.locator('#source-S100').innerText()).includes('Olmo 3'));
 await page.locator('#agentBtn').click();await page.waitForSelector('#agentWalk');
 assert.equal(await page.locator('.question').count(),50);
 assert.equal(await page.locator('#agentPrev').isDisabled(),true);
 for(const scenario of ['normal','timeout','unknown']){
   await page.selectOption('#agentScenario',scenario);
   for(let i=0;i<4;i++)await page.locator('#agentNext').click();
   const text=await page.locator('.agent-walk-state').innerText();
   assert(text.includes(scenario==='unknown'?'暂停，结果仍未知':'已验证完成'));
   assert(await page.locator('#agentNext').isDisabled());
   await page.locator('#agentRestart').click();assert(await page.locator('#agentPrev').isDisabled());
 }
 await page.screenshot({path:path.join(work,'reader-agent-desktop.png'),fullPage:true});
 await page.goto(url+'#agent-start');await page.waitForSelector('#q-AG25');
 assert.equal(await page.locator('.question').count(),12);
 assert.deepEqual(await page.locator('.question').evaluateAll(nodes=>nodes.map(n=>n.id.slice(2))),
   ['A01','AG01','A02','AG03','AG04','A04','A05','AG05','AG13','A09','AG25','A07']);
 await page.locator('#search').fill('MCP');await page.locator('#searchBtn').click();await page.waitForSelector('#q-AG05');
 await page.locator('#clearBtn').click();
 const formulaOverflow=[];
 // Check all question routes render and have no raw MathML error nodes.
 for(const q of data.questions){
   await page.evaluate(id=>{location.hash='q-'+id},q.id);
   await page.waitForSelector('#q-'+q.id);
   const info=await page.evaluate(()=>{
     const question=document.querySelector('.question');
     return {text:question.innerText.length, mathError:question.querySelectorAll('merror').length,
       pageOverflow:document.documentElement.scrollWidth>window.innerWidth+2};
   });
   assert(info.text>100,q.id);assert.equal(info.mathError,0,q.id);
   assert.equal(await page.locator('.deep[open]').count(),1,q.id);
   if(info.pageOverflow)formulaOverflow.push(q.id);
 }
 assert.equal(formulaOverflow.length,0,'desktop overflow: '+formulaOverflow.join(','));
 const brokenRefs=await page.evaluate(()=>{
   const db=DB;
   const q=new Set(db.questions.map(x=>x.id)),s=new Set(db.sources.map(x=>x.id));
   const bad=[];
   for(const entry of [...db.questions,...db.documents]){
     const container=document.createElement('div');container.innerHTML=entry.html;
     for(const a of container.querySelectorAll('a[href]')){
       const h=a.getAttribute('href');
       if(h.startsWith('#q-')&&!q.has(h.slice(3)))bad.push(h);
       if(h.startsWith('#source-')&&!s.has(h.slice(8)))bad.push(h);
     }
   }return bad;
 });assert.deepEqual(brokenRefs,[]);
 // A queued expansion must stop immediately after collapse or route changes.
 await page.goto(url+'#search');await page.waitForSelector('#expandAll');
 const lastQuestion=page.locator('.question').last();
 await lastQuestion.locator('summary').scrollIntoViewIfNeeded();
 await lastQuestion.locator('summary').click();
 await lastQuestion.locator('.qbody').waitFor();
 assert((await lastQuestion.locator('.qbody').innerText()).length>100);
 await page.screenshot({path:path.join(work,'reader-last-question.png')});
 await page.evaluate(()=>{expandQuestions();document.getElementById('collapseAll').click();});
 await page.waitForTimeout(100);
 assert.equal(await page.locator('.question[open]').count(),0);
 assert.equal(await page.locator('#expandAll').isDisabled(),false);
 await page.evaluate(()=>{expandQuestions();go('q-M02');});
 await page.waitForSelector('#q-M02 .qbody');await page.waitForTimeout(100);
 assert.equal(await page.locator('.question').count(),1);
 assert.equal(await page.locator('#content[aria-busy]').count(),0);
 // Printing a collapsed list still materializes all content, including formulas.
 await page.goto(url+'#core');await page.waitForSelector('#expandAll');
 const printed=await page.evaluate(()=>{
  window.dispatchEvent(new Event('beforeprint'));
  return {questions:content.querySelectorAll('.question').length,bodies:content.querySelectorAll('.qbody').length,
   deep:content.querySelectorAll('.deep[data-loaded]').length,open:content.querySelectorAll('.question[open]').length};
 });
 assert.equal(printed.bodies,data.core.length);assert.equal(printed.deep,data.core.length);assert.equal(printed.open,data.core.length);
 await page.evaluate(()=>window.dispatchEvent(new Event('afterprint')));
 assert.equal(await page.locator('.question[open]').count(),0);
 await page.setViewportSize({width:390,height:844});
 await page.selectOption('#depth','intro');
 await page.goto(url+'#q-M02');await page.waitForSelector('#q-M02');
 await page.screenshot({path:path.join(work,'reader-mobile.png'),fullPage:true});
 const mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+2);
 assert(!mobileOverflow,'mobile overflow');
 for(const id of ['KD27','KD29','PT02','PT11']){
   await page.selectOption('#depth','full');await page.goto(url+'#q-'+id);await page.waitForSelector('#q-'+id);
   assert(!await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+2),id+' mobile overflow');
   await page.screenshot({path:path.join(work,'reader-mobile-'+id+'.png'),fullPage:true});
 }
 await page.goto(url+'#agent');await page.waitForSelector('#agentWalk');
 await page.selectOption('#agentScenario','timeout');for(let i=0;i<4;i++)await page.locator('#agentNext').click();
 for(const width of [390,320]){
   await page.setViewportSize({width,height:844});
   assert(!await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+2),'agent mobile overflow');
 }
 await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:path.join(work,'reader-agent-mobile.png'),fullPage:true});
 await page.goto(url+'#q-AG05');await page.waitForSelector('#q-AG05');
 await page.screenshot({path:path.join(work,'reader-agent-mcp-mobile.png'),fullPage:true});
 await page.goto(url+'#home');await page.waitForSelector('.hero');
 await page.screenshot({path:path.join(work,'reader-mobile-home.png'),fullPage:true});
 assert.equal(errors.length,0);assert.equal(network.length,0);
 // Never ship QA progress as the user's personal progress.
 await page.evaluate(()=>localStorage.clear());
 const report={tested_questions:data.questions.length,core_questions:data.core.length,sources:data.sources.length,
   version:data.version,beginner_questions:data.beginner.length,beginner_order:true,distillation_route_34:true,opd_route_10:true,posttraining_route_8:true,pretraining_route_20:true,three_digit_source_links:true,agent_course_50:true,agent_start_12:true,agent_walkthrough_3_scenarios:true,
   new_guide_and_glossary:true,reading_depth_switch_persistence_and_manual_toggle:true,
   search_count:searchCount,progress_save_reload_export_import:true,math_render:true,
   lazy_records_and_question_bodies:true,lazy_deep_content:true,keyboard_expansion:true,
   batch_expand_collapse_and_navigation_cancellation:true,print_deferred_content:true,
   desktop_overflow:formulaOverflow,mobile_overflow:mobileOverflow,broken_refs:brokenRefs,
   javascript_errors:errors,automatic_http_requests:network,viewport_desktop:[1440,1080],viewport_mobile:[390,844]};
 fs.writeFileSync(path.join(base,'90_维护与来源/browser-qa.json'),JSON.stringify(report,null,2),'utf8');
 console.log(JSON.stringify(report,null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
