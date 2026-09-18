/* Runs after the offline reader. Native bridge carries explicit user actions only. */
(() => {
  'use strict';
  const bridge = () => window.webkit?.messageHandlers?.nativeBridge;
  const send = message => { const handler=bridge(); if (!handler) return false; handler.postMessage(message); return true; };
  const aside=document.querySelector('aside');
  const toggle=document.createElement('button');
  toggle.id='mobileNavToggle';toggle.type='button';toggle.textContent='章节与学习路线';
  toggle.setAttribute('aria-controls','nav');toggle.setAttribute('aria-expanded','false');
  aside.prepend(toggle);
  function closeNav(){nav.classList.remove('mobile-open');toggle.setAttribute('aria-expanded','false');}
  toggle.onclick=()=>{const open=toggle.getAttribute('aria-expanded')!=='true';toggle.setAttribute('aria-expanded',String(open));nav.classList.toggle('mobile-open',open);};
  nav.addEventListener('click',event=>{if(event.target.closest('button'))closeNav();});
  window.addEventListener('hashchange',closeNav);

  const filters=document.querySelector('.filters'),options=document.createElement('details');
  options.id='mobileOptions';
  const summary=document.createElement('summary');summary.textContent='阅读深度、筛选与进度备份';
  filters.before(options);options.append(summary,filters);
  const shortcuts=document.createElement('div');shortcuts.id='mobileQuickNav';
  shortcuts.setAttribute('aria-label','快捷学习入口');
  for(const [title,hash] of [['入门20题','beginner'],['Agent','agent'],['手写教程','transformer'],['全部题目','search']]){
    const button=document.createElement('button');button.type='button';button.className='btn';button.textContent=title;
    button.onclick=()=>{search.value='';priority.value='';statusFilter.value='';go(hash);};shortcuts.append(button);
  }
  options.before(shortcuts);
  search.placeholder='搜索题号或知识点';
  search.setAttribute('enterkeyhint','search');
  document.getElementById('searchBtn').addEventListener('click',()=>search.blur());
  search.addEventListener('keydown',event=>{if(event.key==='Enter')search.blur();});

  function exportProgress(){
    const data={schema:'llm-study-progress-v1',exported:new Date().toISOString(),progress};
    if(send({action:'export',filename:'八股王_复习进度.json',content:JSON.stringify(data,null,2),mimeType:'application/json'})){
      notice('请选择“存储到文件”保存进度备份。');return true;
    }
    notice('请在八股王 App 中使用系统文件导出。');return false;
  }
  function importProgress(text){
    try{
      if(typeof text!=='string'||new TextEncoder().encode(text).length>2000000)throw new Error('文件过大或不是文本');
      const data=JSON.parse(text);
      if(data.schema!=='llm-study-progress-v1'||!data.progress||typeof data.progress!=='object'||Array.isArray(data.progress))throw new Error('进度格式不匹配');
      const incoming=safeProgress(data.progress);
      progress={...progress,...incoming};persist();route();
      const count=Object.keys(incoming).length,message='已导入 '+count+' 道题的进度。';notice(message);return {ok:true,count,message};
    }catch(error){const message='未导入：'+error.message+'。请选择阅读器导出的 JSON。';notice(message);return {ok:false,error:error.message,message};}
  }
  window.AI8GU={exportProgress,importProgress,goHome:()=>go('home')};
  document.getElementById('exportBtn').onclick=exportProgress;
  document.getElementById('importBtn').onclick=()=>{
    if(!send({action:'importProgress'}))document.getElementById('importFile').click();
  };
  // Code downloads use the same native share sheet rather than unsupported blob navigation.
  const browserDownloadCode=downloadCode;
  downloadCode=function(name){
    if(!bridge())return browserDownloadCode(name);
    if(!codeAssets)codeAssets=JSON.parse(document.getElementById('code-assets').textContent);
    if(Object.hasOwn(codeAssets,name))send({action:'export',filename:name,content:codeAssets[name],mimeType:'text/plain'});
  };
})();
