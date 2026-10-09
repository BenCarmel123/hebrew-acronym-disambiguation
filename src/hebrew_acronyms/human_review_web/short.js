'use strict';
(function(root){
  const LABELS={fits:'מתאימה להקשר',not_fits:'לא מתאימה',unsure:'לא בטוח'};
  const COMPARISONS={agreement:'תואם לניקוד האוטומטי',disagreement:'פער מול הניקוד האוטומטי',undetermined:'אין השוואה חד־משמעית'};
  const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function labelOf(value){return typeof value==='string'?value:value?.label||'';}
  function savedLabels(record={}){return Object.fromEntries(Object.entries(record.judgments||{}).map(([id,value])=>[id,labelOf(value)]));}
  function firstIncomplete(state){return state.queue.find(id=>state.records[id]?.completion?.status!=='complete');}
  function progressFor(item,record={}){
    const labels=savedLabels(record);
    return {judged:item.answers.filter(a=>Object.hasOwn(LABELS,labels[a.id])).length,total:item.answers.length};
  }
  function highlightedSentence(item){
    let text='',cursor=0;
    for(const span of item.target_spans||[]){
      if(Number.isInteger(span.start)&&Number.isInteger(span.end)&&span.start>=cursor&&span.end>span.start&&span.end<=item.sentence.length){
        text+=esc(item.sentence.slice(cursor,span.start))+'<mark>'+esc(item.sentence.slice(span.start,span.end))+'</mark>';cursor=span.end;
      }
    }
    return text+esc(item.sentence.slice(cursor));
  }
  // Deliberate whitelist: the main answer view never consumes model or score fields.
  function renderMainAnswer(answer,selected=''){
    return `<article class="card"><h3>תשובה ${esc(answer.code)}</h3><div class="answer-text" dir="auto">${esc(answer.text||'לא נמסר מענה')}</div><fieldset class="choices" data-answer="${esc(answer.id)}"><legend>השיפוט שלך לתשובה ${esc(answer.code)}</legend>${Object.entries(LABELS).map(([value,label])=>`<label class="choice"><input type="radio" name="judgment-${esc(answer.id)}" value="${value}" ${selected===value?'checked':''}><span>${label}</span></label>`).join('')}</fieldset></article>`;
  }
  function renderMain(item,record={}){
    const labels=savedLabels(record);
    return `${item.prior_reused?`<p class="notice">${record.completion?.status==='complete'?'השיפוטים הקודמים לפריט הזה נשמרו במסלול הקצר. אין צורך לבדוק אותו שוב.':'העבודה הקודמת נשמרה; יש להשלים רק שיפוטים חסרים.'}</p>`:''}<article class="card"><div class="small">קיצור: ${esc(item.acronym)}</div><div class="sentence">${highlightedSentence(item)}</div><div class="gold"><span>ייחוס המקור (עשוי להיות שגוי)</span><strong>${esc(item.gold??'לא זמין')}</strong></div></article><div class="answers">${item.answers.map(answer=>renderMainAnswer(answer,labels[answer.id])).join('')}</div><section class="card optional"><label><input id="suspect" type="checkbox" ${record.suspect?'checked':''}>חשד לבעיה בייחוס או במשפט</label><label><input id="example" type="checkbox" ${record.example?'checked':''}>דוגמה שכדאי לשמור למאמר</label><label for="note">הערה קצרה (רשות)</label><textarea id="note" rows="2" placeholder="אפשר להשאיר ריק">${esc(record.note||'')}</textarea></section>`;
  }
  function brief(value){
    if(value===null||value===undefined||value==='')return 'לא זמין';
    if(typeof value==='boolean')return value?'כן':'לא';
    if(Array.isArray(value))return value.map(brief).join(' · ');
    if(typeof value==='object')return Object.entries(value).map(([key,val])=>`${key}: ${brief(val)}`).join(' · ');
    return String(value);
  }
  function renderCase(value){
    if(typeof value!=='object'||value===null)return `<div class="case">${esc(value)}</div>`;
    return `<article class="case"><strong><bdi>${esc(value.id||'')}</bdi></strong>${value.sentence?`<p>${esc(value.sentence)}</p>`:''}${value.gold!==undefined?`<p class="small">ייחוס המקור: ${esc(value.gold)}</p>`:''}${value.inherited_from?'<p class="notice">השיפוטים נשמרו מהפרוטוקול הקודם; אין כאן תיוג עצמאי חדש.</p>':''}${value.note?`<p>הערה אנושית: ${esc(value.note)}</p>`:''}${value.reason?`<p>${esc(value.reason)}</p>`:''}${value.answers?.length?`<ul>${value.answers.map(a=>`<li><strong>${esc(a.model||a.code||'תשובה')}</strong>: ${esc(a.text??a.raw??'לא נמסר מענה')}<br><span class="small">שיפוט אנושי: ${esc(LABELS[labelOf(a)]||'טרם סומן')}${a.auto_score!==undefined?' · ציון אוטומטי שמור: '+esc(brief(a.auto_score)):''}${a.comparison?' · '+esc(COMPARISONS[a.comparison]||'אין השוואה חד־משמעית'):''}</span></li>`).join('')}</ul>`:''}</article>`;
  }
  const api={LABELS,esc,labelOf,savedLabels,firstIncomplete,progressFor,highlightedSentence,renderMainAnswer,renderMain,renderCase};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  root.ShortReview=api;
  if(typeof document==='undefined')return;

  const $=id=>document.getElementById(id);
  let state,currentItem,dirty=false,timer=null,pending=Promise.resolve(),lastError=null;
  async function request(path,payload){
    const response=await fetch(path,payload?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}:{});
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||'השמירה לא הושלמה');
    return result;
  }
  function status(message,error=false){$('saveStatus').textContent=message;$('saveStatus').className=error?'error':'';}
  function showError(error){
    lastError=error;
    const conflict=/revision|another window|state changed|reload|conflict/i.test(error.message);
    status(conflict?'העבודה השתנתה בחלון אחר. השינויים בחלון הזה לא נשמרו; השאירו אותו פתוח ואל תדרסו את העבודה השמורה.':'לא נשמר: '+error.message+' · השאירו את החלון פתוח ונסו שוב.',true);
  }
  function locked(task){
    const operation=pending.then(async()=>{
      document.querySelector('main').inert=true;
      try{return await task();}finally{document.querySelector('main').inert=false;}
    });
    pending=operation.catch(()=>{});
    return operation;
  }
  function run(task){locked(task).catch(showError);}
  function showScreen(name){for(const id of ['reviewScreen','detailsScreen','summaryScreen','startupError'])$(id).hidden=id!==name;}
  function updateProgress(){
    const counts=state.counts||{},total=counts.total_items??state.queue.length;
    $('progress').textContent=`${counts.complete_items||0} מתוך ${total} משפטים הושלמו`;
    $('countDetail').textContent=`נבדקו ${counts.reviewed_items??((counts.complete_items||0)+(counts.partial_items||0))} · ${counts.partial_items||0} חלקיים`;
    $('reviewer').textContent=`מתייג/ת: ${state.reviewer||'לא זמין'}`;
    if(currentItem){
      const p=progressFor(currentItem,state.records[currentItem.id]);
      $('itemProgress').textContent=`${p.judged}/${p.total} תשובות סומנו ונשמרו`;
      const index=state.queue.indexOf(currentItem.id);
      $('position').textContent=`משפט ${index+1} מתוך ${state.queue.length}`;
      $('previous').disabled=index<=0;
      $('saveNext').textContent=index===state.queue.length-1?'שמירה וסיכום':'שמירה והבא ←';
    }
  }
  function formPayload(){
    const judgments={};
    for(const group of document.querySelectorAll('[data-answer]'))judgments[group.dataset.answer]=group.querySelector('input:checked')?.value||'';
    return {item_id:currentItem.id,judgments,suspect:$('suspect').checked,example:$('example').checked,note:$('note').value};
  }
  async function saveNow(){
    clearTimeout(timer);
    if(!dirty||!currentItem)return;
    const payload=formPayload();
    status('שומר…');
    state=await request('/api/short/save',{revision:state.revision,...payload});
    dirty=false;lastError=null;updateProgress();status('נשמר בדיסק · אפשר להמשיך או לעצור');
  }
  async function flush(){clearTimeout(timer);await saveNow();}
  function markDirty(){dirty=true;lastError=null;clearTimeout(timer);status('שומר את השינוי…');timer=setTimeout(()=>run(saveNow),500);}
  function renderItem(){
    $('itemView').innerHTML=renderMain(currentItem,state.records[currentItem.id]);
    $('itemView').querySelectorAll('input,textarea').forEach(element=>element.addEventListener('input',markDirty));
    updateProgress();showScreen('reviewScreen');
  }
  async function openItem(id){
    if(!id){$('itemView').innerHTML='<p class="empty">אין פריטים במסלול הזה.</p>';$('saveNext').disabled=true;$('previous').disabled=true;$('detailsButton').disabled=true;return;}
    await flush();
    status('טוען פריט…');
    const result=await request('/api/short/open',{revision:state.revision,item_id:id});
    state=result.state;currentItem=result.item;dirty=false;lastError=null;renderItem();
    status('העבודה השמורה נטענה · שינויים נשמרים אוטומטית בדיסק');
  }
  const detailNames={source:'מקור',source_metadata:'פרטי המקור',selection_reason:'סיבת הכללה',candidates:'מועמדים במקור'};
  function renderDetails(details){
    const rows=Object.entries(detailNames).filter(([key])=>details[key]!==undefined).map(([key,label])=>`<div class="details-row"><strong>${label}</strong>${esc(brief(details[key]))}</div>`).join('');
    $('detailsView').innerHTML=`<section class="card">${rows}</section>${(details.answers||[]).map(a=>`<section class="card"><h3>${esc(a.model||a.code||'תשובה')}</h3><p class="small">${esc(a.task||'')}</p><div class="answer-text">${esc(a.raw??a.text??'לא נמסר מענה')}</div><p>פענוח שמור: ${esc(a.decoded??'לא זמין')}</p><p>ציון אוטומטי שמור: ${esc(brief(a.auto_score))}</p></section>`).join('')}${details.legacy_record?`<section class="card"><h3>העבודה הקודמת שנשמרה</h3><p class="small">הרשומה הקודמת נשמרה בנפרד. הטקסט הבא משקף את הרשומה המקורית.</p><pre class="technical-value">${esc(JSON.stringify(details.legacy_record,null,2))}</pre></section>`:''}`;
  }
  function renderSummary(summary){
    const c=summary.counts||state.counts||{};
    const groups=summary.group_labels||{};
    const plan=summary.composition||{};
    const composition=Object.keys(groups).length?`<div class="table-wrap"><table><thead><tr><th>קבוצה לפי הניקוד השמור</th><th>נבחרו</th><th>נבדקו</th><th>הושלמו</th></tr></thead><tbody>${Object.entries(groups).map(([key,label])=>`<tr><td>${esc(label)}</td><td>${plan.selected?.[key]||0}</td><td>${plan.reviewed?.[key]||0}</td><td>${plan.completed?.[key]||0}</td></tr>`).join('')}</tbody></table></div><p class="small">״נבדקו״: סומנה לפחות תשובה אחת. ״הושלמו״: סומנו שתי תשובות היצירה בלבד, כולל ״לא בטוח״.</p><p>מקורות שנבחרו: ${esc(brief(plan.sources_selected||{}))}</p><p>מקורות שנבדקו: ${esc(brief(plan.sources_reviewed||{}))}</p><p>קיצורים שונים: ${esc(plan.distinct_acronyms??'לא זמין')}</p>${plan.adjustments?.length?`<p>התאמות ברשימה: ${esc(brief(plan.adjustments))}</p>`:''}`:esc(brief(plan));
    const sections=[['disagreements','פערים בין השיפוט האנושי לניקוד השמור'],['suspicions','חשדות שסומנו בידי המתייג'],['examples','דוגמאות שסומנו למאמר'],['unresolved','מקרים שסומנו ״לא בטוח״'],['cases','כל המשפטים שנבדקו']];
    $('summaryView').innerHTML=`<div class="summary-counts"><div><strong>${c.reviewed_items??((c.complete_items||0)+(c.partial_items||0))}</strong> משפטים שנבדקו</div><div><strong>${c.complete_items||0}</strong> משפטים מלאים</div><div><strong>${c.partial_items||0}</strong> משפטים חלקיים</div><div><strong>${c.judged_answers||0}</strong> תשובות שסומנו מתוך ${c.total_answers??state.queue.length*2}</div></div><section class="card"><h3>הרכב רשימת הבדיקה</h3>${composition||'<p>לא נמסר פירוט נוסף.</p>'}</section><p class="small">פערי ניקוד והרכב הרשימה הם חישוב מתוך הנתונים השמורים. התוויות, החשדות והדוגמאות הם החלטות אנושיות; לא נוסף הסבר סמנטי אוטומטי.</p>${sections.map(([key,title])=>`<section class="card"><h3>${title}</h3>${summary[key]?.length?summary[key].map(renderCase).join(''):'<p class="small">אין פריטים בסעיף זה.</p>'}</section>`).join('')}<section class="card"><h3>משפטים שלא הושלמו</h3><p>${summary.incomplete_ids?.length?summary.incomplete_ids.map(esc).join(' · '):'כל המשפטים במסלול הושלמו.'}</p></section><section class="card"><h3>גבולות הסיכום</h3>${(Array.isArray(summary.limitations)?summary.limitations:[summary.limitations||'הרשימה מצומצמת ונבחרה לבדיקה איכותנית. אין להסיק ממנה שיעורי ביצוע בכלל המאגר.']).map(text=>`<p>${esc(text)}</p>`).join('')}</section>`;
  }
  async function openSummary(){
    await flush();status('מכין סיכום…');
    const result=await request('/api/short/summary',{revision:state.revision});
    state=result.state;renderSummary(result.summary);updateProgress();showScreen('summaryScreen');lastError=null;
    status('כל השינויים נשמרו · הסיכום מוכן');window.scrollTo({top:0});
  }
  async function download(path,name){await flush();const link=document.createElement('a');link.href=path;link.download=name;document.body.appendChild(link);link.click();link.remove();}
  async function init(){
    try{
      const [initial,session]=await Promise.all([request('/api/short/state'),request('/api/session')]);
      state=initial;$('qaBanner').hidden=!session.qa;updateProgress();const next=firstIncomplete(state);if(next||!state.queue.length)await openItem(next);else await openSummary();
    }catch(error){showError(error);if(!currentItem)showScreen('startupError');}
  }
  $('previous').onclick=()=>run(()=>openItem(state.queue[state.queue.indexOf(currentItem.id)-1]));
  $('saveNext').onclick=()=>run(async()=>{await flush();const next=state.queue[state.queue.indexOf(currentItem.id)+1];if(next)await openItem(next);else await openSummary();});
  $('stopSummary').onclick=()=>run(openSummary);
  $('detailsButton').onclick=()=>run(async()=>{await flush();const result=await request('/api/short/details',{revision:state.revision,item_id:currentItem.id});state=result.state;renderDetails(result.details);showScreen('detailsScreen');status('פתיחת הפרטים תועדה · העבודה נשמרה');window.scrollTo({top:0});});
  $('backDetails').onclick=()=>{showScreen('reviewScreen');window.scrollTo({top:0});};
  $('backSummary').onclick=()=>run(async()=>{if(!currentItem)await openItem(state.queue[0]);else showScreen('reviewScreen');window.scrollTo({top:0});});
  $('exportAnnotations').onclick=()=>run(()=>download('/api/short/export.json','short-review-annotations.json'));
  $('exportSummary').onclick=()=>run(()=>download('/api/short/summary/export.md','short-review-summary.md'));
  $('retryLoad').onclick=()=>run(init);
  window.addEventListener('beforeunload',event=>{if(dirty||lastError){event.preventDefault();event.returnValue='';}});
  run(init);
})(typeof globalThis!=='undefined'?globalThis:this);
