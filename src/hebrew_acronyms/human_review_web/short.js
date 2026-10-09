'use strict';
(function(root){
  const LABELS={fits:'מתאימה להקשר',not_fits:'לא מתאימה',unsure:'לא בטוח'};
  const TAGS={gibberish:'ג׳יבריש / טקסט לא מובן',inflection:'יחיד–רבים / נטייה',punctuation:'רווחים / מקפים / גרשיים',spelling:'הבדל כתיב, למשל י׳/ו׳',equivalent:'ניסוח חלופי שקול',extra_text:'טקסט עודף / סותר'};
  const TAG_HELP={gibberish:'טקסט משובש או לא קוהרנטי; לא כל פירוש שגוי או מילה לא מוכרת. אפשר ג׳יבריש חלקי סביב פירוש מזוהה.',inflection:'הבדל נטייה אינו בהכרח שקילות במשמעות; יש לשפוט לפי ההקשר.',punctuation:'הבדל ברווחים או בסימני פיסוק; אינו קובע כשלעצמו נכונות.',spelling:'הבדל כתיב אינו בהכרח שקילות במשמעות; יש לשפוט לפי ההקשר.',equivalent:'אותה משמעות בהקשר, לא רק דמיון בנושא.',extra_text:'מידע נוסף או סותר סביב הפירוש; השיפוט הסמנטי נשאר החלטה נפרדת.'};
  const PHASES={prior_protocol:'בפרוטוקול קודם',before_reveal:'בפרוטוקול החדש לפני חשיפה',after_reveal:'לאחר חשיפה מתועדת',unknown:'מצב החשיפה אינו ידוע'};
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
  function renderMainAnswer(answer,selected='',tags=[]){
    return `<article class="card"><h3>תשובה ${esc(answer.code)}</h3><div class="answer-text" dir="auto">${esc(answer.text??'לא נמסר מענה')}</div><fieldset class="choices" data-answer="${esc(answer.id)}"><legend>השיפוט שלך לתשובה ${esc(answer.code)}</legend>${Object.entries(LABELS).map(([value,label])=>`<label class="choice"><input type="radio" name="judgment-${esc(answer.id)}" value="${value}" ${selected===value?'checked':''}><span>${label}</span></label>`).join('')}</fieldset><fieldset class="tag-chips" data-tags-answer="${esc(answer.id)}"><legend>תופעות בולטות (רשות, אפשר כמה)</legend>${Object.entries(TAGS).map(([value,label])=>`<label class="tag-chip" title="${esc(TAG_HELP[value])}"><input type="checkbox" aria-label="${esc(label)}" value="${value}" ${tags.includes(value)?'checked':''}><span>${label}</span></label>`).join('')}</fieldset><p class="small tag-help">סמן תופעה בולטת אם יש; אין צורך לחפש בכוח או להסביר כל החלטה.</p><p class="small">תגית אינה קובעת נכונות. ללא תגיות: ״לא סומן״, ולא אישור שאין תופעות.</p></article>`;
  }
  function renderMain(item,record={}){
    const labels=savedLabels(record);
    return `${item.prior_reused?`<p class="notice">${record.completion?.status==='complete'?'השיפוטים הקודמים לפריט הזה נשמרו במסלול הקצר. אין צורך לבדוק אותו שוב.':'העבודה הקודמת נשמרה; יש להשלים רק שיפוטים חסרים.'}</p>`:''}<article class="card"><div class="small">קיצור: ${esc(item.acronym)}</div><div class="sentence">${highlightedSentence(item)}</div><div class="gold"><span>ייחוס המקור (עשוי להיות שגוי)</span><strong>${esc(item.gold??'לא זמין')}</strong></div></article><div class="answers">${item.answers.map(answer=>renderMainAnswer(answer,labels[answer.id],record.judgments?.[answer.id]?.tags||[])).join('')}</div><section class="card optional"><label><input id="suspect" type="checkbox" ${record.suspect?'checked':''}>חשד לבעיה בייחוס או במשפט</label><label><input id="example" type="checkbox" ${record.example?'checked':''}>דוגמה שכדאי לשמור למאמר</label>${record.has_previous_note?'<p class="small">הערה קודמת נשמרה ותוצג לאחר החשיפה.</p>':''}<label for="note">הערה קצרה (רשות)</label><textarea id="note" rows="2" placeholder="אפשר להשאיר ריק">${esc(record.note||'')}</textarea></section>`;
  }
  function brief(value){
    if(value===null||value===undefined||value==='')return 'לא זמין';
    if(typeof value==='boolean')return value?'כן':'לא';
    if(Array.isArray(value))return value.map(brief).join(' · ');
    if(typeof value==='object')return Object.entries(value).map(([key,val])=>`${key}: ${brief(val)}`).join(' · ');
    return String(value);
  }
  function tagsText(answer){const tags=(answer.tags||[]).filter(tag=>Object.hasOwn(TAGS,tag));return tags.length?tags.map(tag=>TAGS[tag]).join(' · '):'לא סומן';}
  function phaseText(value){return PHASES[value]||PHASES.unknown;}
  function judgmentPhaseText(answer){return labelOf(answer)?phaseText(answer.label_phase||answer.judgment_phase):'טרם נשפט';}
  function renderCase(value,full=false){
    if(typeof value!=='object'||value===null)return '';
    const inherited=value.inherited_from||(value.answers||[]).some(a=>a.label_phase==='prior_protocol'||a.judgment_phase==='prior_protocol');
    return `<article class="case"><strong><bdi>${esc(value.id||'')}</bdi></strong>${value.sentence?`<p>${esc(value.sentence)}</p>`:''}${value.gold!==undefined?`<p class="small">ייחוס המקור: ${esc(value.gold)}</p>`:''}${inherited?'<p class="notice">נשמרו שיפוטים מהפרוטוקול הקודם; אין להציגם כתיוג עצמאי חדש.</p>':''}${value.note?`<p>הערה אנושית: ${esc(value.note)}</p>`:''}${full&&value.previous_note?`<p>הערה מהפרוטוקול הקודם: ${esc(value.previous_note)}</p>`:''}${value.answers?.length?`<ul>${value.answers.map(a=>`<li><strong>${esc(full?(a.model||a.code||'תשובה'):('תשובה '+(a.code||'')))}</strong>: ${esc(a.text??'לא נמסר מענה')}<br><span class="small">שיפוט אנושי: ${esc(LABELS[labelOf(a)]||'טרם סומן')} · תגיות: ${esc(tagsText(a))}<br>מועד השיפוט: ${esc(judgmentPhaseText(a))} · מועד התגיות: ${a.tags?.length?esc(phaseText(a.tags_phase||a.tag_phase)):'לא סומן'}${full&&a.auto_score!==undefined?' · ציון אוטומטי שמור: '+esc(brief(a.auto_score)):''}${full&&a.comparison?' · '+esc(COMPARISONS[a.comparison]||'אין השוואה חד־משמעית'):''}</span></li>`).join('')}</ul>`:''}</article>`;
  }
  // The assistance view always uses a narrow allowlist, including after reveal.
  function renderMaskedDetails(details){return `<section class="card"><h3>מועמדים במקור</h3><p>${(details.candidates||[]).map(esc).join(' · ')||'לא זמינים מועמדים נוספים.'}</p><p class="small">פתיחת המועמדים מתועדת. זהויות, ניקוד ופרטי מקור אינם מוצגים כאן.</p></section>`;}
  function fullCasesTable(cases){
    const rows=cases.flatMap(c=>(c.answers||[]).map(a=>`<tr><td>${esc(c.id)}</td><td class="original-answer">${esc(c.sentence||'')}</td><td>${esc(a.original_answer_id||a.answer_id||a.id||'')}</td><td>${esc(a.source||c.source||'')}</td><td class="original-answer">${esc(a.text??'')}</td><td>${esc(c.gold)}</td><td>${esc(a.model)}</td><td>${esc(brief(a.auto_score))}</td><td>${esc(LABELS[labelOf(a)]||'טרם סומן')}</td><td>${esc(tagsText(a))}</td><td>${esc(c.note||'')}${c.previous_note?`<br>הערה קודמת: ${esc(c.previous_note)}`:''}</td><td>${esc(judgmentPhaseText(a))}${labelOf(a)&&a.exposure_evidence?.status?`<br>עדות חשיפה: ${esc({before_reveal:'לפי התיעוד, לפני חשיפה',after_reveal:'חשיפה מוקדמת מתועדת',unknown:'מצב קודם אינו ידוע'}[a.exposure_evidence.status]||'מצב קודם אינו ידוע')}`:''}<br>תגיות: ${a.tags?.length?esc(phaseText(a.tags_phase||a.tag_phase)):'לא סומן'}</td></tr>`));
    return `<div class="table-wrap"><table><thead><tr>${['מזהה משפט','משפט','מזהה תשובה מקורית','מקור','טקסט תשובה מקורי','ייחוס','מודל','ציון מקורי','שיפוט אנושי','תגיות','הערה','חשיפה בזמן השיפוט'].map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
  }
  function renderFullComposition(composition){
    if(!composition)return '';
    const groups={generation_fail_selection_pass:'כשל ביצירה והצלחה בבחירה של אותו מודל',both_fail:'כשל ביצירה ובבחירה של אותו מודל',both_generation_pass:'שתי תשובות היצירה קיבלו ציון נכון'};
    return `<section class="card"><h3>הרכב הרשימה והבדיקה בפועל</h3><div class="table-wrap"><table><thead><tr><th>קבוצה לפי הניקוד המקורי</th><th>נבחרו</th><th>נבדקו</th><th>הושלמו</th></tr></thead><tbody>${Object.entries(groups).map(([key,label])=>`<tr><td>${label}</td><td>${Number(composition.selected?.[key]||0)}</td><td>${Number(composition.reviewed?.[key]||0)}</td><td>${Number(composition.completed?.[key]||0)}</td></tr>`).join('')}</tbody></table></div><p class="small">״נבדקו״: יש לפחות שיפוט אחד. ״הושלמו״: סומנו שתי התשובות, כולל ״לא בטוח״. הרשימה אבחונית ואינה מדגם מייצג.</p><p>מקורות שנבחרו: ${esc(brief(composition.sources_selected||{}))}</p><p>מקורות שנבדקו: ${esc(brief(composition.sources_reviewed||{}))}</p>${composition.adjustments?.length?`<p>התאמות בבחירה: ${esc(brief(composition.adjustments))}</p>`:''}</section>`;
  }
  function renderSummaryContent(summary,allowFull=false){
    const full=allowFull&&summary.masked===false,c=summary.counts||{};
    const sections=full?[['accepted_human_rejected_auto','נפסלה אוטומטית והאדם קיבל'],['rejected_human_accepted_auto','התקבלה אוטומטית והאדם דחה'],['unresolved','מקרים לא מוכרעים'],['suspicions','חשדות לבעיה בייחוס או במשפט'],['examples','דוגמאות שסומנו למאמר']]:[['cases','השיפוטים והתגיות שנשמרו'],['unresolved','מקרים לא מוכרעים'],['suspicions','חשדות לבעיה בייחוס או במשפט'],['examples','דוגמאות שסומנו למאמר']];
    const tagCounts=Object.entries(TAGS).map(([key,label])=>`<span class="tag-count">${label}: ${Number(summary.tag_counts?.[key]||0)}</span>`).join('');
    const rule=summary.scoring_rule;
    const rules=full&&rule?`<section class="card"><h3>כלל הניקוד המקורי שנבדק</h3><p>${esc(rule.description||'')}</p><p>${esc(rule.normalization||'')}</p>${rule.saved_generation_answers_checked!==undefined?`<p>נבדקו ${Number(rule.saved_generation_answers_checked)} תשובות יצירה שמורות לשחזור הניקוד; ${(rule.score_mismatches||[]).length} אי־התאמות לכלל המשוחזר.</p>`:''}<p class="small">${esc(rule.provenance_limit||'')}</p></section>`:full&&summary.scoring_rules?`<section class="card"><h3>כלל הניקוד המקורי שנבדק</h3><p>${esc(brief(summary.scoring_rules))}</p></section>`:'';
    const recommendations=full&&summary.recommendations?.length?`<section class="card"><h3>המלצות לבדיקה ולהחלטה</h3><p class="small">לא שונה הניקוד הרשמי. תגית לצד פער אינה הוכחה לסיבת הפער; אלו הצעות לבדיקה ולא אימות עצמאי.</p>${summary.recommendations.map(r=>`<div class="recommendation">${typeof r==='string'?esc(r):`<h3>${esc(r.title||'הצעה לבדיקה')}</h3>${r.status?`<p class="small">${esc(r.status)}</p>`:''}<p><strong>כשל שהכלל עשוי לפתור:</strong> ${esc(r.possible_failure||'טרם פורט')}</p><p><strong>סיכון לקבלה שגויה:</strong> ${esc(r.false_acceptance_risk||'טרם פורט')}</p>${r.causal_limit?`<p class="small">${esc(r.causal_limit)}</p>`:''}${r.evidence?.length?`<p>מקרים לבדיקה:</p><ul>${r.evidence.map(e=>`<li><bdi>${esc(e.item_id)} / ${esc(e.answer_id)}</bdi> · ${esc(LABELS[e.label]||'טרם סומן')} · ציון שמור: ${esc(brief(e.auto_score))} · תגיות: ${esc(tagsText(e))}</li>`).join('')}</ul>`:'<p class="small">לא סומנו עדיין מקרים התומכים בכלל זה.</p>'}`}</div>`).join('')}${summary.next_step?`<p class="notice">${esc(summary.next_step)}</p>`:''}</section>`:'';
    return `<div class="summary-counts"><div><strong>${c.reviewed_items??((c.complete_items||0)+(c.partial_items||0))}</strong> משפטים שנבדקו</div><div><strong>${c.complete_items||0}</strong> משפטים מלאים</div><div><strong>${c.partial_items||0}</strong> משפטים חלקיים</div><div><strong>${c.judged_answers||0}</strong> תשובות שסומנו מתוך ${c.total_answers||0}</div></div><p class="small">השלמה מתייחסת לשני שיפוטים בלבד, כולל ״לא בטוח״. תגיות והערות אינן חובה ואינן משלימות שיפוט חסר.</p><section class="card"><h3>תגיות שסומנו</h3><div class="tag-counts">${tagCounts}</div><p class="small">היעדר תגית פירושו ״לא סומן״. תגיות מוצגות גם כשאין פער בין השיפוט לניקוד.</p></section>${sections.map(([key,title])=>`<section class="card"><h3>${title}</h3>${summary[key]?.length?summary[key].map(v=>renderCase(v,full)).join(''):'<p class="small">אין פריטים בסעיף זה.</p>'}</section>`).join('')}${full?`<section class="card"><h3>כל המקרים — כולל הסכמה בין השיפוט לניקוד</h3>${fullCasesTable(summary.cases||summary.all_cases||[])}</section>`:''}${full?renderFullComposition(summary.composition||summary.sample_composition):''}${rules}${recommendations}<section class="card"><h3>משפטים שלא הושלמו</h3><p>${summary.incomplete_ids?.length?summary.incomplete_ids.map(esc).join(' · '):c.complete_items===c.total_items?'כל המשפטים במסלול הושלמו.':'אפשר להמשיך בפריטים שטרם הושלמו.'}</p></section><section class="card"><h3>גבולות הסיכום</h3>${(Array.isArray(summary.limitations)?summary.limitations:[summary.limitations||'בדיקה איכותנית מצומצמת עם ייחוס מוצג. אין להסיק ממנה שיעורים לכל המאגר; סגנון התשובה עשוי לרמוז לזהות.']).map(text=>`<p>${esc(text)}</p>`).join('')}</section>`;
  }
  const api={judgmentPhaseText,TAGS,TAG_HELP,PHASES,tagsText,renderMaskedDetails,renderSummaryContent,fullCasesTable,LABELS,esc,labelOf,savedLabels,firstIncomplete,progressFor,highlightedSentence,renderMainAnswer,renderMain,renderCase};
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
    $('exposureNotice').hidden=!(state.revealed||state.exposed||state.prior_exposure);
    $('exposureNotice').textContent=state.revealed||state.exposed?'התוצאות כבר נחשפו. שינויים חדשים יתועדו לאחר חשיפה; תמונת המצב שלפניה נשמרת.':'העבודה והחשיפות מהפרוטוקול הקודם נשמרו. אין לראות בשיפוטים הקודמים שיפוט חדש ללא חשיפה.';
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
    const judgments={},tags={};
    for(const group of document.querySelectorAll('[data-answer]'))judgments[group.dataset.answer]=group.querySelector('input[type=radio]:checked')?.value||'';
    for(const group of document.querySelectorAll('[data-tags-answer]'))tags[group.dataset.tagsAnswer]=[...group.querySelectorAll('input:checked')].map(el=>el.value);
    return {item_id:currentItem.id,judgments,tags,suspect:$('suspect').checked,example:$('example').checked,note:$('note').value};
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
  function renderDetails(details){$('detailsView').innerHTML=renderMaskedDetails(details);}
  function renderSummary(summary){
    const full=Boolean(state.revealed||state.exposed)&&summary.masked===false;
    $('summaryTitle').textContent=full?'סיכום לאחר חשיפת התוצאות':'סיכום ביניים מוסתר';
    $('summaryIntro').textContent=full?'תמונת המצב שלפני החשיפה נשמרה. עריכות נוספות מסומנות לאחר חשיפה. זהו סיכום איכותני, ללא אומדן לכל המאגר.':'הסיכום מציג התקדמות, שיפוטים, תגיות והערות בלבד. אפשר לשמור ולצאת בלי לחשוף תוצאות.';
    $('summaryView').innerHTML=renderSummaryContent(summary,full);
    $('revealPanel').hidden=Boolean(state.revealed||state.exposed);
    $('fullDownloads').hidden=!Boolean(state.revealed||state.exposed);
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
  $('exportAnnotations').onclick=()=>run(()=>download('/api/short/export.json','short-review-masked-annotations.json'));
  $('exportSummary').onclick=()=>run(()=>download('/api/short/summary/export.md','short-review-masked-summary.md'));
  $('revealResults').onclick=()=>run(async()=>{await flush();status('שומר תמונת מצב ופותח תוצאות…');const result=await request('/api/short/reveal',{revision:state.revision});state=result.state;renderSummary(result.summary);updateProgress();showScreen('summaryScreen');lastError=null;status('תמונת המצב נשמרה · התוצאות נחשפו');window.scrollTo({top:0});});
  $('exportFull').onclick=()=>run(()=>download('/api/short/full-export.json','short-review-full-results.json'));
  $('exportFullSummary').onclick=()=>run(()=>download('/api/short/full-summary.md','short-review-full-summary.md'));
  $('retryLoad').onclick=()=>run(init);
  window.addEventListener('beforeunload',event=>{if(dirty||lastError){event.preventDefault();event.returnValue='';}});
  run(init);
})(typeof globalThis!=='undefined'?globalThis:this);
