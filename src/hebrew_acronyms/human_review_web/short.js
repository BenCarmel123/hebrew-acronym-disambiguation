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
  const FILTER_STATUSES=['exacttrim','technical','missing'];
  const REVIEW_STATUSES={human:'שיפוט אנושי קיים',pending:'ממתינה לבדיקה',exacttrim:'סוננה מכנית — התאמה מלאה',technical:'סוננה מכנית — נרמול טכני מצומצם',missing:'אין תשובה / כשל טכני מתועד'};
  function isContinuation(state){return Boolean(state.queues);}
  function canShowFull(state,summary,explicitFull=false){return (!isContinuation(state)||explicitFull)&&Boolean(state.revealed||state.exposed)&&summary.masked===false;}
  function isGroupView(view){return view==='groups'||view==='group_foreign';}
  function groupBackendView(view){return view==='group_foreign'?'foreign':'all';}
  function defaultView(state){return Array.isArray(state.group_queue)?'groups':'all';}
  function queueIds(state,view='all'){return isGroupView(view)?state.group_queues?.[groupBackendView(view)]||state.group_queue||[]:state.queues?state.queues[view]||[]:state.queue||[];}
  function firstIncomplete(state){return Array.isArray(state.group_queue)?state.group_queue[0]:isContinuation(state)?queueIds(state)[0]:state.queue.find(id=>state.records[id]?.completion?.status!=='complete');}
  function nextQueueId(before,current,remaining){const later=before.slice(before.indexOf(current)+1);return later.find(id=>remaining.includes(id))||remaining.find(id=>id!==current);}
  function editableAnswer(answer,view='all'){return view!=='filtered'&&!FILTER_STATUSES.includes(answer.review_status);}
  function renderFilteredAnswer(answer){
    const reasons={exacttrim:'התאמה של כל התשובה לייחוס לאחר הסרת רווחים בקצוות',technical:'התאמה של כל התשובה לאחר נרמול טכני מצומצם: NFC, רצפי רווחים וגרש/גרשיים מקבילים',missing:'אין תשובה / כשל טכני מתועד'};
    return `<article class="card filtered-answer"><h3>תשובה ${esc(answer.code)}</h3>${answer.occurrence_count>1?`<p class="small">טקסט זהה ב־${Number(answer.occurrence_count)} מופעים באותו משפט. שיפוט אחד יחול עליהם, ולא ייספר כהכרעות עצמאיות.</p>`:''}<div class="answer-text" dir="auto">${esc(answer.text??'לא נמסר מענה')}</div><p class="small">${esc(reasons[answer.review_status]||'התשובה אינה דורשת שיפוט בתור הנוכחי.')}</p><p class="small">${answer.review_status==='missing'?'אין כאן שיפוט אנושי או תגית ג׳יבריש.':'סוננה מכנית; לא אושרה בידי אדם. התאמה לייחוס אינה אישור שהייחוס נכון.'}</p>${FILTER_STATUSES.includes(answer.review_status)?`<button data-restore-answer="${esc(answer.id)}" class="quiet">החזרה לבדיקה</button>`:''}</article>`;
  }
  function renderPartition(counts){
    if(counts.source_answers===undefined)return '';
    const rows=[['human_answers','נשפטו בידי אדם, כולל ״לא בטוח״'],['exacttrim_answers','סוננו: התאמה מלאה לאחר הסרת רווחים בקצוות'],['technical_answers','סוננו: התאמה לאחר נרמול טכני מצומצם'],['missing_answers','אין תשובה / כשל טכני מתועד'],['pending_answers','תשובות שממתינות לבדיקה']];
    const sum=rows.reduce((n,[key])=>n+Number(counts[key]||0),0),decisions=Number(counts.pending_decisions||0);
    return `<section class="card"><h3>תשובות המקור והעבודה שנותרה</h3><table><tbody><tr><th>תשובות המקור</th><td>${Number(counts.source_answers)}</td></tr>${rows.map(([key,label])=>`<tr><td>${label}</td><td>${Number(counts[key]||0)}</td></tr>`).join('')}<tr><th>סכום הקבוצות הראשיות</th><td>${sum}</td></tr></tbody></table><p><strong>${decisions}</strong> הקשרי תשובה ממתינים להכרעה.</p>${counts.grouped_pending_decisions!==undefined?`<p>${Number(counts.grouped_pending_decisions)} קבוצות לתצוגה; בחירה משותפת עשויה לחסוך עד ${Number(counts.cross_context_savings||0)} פעולות, בהתאם לחריגים. כל ההקשרים עדיין דורשים קריאה.</p><p class="small">${Number(counts.human_decisions||0)} פעולות שיפוט שמורות מייצגות ${Number(counts.human_answers||0)} מופעי תשובה. החלת בחירה משותפת על כמה הקשרים אינה כמה הכרעות עצמאיות.</p>`:''}<p class="small">מופעים זהים באותו משפט חוסכים ${Number(counts.duplicate_savings||0)} הצגות נוספות; החיסכון אינו מופחת שוב מספירת המקור. ${Number(counts.foreign_pending_answers||0)} מהממתינות מכילות אותיות משפה אחרת — תכונה חופפת, לא קבוצה שמפחיתים שוב.</p><p class="small">אומדן גס: ${Math.ceil(decisions/2)}–${Math.ceil(decisions*1.5)} דקות, בהנחת חצי דקה עד דקה וחצי להכרעה. הסינון המכני אינו שיפוט אנושי, והבדיקה המורחבת אינה מדגם אקראי.</p></section>`;
  }
  function progressFor(item,record={}){
    const labels=savedLabels(record);
    return {judged:item.answers.filter(a=>editableAnswer(a)&&Object.hasOwn(LABELS,labels[a.id])).length,total:item.answers.filter(a=>editableAnswer(a)).length};
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
    return `<article class="answer-row" data-row-answer="${esc(answer.id)}" tabindex="0" aria-label="תשובה ${esc(answer.code)}"><div class="answer-main"><h3>תשובה ${esc(answer.code)} <span class="active-indicator" aria-hidden="true">פעילה</span>${answer.occurrence_count>1?` <span class="duplicate-badge" title="שיפוט אחד חל על ${Number(answer.occurrence_count)} מופעים זהים באותו משפט; לא הכרעות עצמאיות">×${Number(answer.occurrence_count)} זהה</span>`:''}</h3><div class="answer-text" dir="auto">${esc(answer.text??'לא נמסר מענה')}</div></div><fieldset class="choices" data-answer="${esc(answer.id)}"><legend class="sr-only">השיפוט שלך לתשובה ${esc(answer.code)}</legend>${Object.entries(LABELS).map(([value,label],index)=>`<label class="choice"><input type="radio" name="judgment-${esc(answer.id)}" value="${value}" ${selected===value?'checked':''}><span><small aria-hidden="true">${index+1}</small> ${label}</span></label>`).join('')}</fieldset><fieldset class="tag-chips" data-tags-answer="${esc(answer.id)}"><legend class="sr-only">תופעות בולטות (רשות, אפשר כמה)</legend>${Object.entries(TAGS).map(([value,label])=>`<label class="tag-chip" title="${esc(TAG_HELP[value])}"><input type="checkbox" aria-label="${esc(label)}" value="${value}" ${tags.includes(value)?'checked':''}><span>${label}</span></label>`).join('')}</fieldset></article>`;
  }
  function renderMain(item,record={},view='all'){
    const labels=savedLabels(record);
    return `<article class="card context-card"><div class="sentence">${highlightedSentence(item)}</div><div class="context-meta"><span>קיצור: <strong>${esc(item.acronym)}</strong></span><span>ייחוס (עשוי להיות שגוי): <strong>${esc(item.gold??'לא זמין')}</strong></span>${item.prior_reused?'<span class="small" title="השיפוטים הקודמים נשמרו. אין צורך לבדוק שוב תשובה שכבר נשפטה; משלימים רק שיפוטים חסרים.">עבודה קודמת נשמרה</span>':''}</div></article><div class="answers">${item.answers.map(answer=>editableAnswer(answer,view)?renderMainAnswer(answer,labels[answer.id],record.judgments?.[answer.id]?.tags||[]):renderFilteredAnswer(answer)).join('')||'<p class="empty">אין כאן תשובות נוספות שדורשות שיפוט בתור הנוכחי.</p>'}</div><section class="optional compact-optional"><label><input id="suspect" type="checkbox" ${record.suspect?'checked':''}>חשד בייחוס / במשפט</label><label><input id="example" type="checkbox" ${record.example?'checked':''}>דוגמה למאמר</label><label for="note" class="sr-only">הערה קצרה (רשות)</label><textarea id="note" rows="1" placeholder="הערה (רשות)">${esc(record.note||'')}</textarea>${record.has_previous_note?'<span class="small" title="הערה קודמת נשמרה ותוצג לאחר החשיפה.">הערה קודמת שמורה</span>':''}</section>`;
  }
  function renderGroup(group){
    const form=group.form||{},exceptions=form.exceptions||{},tags=form.tags||[];
    return `<article class="card context-card"><div class="context-meta"><span>קיצור: <strong>${esc(group.acronym)}</strong></span><span>ייחוס (עשוי להיות שגוי): <strong>${esc(group.gold)}</strong></span><span>${group.contexts.length} הקשרים</span></div><div class="shared-answer"><strong>התשובה המשותפת</strong><div class="answer-text" dir="auto">${esc(group.text??'')}</div></div></article><section class="group-contexts" aria-label="כל המשפטים בקבוצה">${group.contexts.map((context,index)=>{const exception=exceptions[context.id]||{};return `<article class="group-context" data-group-context="${esc(context.id)}"><div class="group-sentence"><span class="context-number">${index+1}</span><div class="sentence">${highlightedSentence(context)}</div></div><div class="context-override"><label for="override-${esc(context.id)}">חריג למשפט ${index+1}</label><select id="override-${esc(context.id)}" data-context-override><option value="">כמו הבחירה המשותפת</option>${Object.entries({...LABELS,defer:'להשאיר להמשך'}).map(([value,label])=>`<option value="${value}" ${exception.label===value?'selected':''}>${label}</option>`).join('')}</select></div><div class="context-options"><label><input type="checkbox" data-context-suspect ${context.suspect?'checked':''}>חשד בייחוס / במשפט</label><label><input type="checkbox" data-context-example ${context.example?'checked':''}>דוגמה למאמר</label><label class="sr-only" for="context-note-${esc(context.id)}">הערה למשפט ${index+1}</label><input id="context-note-${esc(context.id)}" data-context-note type="text" value="${esc(context.note||'')}" placeholder="הערה (רשות)"></div></article>`;}).join('')}</section><section class="group-decision" tabindex="0" id="groupDecision"><h3>בחירה משותפת לכל המשפטים המוצגים</h3><p class="small">קרא את כל ההקשרים. החריגים גוברים על הבחירה המשותפת; ״להשאיר להמשך״ אינו מקבל שיפוט.</p><fieldset class="choices" data-group-label><legend class="sr-only">שיפוט משותף לכל ההקשרים</legend>${Object.entries(LABELS).map(([value,label],index)=>`<label class="choice"><input type="radio" name="group-label" value="${value}" ${form.label===value?'checked':''}><span><small aria-hidden="true">${index+1}</small> ${label} בכל ההקשרים</span></label>`).join('')}</fieldset><fieldset class="tag-chips" data-group-tags><legend>תגיות משותפות (רשות)</legend>${Object.entries(TAGS).map(([value,label])=>`<label class="tag-chip" title="${esc(TAG_HELP[value])}"><input type="checkbox" aria-label="${esc(label)}" value="${value}" ${tags.includes(value)?'checked':''}><span>${label}</span></label>`).join('')}</fieldset></section>`;
  }
  function groupExceptionLabel(previous,chosen,common){return !chosen&&!common&&previous?'defer':chosen;}
  function buildGroupPayload(group,form,view='groups'){
    const exceptions={},context_updates={};
    for(const context of group.contexts){const supplied=form.contexts?.[context.id]||{};const override=groupExceptionLabel(group.form?.exceptions?.[context.id]?.label,supplied.override,form.label);if(override)exceptions[context.id]={label:override};context_updates[context.id]={suspect:supplied.suspect??Boolean(context.suspect),example:supplied.example??Boolean(context.example),note:supplied.note??context.note??''};}
    return {group_id:group.id,open_id:group.open_id,view:groupBackendView(view),common_label:form.label||'',common_tags:form.tags||[],common_note:group.form?.note||'',exceptions,context_updates};
  }
  function shortcutIntent(event,activeId){
    if(event.repeat||event.ctrlKey||event.altKey||event.metaKey||event.shiftKey)return null;
    const target=event.target||{},tag=String(target.tagName||'').toUpperCase(),type=String(target.type||'text').toLowerCase();
    if(target.isContentEditable||tag==='TEXTAREA'||tag==='SELECT'||(tag==='INPUT'&&!['radio','checkbox','button','submit','reset'].includes(type)))return null;
    if(event.key==='Enter')return tag==='BUTTON'||tag==='A'||target.closest?.('button,a')?null:{type:'next'};
    const label={'1':'fits','2':'not_fits','3':'unsure'}[event.key];
    return label&&activeId?{type:'label',answerId:activeId,label}:null;
  }
  function nextActiveAnswer(ids,labels,currentId){
    const index=ids.indexOf(currentId),ordered=[...ids.slice(index+1),...ids.slice(0,index)];
    return ordered.find(id=>!Object.hasOwn(LABELS,labels[id]))||currentId||ids[0];
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
    return `<article class="case"><strong><bdi>${esc(value.id||'')}</bdi></strong>${value.sentence?`<p>${esc(value.sentence)}</p>`:''}${value.gold!==undefined?`<p class="small">ייחוס המקור: ${esc(value.gold)}</p>`:''}${inherited?'<p class="notice">נשמרו שיפוטים מהפרוטוקול הקודם; אין להציגם כתיוג עצמאי חדש.</p>':''}${value.note?`<p>הערה אנושית: ${esc(value.note)}</p>`:''}${full&&value.previous_note?`<p>הערה מהפרוטוקול הקודם: ${esc(value.previous_note)}</p>`:''}${value.answers?.length?`<ul>${value.answers.map(a=>`<li><strong>${esc(full?(a.model||a.code||'תשובה'):('תשובה '+(a.code||'')))}</strong>: ${esc(a.text??'לא נמסר מענה')}<br><span class="small">${a.review_status?`מצב: ${esc(REVIEW_STATUSES[a.review_status]||'לא ידוע')}<br>`:''}שיפוט אנושי: ${esc(LABELS[labelOf(a)]||'טרם סומן')} · תגיות: ${esc(tagsText(a))}<br>מועד השיפוט: ${esc(judgmentPhaseText(a))} · מועד התגיות: ${a.tags?.length?esc(phaseText(a.tags_phase||a.tag_phase)):'לא סומן'}${full&&a.auto_score!==undefined?' · ציון אוטומטי שמור: '+esc(brief(a.auto_score)):''}${full&&a.comparison?' · '+esc(COMPARISONS[a.comparison]||'אין השוואה חד־משמעית'):''}</span></li>`).join('')}</ul>`:''}</article>`;
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
    return `${renderPartition(c)}${c.source_answers===undefined?`<div class="summary-counts"><div><strong>${c.reviewed_items??((c.complete_items||0)+(c.partial_items||0))}</strong> משפטים שנבדקו</div><div><strong>${c.complete_items||0}</strong> משפטים מלאים</div><div><strong>${c.partial_items||0}</strong> משפטים חלקיים</div><div><strong>${c.judged_answers||0}</strong> תשובות שסומנו מתוך ${c.total_answers||0}</div></div>`:''}<p class="small">${c.source_answers!==undefined?'״לא בטוח״ נשמר כשיפוט קיים בתור נפרד. סינון מכני אינו אישור אנושי.':'השלמה מתייחסת לשני שיפוטים בלבד, כולל ״לא בטוח״.'} תגיות והערות אינן חובה ואינן משלימות שיפוט חסר.</p><section class="card"><h3>תגיות שסומנו</h3><div class="tag-counts">${tagCounts}</div><p class="small">היעדר תגית פירושו ״לא סומן״. תגיות מוצגות גם כשאין פער בין השיפוט לניקוד.${c.source_answers!==undefined?' הספירה היא לפי מופעי תשובה; שיפוט יחיד שהוחל על מופעים זהים אינו כמה הכרעות עצמאיות.':''}</p></section>${sections.map(([key,title])=>`<section class="card"><h3>${title}</h3>${summary[key]?.length?summary[key].map(v=>renderCase(v,full)).join(''):'<p class="small">אין פריטים בסעיף זה.</p>'}</section>`).join('')}${full?`<section class="card"><h3>כל המקרים — כולל הסכמה בין השיפוט לניקוד</h3>${fullCasesTable(summary.cases||summary.all_cases||[])}</section>`:''}${full?renderFullComposition(summary.composition||summary.sample_composition):''}${rules}${recommendations}<section class="card"><h3>${c.source_answers!==undefined?'משפטים עם תשובות שממתינות לבדיקה':'משפטים שלא הושלמו'}</h3><p>${summary.incomplete_ids?.length?summary.incomplete_ids.map(esc).join(' · '):c.source_answers!==undefined&&c.pending_answers===0?'אין תשובות שממתינות לבדיקה; שיפוטי לא בטוח נשארים בתור החזרה.':c.source_answers===undefined&&c.complete_items===c.total_items?'כל המשפטים במסלול הושלמו.':'אפשר להמשיך בפריטים שטרם הושלמו.'}</p></section><section class="card"><h3>גבולות הסיכום</h3>${(Array.isArray(summary.limitations)?summary.limitations:[summary.limitations||'בדיקה איכותנית מצומצמת עם ייחוס מוצג. אין להסיק ממנה שיעורים לכל המאגר; סגנון התשובה עשוי לרמוז לזהות.']).map(text=>`<p>${esc(text)}</p>`).join('')}</section>`;
  }
  const api={groupExceptionLabel,isGroupView,groupBackendView,defaultView,renderGroup,buildGroupPayload,shortcutIntent,nextActiveAnswer,canShowFull,isContinuation,queueIds,nextQueueId,editableAnswer,renderFilteredAnswer,renderPartition,judgmentPhaseText,TAGS,TAG_HELP,PHASES,tagsText,renderMaskedDetails,renderSummaryContent,fullCasesTable,LABELS,esc,labelOf,savedLabels,firstIncomplete,progressFor,highlightedSentence,renderMainAnswer,renderMain,renderCase};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  root.ShortReview=api;
  if(typeof document==='undefined')return;

  const $=id=>document.getElementById(id);
  let state,currentItem,dirty=false,timer=null,pending=Promise.resolve(),lastError=null,view='all',routeIds=[],currentPosition=-1,editVersion=0,lockCount=0,activeAnswerId=null;
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
  function locked(task,blocking=true){
    // Autosave stays serialized but leaves controls usable; a navigation click
    // immediately locks the form and waits behind any save already in flight.
    if(blocking){lockCount++;document.querySelector('main').inert=true;}
    const operation=pending.then(async()=>{
      try{return await task();}finally{if(blocking){lockCount--;document.querySelector('main').inert=lockCount>0;}}
    });
    pending=operation.catch(()=>{});
    return operation;
  }
  function run(task,blocking=true){locked(task,blocking).catch(showError);}
  function showScreen(name){for(const id of ['reviewScreen','detailsScreen','summaryScreen','startupError'])$(id).hidden=id!==name;}
  function updateProgress(){
    const counts=state.counts||{},total=counts.total_items??state.queue.length;
    $('progress').textContent=isContinuation(state)?`${counts.pending_answers||0} תשובות ממתינות`:`${counts.complete_items||0} מתוך ${total} משפטים הושלמו`;
    $('countDetail').textContent=isGroupView(view)?`${counts.grouped_pending_decisions??state.group_queue?.length??0} קבוצות · ${counts.pending_decisions||0} הקשרים לבדיקה`:isContinuation(state)?`${counts.pending_decisions||0} הכרעות נדרשות · ${counts.human_answers||0} תשובות עם שיפוט אנושי`:`נבדקו ${counts.reviewed_items??((counts.complete_items||0)+(counts.partial_items||0))} · ${counts.partial_items||0} חלקיים`;
    $('exposureNotice').hidden=!(state.revealed||state.exposed||state.prior_exposure);
    $('exposureNotice').textContent=isContinuation(state)?'החשיפות הקודמות נשמרו; הזהויות והניקוד מוסתרים כעת.':state.revealed||state.exposed?'התוצאות כבר נחשפו. שינויים חדשים יתועדו לאחר חשיפה; תמונת המצב שלפניה נשמרת.':'העבודה והחשיפות מהפרוטוקול הקודם נשמרו. אין לראות בשיפוטים הקודמים שיפוט חדש ללא חשיפה.';
    $('reviewer').textContent=`מתייג/ת: ${state.reviewer||'לא זמין'}`;
    $('queueView').value=view;
    $('queueHint').textContent={groups:'תשובה זהה לגמרי, אותו קיצור ואותו ייחוס. כל ההקשרים מוצגים; רק בחירה שלך מחילה שיפוט משותף.',group_foreign:'קבוצות ממתינות עם אותיות משפה אחרת; אין תגית או שיפוט אוטומטיים.',all:'כל תשובה שטרם נשפטה ולא סוננה נמצאת כאן. ״לא בטוח״ נמצא בתור נפרד.',foreign:'אותיות משפה אחרת הן סימון מכני בלבד; אין כאן שיפוט או תגית ג׳יבריש אוטומטיים.',unsure:'שיפוטים קיימים שסומנו ״לא בטוח״. חזרה אליהם היא בחירה נפרדת.',suspicions:'מקרים עם חשד לייחוס או למשפט נשארים נגישים גם כאשר תשובה סוננה.',filtered:'עיון בלבד בתשובות שסוננו; אפשר להחזיר תשובה לתור הממתינות.'}[view];
    $('queueView').title=$('queueHint').textContent;
    if(currentItem){
      const p=isGroupView(view)?{judged:currentItem.contexts.filter(c=>c.answer_ids?.every(id=>labelOf(state.records[c.id]?.judgments?.[id]))).length,total:currentItem.contexts.length}:progressFor(currentItem,state.records[currentItem.id]);
      $('itemProgress').textContent=isGroupView(view)?`${p.judged}/${p.total} הקשרים עם שיפוט שמור`:view==='filtered'?'עיון במסוננות — אין צורך לשפוט כאן':`${p.judged}/${p.total} הכרעות בכרטיסים המוצגים נשמרו`;
      $('position').textContent=currentPosition>=0?`${isGroupView(view)?'קבוצה':'משפט'} ${currentPosition+1} מתוך ${routeIds.length} בתור`:'המשפט הנוכחי';
      $('previous').disabled=currentPosition<=0;
      $('saveNext').textContent=view==='filtered'?'הבא ←':'שמירה והבא ←';
    } else {$('itemProgress').textContent='';$('position').textContent='';}
  }
  function formPayload(){
    if(isGroupView(view)){
      const contexts={};
      for(const row of $('itemView').querySelectorAll('[data-group-context]'))contexts[row.dataset.groupContext]={override:row.querySelector('[data-context-override]').value,suspect:row.querySelector('[data-context-suspect]').checked,example:row.querySelector('[data-context-example]').checked,note:row.querySelector('[data-context-note]').value};
      return buildGroupPayload(currentItem,{label:$('itemView').querySelector('[data-group-label] input:checked')?.value||'',tags:[...$('itemView').querySelectorAll('[data-group-tags] input:checked')].map(el=>el.value),contexts},view);
    }
    const judgments={},tags={};
    for(const group of document.querySelectorAll('[data-answer]'))judgments[group.dataset.answer]=group.querySelector('input[type=radio]:checked')?.value||'';
    for(const group of document.querySelectorAll('[data-tags-answer]'))tags[group.dataset.tagsAnswer]=[...group.querySelectorAll('input:checked')].map(el=>el.value);
    return {item_id:currentItem.id,view,judgments,tags,suspect:$('suspect').checked,example:$('example').checked,note:$('note').value};
  }
  async function saveNow(){
    clearTimeout(timer);
    if(!dirty||!currentItem)return;
    const payload=formPayload(),savedItem=currentItem.id,savedVersion=editVersion;
    status('שומר…');
    const result=await request(isGroupView(view)?'/api/short/group-save':'/api/short/save',{revision:state.revision,...payload});
    if(isGroupView(view)){state=result.state;if(currentItem?.id===savedItem)currentItem=result.group;}else state=result;
    if(currentItem?.id===savedItem&&editVersion===savedVersion)dirty=false;
    lastError=null;updateProgress();status(dirty?'השינוי הקודם נשמר · שינויים נוספים ממתינים לשמירה…':'נשמר בדיסק · אפשר להמשיך או לעצור');
  }
  async function flush(){clearTimeout(timer);await saveNow();}
  function markDirty(){editVersion++;dirty=true;lastError=null;clearTimeout(timer);status('שומר את השינוי…');timer=setTimeout(()=>run(saveNow,false),500);}
  function setActiveAnswer(id,focus=false){
    const rows=[...$('itemView').querySelectorAll('[data-row-answer]')];
    const selected=rows.find(row=>row.dataset.rowAnswer===id);
    activeAnswerId=selected?id:null;
    for(const row of rows){const active=row===selected;row.classList.toggle('active-row',active);row.setAttribute('aria-current',active?'true':'false');}
    const code=currentItem?.answers?.find(answer=>answer.id===activeAnswerId)?.code;
    $('activeRowHint').textContent=code?`1–3: תשובה ${code}`:'';
    if(focus&&selected)selected.focus({preventScroll:true});
  }
  function renderItem(){
    $('itemView').innerHTML=isGroupView(view)?renderGroup(currentItem):renderMain(currentItem,state.records[currentItem.id],view);
    $('itemView').querySelectorAll('[data-row-answer]').forEach(row=>{row.addEventListener('click',()=>setActiveAnswer(row.dataset.rowAnswer));row.addEventListener('focusin',()=>setActiveAnswer(row.dataset.rowAnswer));});
    const ids=(currentItem.answers||[]).filter(answer=>editableAnswer(answer,view)).map(answer=>answer.id),labels=savedLabels(state.records[currentItem.id]);
    if(isGroupView(view)){activeAnswerId='group-common';$('activeRowHint').textContent='1–3: לכל ההקשרים';}else setActiveAnswer(ids.find(id=>!Object.hasOwn(LABELS,labels[id]))||ids[0],true);
    $('itemView').querySelectorAll('input,textarea,select').forEach(element=>element.addEventListener('input',event=>{if(isGroupView(view)&&event?.target?.matches('[data-context-override]')){const select=event.target,id=select.closest('[data-group-context]').dataset.groupContext;select.value=groupExceptionLabel(currentItem.form?.exceptions?.[id]?.label,select.value,$('itemView').querySelector('[data-group-label] input:checked')?.value||'');}markDirty();}));
    $('itemView').querySelectorAll('[data-restore-answer]').forEach(button=>button.onclick=()=>run(async()=>{const id=currentItem.id;await flush();state=await request('/api/short/restore',{revision:state.revision,item_id:id,answer_id:button.dataset.restoreAnswer});view='all';routeIds=queueIds(state,view).slice();await openItem(id);status('התשובה הוחזרה לבדיקה; לא נוצר שיפוט אנושי אוטומטי.');}));
    updateProgress();showScreen('reviewScreen');
  }
  async function openItem(id){
    await flush();
    if(!id){activeAnswerId=null;$('activeRowHint').textContent='';currentItem=null;currentPosition=-1;dirty=false;$('itemView').innerHTML='<p class="empty">אין תשובות בתור שנבחר. אפשר לבחור ״כל התשובות הממתינות״ או לפתוח סיכום.</p>';$('saveNext').disabled=true;$('previous').disabled=true;$('detailsButton').disabled=true;updateProgress();showScreen('reviewScreen');status('התור שנבחר ריק · כל העבודה נשמרה');return;}
    status('טוען פריט…');
    const result=await request(isGroupView(view)?'/api/short/group-open':'/api/short/open',isGroupView(view)?{revision:state.revision,group_id:id,view:groupBackendView(view)}:{revision:state.revision,item_id:id,view});
    state=result.state;currentItem=isGroupView(view)?result.group:result.item;currentPosition=routeIds.indexOf(id);dirty=false;lastError=null;$('saveNext').disabled=false;$('detailsButton').disabled=false;renderItem();
    status('העבודה השמורה נטענה · שינויים נשמרים אוטומטית בדיסק');
  }
  async function switchView(nextView){await flush();view=nextView;routeIds=queueIds(state,view).slice();await openItem(routeIds[0]);}
  function renderDetails(details){$('detailsView').innerHTML=details.contexts?details.contexts.map((context,index)=>`<section class="card"><h3>משפט ${index+1}</h3><p>${esc(context.sentence)}</p>${renderMaskedDetails(context)}</section>`).join(''):renderMaskedDetails(details);}
  function renderSummary(summary,explicitFull=false){
    const full=canShowFull(state,summary,explicitFull);
    $('summaryTitle').textContent=full?'סיכום לאחר חשיפת התוצאות':'סיכום ביניים מוסתר';
    $('summaryIntro').textContent=full?'תמונת המצב שלפני החשיפה נשמרה. עריכות נוספות מסומנות לאחר חשיפה. זהו סיכום איכותני, ללא אומדן לכל המאגר.':'הסיכום מציג התקדמות, שיפוטים, תגיות והערות בלבד. אפשר לשמור ולצאת בלי לחשוף תוצאות.';
    $('summaryView').innerHTML=renderSummaryContent(summary,full);
    $('revealPanel').hidden=isContinuation(state)?explicitFull:Boolean(state.revealed||state.exposed);
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
      state=initial;$('qaBanner').hidden=!session.qa;view=defaultView(state);routeIds=queueIds(state,view).slice();updateProgress();const next=firstIncomplete(state);if(isContinuation(state)||next||!state.queue.length)await openItem(next);else await openSummary();
    }catch(error){showError(error);if(!currentItem)showScreen('startupError');}
  }
  $('previous').onclick=()=>run(()=>openItem(routeIds[currentPosition-1]));
  $('queueView').onchange=()=>{const selected=$('queueView').value;run(()=>switchView(selected));};
  $('filteredButton').onclick=()=>run(()=>switchView('filtered'));
  $('saveNext').onclick=()=>run(async()=>{const before=routeIds.slice(),id=currentItem.id;await flush();const remaining=queueIds(state,view);const next=isContinuation(state)?nextQueueId(before,id,remaining):before[before.indexOf(id)+1];routeIds=remaining.slice();if(next)await openItem(next);else await openSummary();});
  $('stopSummary').onclick=()=>run(openSummary);
  $('detailsButton').onclick=()=>run(async()=>{await flush();const result=await request(isGroupView(view)?'/api/short/group-details':'/api/short/details',isGroupView(view)?{revision:state.revision,group_id:currentItem.id,open_id:currentItem.open_id}:{revision:state.revision,item_id:currentItem.id});state=result.state;renderDetails(result.details);showScreen('detailsScreen');status('פתיחת הפרטים תועדה · העבודה נשמרה');window.scrollTo({top:0});});
  $('backDetails').onclick=()=>{showScreen('reviewScreen');window.scrollTo({top:0});};
  $('backSummary').onclick=()=>run(async()=>{if(!currentItem){routeIds=queueIds(state,view).slice();await openItem(routeIds[0]);}else showScreen('reviewScreen');window.scrollTo({top:0});});
  $('exportAnnotations').onclick=()=>run(()=>download('/api/short/export.json','short-review-masked-annotations.json'));
  $('exportSummary').onclick=()=>run(()=>download('/api/short/summary/export.md','short-review-masked-summary.md'));
  $('revealResults').onclick=()=>run(async()=>{await flush();status('שומר תמונת מצב ופותח תוצאות…');const result=await request('/api/short/reveal',{revision:state.revision});state=result.state;renderSummary(result.summary,true);updateProgress();showScreen('summaryScreen');lastError=null;status('תמונת המצב נשמרה · התוצאות נחשפו');window.scrollTo({top:0});});
  $('exportFull').onclick=()=>run(()=>download('/api/short/full-export.json','short-review-full-results.json'));
  $('exportFullSummary').onclick=()=>run(()=>download('/api/short/full-summary.md','short-review-full-summary.md'));
  $('retryLoad').onclick=()=>run(init);
  window.addEventListener('keydown',event=>{
    if($('reviewScreen').hidden||document.querySelector('main').inert)return;
    const intent=shortcutIntent(event,activeAnswerId);if(!intent)return;
    if(intent.type==='next'){if($('saveNext').disabled)return;event.preventDefault();$('saveNext').onclick();return;}
    if(isGroupView(view)){const radio=$('itemView').querySelector(`[data-group-label] input[value="${intent.label}"]`);if(radio){event.preventDefault();radio.checked=true;markDirty();}return;}
    const rows=[...$('itemView').querySelectorAll('[data-row-answer]')],row=rows.find(row=>row.dataset.rowAnswer===intent.answerId);
    const radio=row?.querySelector(`input[type=radio][value="${intent.label}"]`);if(!radio)return;
    event.preventDefault();radio.checked=true;markDirty();
    const labels=Object.fromEntries(rows.map(r=>[r.dataset.rowAnswer,r.querySelector('input[type=radio]:checked')?.value||'']));
    setActiveAnswer(nextActiveAnswer(rows.map(r=>r.dataset.rowAnswer),labels,intent.answerId),true);
  });
  window.addEventListener('beforeunload',event=>{if(dirty||lastError){event.preventDefault();event.returnValue='';}});
  run(init);
})(typeof globalThis!=='undefined'?globalThis:this);
