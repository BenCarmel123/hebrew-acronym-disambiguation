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
  function queueIds(state,view='all'){return state.queues?state.queues[view]||[]:state.queue||[];}
  function firstIncomplete(state){return isContinuation(state)?queueIds(state)[0]:state.queue.find(id=>state.records[id]?.completion?.status!=='complete');}
  function nextQueueId(before,current,remaining){const later=before.slice(before.indexOf(current)+1);return later.find(id=>remaining.includes(id))||remaining.find(id=>id!==current);}
  function editableAnswer(answer,view='all'){return view!=='filtered'&&!FILTER_STATUSES.includes(answer.review_status);}
  function renderFilteredAnswer(answer){
    const reasons={exacttrim:'התאמה של כל התשובה לייחוס לאחר הסרת רווחים בקצוות',technical:'התאמה של כל התשובה לאחר נרמול טכני מצומצם: NFC, רצפי רווחים וגרש/גרשיים מקבילים',missing:'אין תשובה / כשל טכני מתועד'};
    return `<article class="card filtered-answer"><h3>תשובה ${esc(answer.code)}</h3>${answer.occurrence_count>1?`<p class="small">טקסט זהה ב־${Number(answer.occurrence_count)} מופעים באותו משפט. שיפוט אחד יחול עליהם, ולא ייספר כהכרעות עצמאיות.</p>`:''}<div class="answer-text" dir="auto">${esc(answer.text??'לא נמסר מענה')}</div><p class="small">${esc(reasons[answer.review_status]||'התשובה אינה דורשת שיפוט בתור הנוכחי.')}</p><p class="small">${answer.review_status==='missing'?'אין כאן שיפוט אנושי או תגית ג׳יבריש.':'סוננה מכנית; לא אושרה בידי אדם. התאמה לייחוס אינה אישור שהייחוס נכון.'}</p>${answer.can_restore!==false&&FILTER_STATUSES.includes(answer.review_status)?`<button data-restore-answer="${esc(answer.id)}" class="quiet">החזרה לבדיקה</button>`:''}</article>`;
  }
  function renderPartition(counts){
    if(counts.source_answers===undefined)return '';
    const rows=[['human_answers','נשפטו בידי אדם, כולל ״לא בטוח״'],['exacttrim_answers','סוננו: התאמה מלאה לאחר הסרת רווחים בקצוות'],['technical_answers','סוננו: התאמה לאחר נרמול טכני מצומצם'],['missing_answers','אין תשובה / כשל טכני מתועד'],['pending_answers','תשובות שממתינות לבדיקה']];
    if(counts.reused_answers!==undefined){rows[0][1]='שיפוטים חדשים, כולל ״לא בטוח״';rows.splice(1,0,['reused_answers','הכרעות היסטוריות שאושרו לשימוש חוזר — לא שיפוטים חדשים']);}
    const sum=rows.reduce((n,[key])=>n+Number(counts[key]||0),0),decisions=Number(counts.pending_decisions||0);
    return `<section class="card"><h3>תשובות המקור והעבודה שנותרה</h3><table><tbody><tr><th>תשובות המקור</th><td>${Number(counts.source_answers)}</td></tr>${rows.map(([key,label])=>`<tr><td>${label}</td><td>${Number(counts[key]||0)}</td></tr>`).join('')}<tr><th>סכום הקבוצות הראשיות</th><td>${sum}</td></tr></tbody></table><p><strong>${decisions}</strong> הכרעות אנושיות נדרשות כעת.</p><p class="small">מופעים זהים באותו משפט חוסכים ${Number(counts.duplicate_savings||0)} הצגות נוספות; החיסכון אינו מופחת שוב מספירת המקור. ${Number(counts.foreign_pending_answers||0)} מהממתינות מכילות אותיות משפה אחרת — תכונה חופפת, לא קבוצה שמפחיתים שוב.</p><p class="small">אומדן גס: ${Math.ceil(decisions/2)}–${Math.ceil(decisions*1.5)} דקות, בהנחת חצי דקה עד דקה וחצי להכרעה. הסינון המכני אינו שיפוט אנושי, והבדיקה המורחבת אינה מדגם אקראי.</p></section>`;
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
    return `<article class="card context-card"><div class="sentence">${highlightedSentence(item)}</div><div class="context-meta">${item.task_label?`<strong>${esc(item.task_label)}</strong>`:""}<span>קיצור: <strong>${esc(item.acronym)}</strong></span><span>ייחוס (עשוי להיות שגוי): <strong>${esc(item.gold??'לא זמין')}</strong></span>${item.prior_reused?'<span class="small" title="השיפוטים הקודמים נשמרו. אין צורך לבדוק שוב תשובה שכבר נשפטה; משלימים רק שיפוטים חסרים.">עבודה קודמת נשמרה</span>':''}</div></article><div class="answers">${item.answers.map(answer=>editableAnswer(answer,view)?renderMainAnswer(answer,labels[answer.id],record.judgments?.[answer.id]?.tags||[]):renderFilteredAnswer(answer)).join('')||'<p class="empty">אין כאן תשובות נוספות שדורשות שיפוט בתור הנוכחי.</p>'}</div><section class="optional compact-optional"><label><input id="suspect" type="checkbox" ${record.suspect?'checked':''}>חשד בייחוס / במשפט</label><label><input id="example" type="checkbox" ${record.example?'checked':''}>דוגמה למאמר</label><label for="note" class="sr-only">הערה קצרה (רשות)</label><textarea id="note" rows="1" placeholder="הערה (רשות)">${esc(record.note||'')}</textarea>${record.has_previous_note?'<span class="small" title="הערה קודמת נשמרה ותוצג לאחר החשיפה.">הערה קודמת שמורה</span>':''}</section>`;
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
  function continuedSession(state){return state?.manual_session?.continuation?.mode==='user_controlled_no_deadline';}
  function tableQueueIds(state,view){return state.table_queues?.[view]||queueIds(state,view);}
  function batchPairGroups(items){
    const groups=new Map();
    for(const item of items){
      // Server keys also verify the decoded answer and candidate mapping for selection.
      const key=item.display_pair_key||(item.task_label==='יצירה'?JSON.stringify([item.acronym,item.gold,item.answers[0].text]):item.id);
      if(!groups.has(key))groups.set(key,[]);
      groups.get(key).push(item);
    }
    return [...groups.values()];
  }
  function renderBatchTable(items,labels){
    let rowNumber=0;
    return '<table class="review-table"><thead><tr><th># / קיצור</th><th>המשפט המלא</th><th>ייחוס</th><th>תשובת המערכת המלאה</th><th>הכרעה מוצעת</th></tr></thead>'+batchPairGroups(items).map(group=>'<tbody>'+group.map((item,index)=>{
      rowNumber++;
      const shared=index===0?`<td rowspan="${group.length}" dir="auto">${esc(item.gold)}</td><td rowspan="${group.length}"><div dir="auto">${esc(item.answers[0].text)}</div>${group.length>1?`<p class="small">אותו צמד ב־${group.length} משפטים. סרוק כל הקשר לפני אישור.</p><div class="pair-actions">${Object.entries(LABELS).map(([value,label])=>`<button type="button" data-pair-label="${value}">${esc(label)} בכולם</button>`).join('')}</div><p class="small">אפשר לשנות חריג בכל שורה. הכפתורים משנים טיוטה בלבד.</p>`:''}</td>`:'';
      return `<tr><td>${rowNumber}<br><strong>${esc(item.acronym)}</strong><br><small>${esc(item.task_label)} · ${item.answers[0].occurrence_count} תשובות</small></td><td dir="auto">${esc(item.sentence)}</td>${shared}<td><select data-batch-item="${esc(item.id)}" aria-label="הכרעה בשורה ${rowNumber}">${Object.entries({...LABELS,'':'לא קראתי — דילוג'}).map(([value,label])=>`<option value="${value}" ${value===labels[item.id]?'selected':''}>${label}</option>`).join('')}</select></td></tr>`;
    }).join('')+'</tbody>').join('')+'</table>';
  }
  const api={continuedSession,tableQueueIds,batchPairGroups,renderBatchTable,shortcutIntent,nextActiveAnswer,canShowFull,isContinuation,queueIds,nextQueueId,editableAnswer,renderFilteredAnswer,renderPartition,judgmentPhaseText,TAGS,TAG_HELP,PHASES,tagsText,renderMaskedDetails,renderSummaryContent,fullCasesTable,LABELS,esc,labelOf,savedLabels,firstIncomplete,progressFor,highlightedSentence,renderMainAnswer,renderMain,renderCase};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  root.ShortReview=api;
  if(typeof document==='undefined')return;

  const $=id=>document.getElementById(id);
  let state,currentItem,dirty=false,timer=null,pending=Promise.resolve(),lastError=null,view='all',routeIds=[],currentPosition=-1,editVersion=0,lockCount=0,activeAnswerId=null;
  let tableMode=false,batchItems=[],batchId=null,batchDirty=false,skipped=new Set(),draftEdits=new Set(),deadlineHandled=false;
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
  function showScreen(name){for(const id of ['reviewScreen','detailsScreen','summaryScreen','startupError','reusePanel'])$(id).hidden=id!==name;}
  function updateProgress(){
    const counts=state.counts||{},total=counts.total_items??state.queue.length;
    $('progress').textContent=isContinuation(state)?`${counts.pending_answers||0} תשובות ממתינות`:`${counts.complete_items||0} מתוך ${total} משפטים הושלמו`;
    $('countDetail').textContent=isContinuation(state)?`${counts.pending_decisions||0} הכרעות נדרשות · ${counts.human_answers||0} תשובות עם שיפוט אנושי`:`נבדקו ${counts.reviewed_items??((counts.complete_items||0)+(counts.partial_items||0))} · ${counts.partial_items||0} חלקיים`;
    $('exposureNotice').hidden=!(state.revealed||state.exposed||state.prior_exposure);
    $('exposureNotice').textContent=isContinuation(state)?'החשיפות הקודמות נשמרו; הזהויות והניקוד מוסתרים כעת.':state.revealed||state.exposed?'התוצאות כבר נחשפו. שינויים חדשים יתועדו לאחר חשיפה; תמונת המצב שלפניה נשמרת.':'העבודה והחשיפות מהפרוטוקול הקודם נשמרו. אין לראות בשיפוטים הקודמים שיפוט חדש ללא חשיפה.';
    $('reviewer').textContent=`מתייג/ת: ${state.reviewer||'לא זמין'}`; if(continuedSession(state)){$('reviewer').textContent+=' · המשך באישורך, ללא מגבלת שעה';}else if(state.manual_session){const minutes=Math.max(0,Math.ceil((Date.parse(state.manual_session.deadline)-Date.now())/60000));$('reviewer').textContent+=` · נותרו עד ${minutes} דקות`;}
    if(state.protocol_version==='identified-test-review-v1'){$('progress').textContent=`${queueIds(state,view).length} שורות בתור שנבחר`;$('countDetail').textContent=`${counts.human_decisions||0} הכרעות חדשות (${counts.human_answers||0} תשובות) · ${counts.reused_groups||0} קבוצות משומשות (${counts.reused_answers||0} תשובות)`;for(const [value,label] of Object.entries({calibration:'כיול — תור מגוון',prioritized:'יצירה — כל השליליים',hebrew:'יצירה — ללא אותיות זרות (תור מקוצר)',positives:'ביקורת חיוביים',selection:'בחירה — חשדות ופענוח'})){if(!$('queueView').querySelector(`option[value="${value}"]`)){const option=document.createElement('option');option.value=value;option.textContent=label;$('queueView').appendChild(option);}}$('exposureNotice').textContent='הייחוס מוצג; שימוש חוזר מאושר נשמר בנפרד עם המקור והחשיפה ההיסטוריים.';$('filteredButton').textContent='עיון בכשלים הטכניים';}
    $('queueView').value=view;
    $('queueHint').textContent={all:'כל תשובה שטרם נשפטה ולא סוננה נמצאת כאן. ״לא בטוח״ נמצא בתור נפרד.',foreign:'אותיות משפה אחרת הן סימון מכני בלבד; אין כאן שיפוט או תגית ג׳יבריש אוטומטיים.',unsure:'שיפוטים קיימים שסומנו ״לא בטוח״. חזרה אליהם היא בחירה נפרדת.',suspicions:'מקרים עם חשד לייחוס או למשפט נשארים נגישים גם כאשר תשובה סוננה.',hebrew:'תור יצירה ללא אותיות זרות. האחרות נדחו בלבד ונשארו ללא שיפוט.',calibration:'כיול מגוון: כעשר דקות. אפשר לעבור לתור המקוצר אחרי הכיול.',prioritized:'כל השליליים האוטומטיים ביצירה, כולל תשובות עם אותיות זרות.',positives:'ביקורת חיוביים ותוספות: ברירת המחדל בטבלה אינה ראיה לשגיאה.',selection:'בדיקת פירוש ופענוח; אפשר לעבור לכרטיס בודד ולפתוח פרטים.',filtered:'עיון בלבד בכשלים הטכניים; הם נשארים במכנים.'}[view];
    $('queueView').title=$('queueHint').textContent;$('queueHint').hidden=false;
    if(currentItem){
      const p=progressFor(currentItem,state.records[currentItem.id]);
      $('itemProgress').textContent=view==='filtered'?'עיון במסוננות — אין צורך לשפוט כאן':`${p.judged}/${p.total} הכרעות בכרטיסים המוצגים נשמרו`;
      $('position').textContent=currentPosition>=0?`משפט ${currentPosition+1} מתוך ${routeIds.length} בתור`:'המשפט הנוכחי';
      $('previous').disabled=currentPosition<=0;
      $('saveNext').textContent=view==='filtered'?'הבא ←':'שמירה והבא ←';
    } else {$('itemProgress').textContent='';$('position').textContent='';}
  }
  function formPayload(){
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
    state=await request('/api/short/save',{revision:state.revision,...payload});
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
    const code=currentItem?.answers.find(answer=>answer.id===activeAnswerId)?.code;
    $('activeRowHint').textContent=code?`1–3: תשובה ${code}`:'';
    if(focus&&selected)selected.focus({preventScroll:true});
  }
  function renderItem(){
    $('itemView').innerHTML=renderMain(currentItem,state.records[currentItem.id],view);
    $('itemView').querySelectorAll('[data-row-answer]').forEach(row=>{row.addEventListener('click',()=>setActiveAnswer(row.dataset.rowAnswer));row.addEventListener('focusin',()=>setActiveAnswer(row.dataset.rowAnswer));});
    const ids=currentItem.answers.filter(answer=>editableAnswer(answer,view)).map(answer=>answer.id),labels=savedLabels(state.records[currentItem.id]);
    setActiveAnswer(ids.find(id=>!Object.hasOwn(LABELS,labels[id]))||ids[0],true);
    $('itemView').querySelectorAll('input,textarea').forEach(element=>element.addEventListener('input',markDirty));
    $('itemView').querySelectorAll('[data-restore-answer]').forEach(button=>button.onclick=()=>run(async()=>{const id=currentItem.id;await flush();state=await request('/api/short/restore',{revision:state.revision,item_id:id,answer_id:button.dataset.restoreAnswer});view='all';routeIds=queueIds(state,view).slice();await openItem(id);status('התשובה הוחזרה לבדיקה; לא נוצר שיפוט אנושי אוטומטי.');}));
    updateProgress();showScreen('reviewScreen');
  }
  async function openItem(id){
    await flush();
    if(!id){activeAnswerId=null;$('activeRowHint').textContent='';currentItem=null;currentPosition=-1;dirty=false;$('itemView').innerHTML='<p class="empty">אין תשובות בתור שנבחר. אפשר לבחור ״כל התשובות הממתינות״ או לפתוח סיכום.</p>';$('saveNext').disabled=true;$('previous').disabled=true;$('detailsButton').disabled=true;updateProgress();showScreen('reviewScreen');status('התור שנבחר ריק · כל העבודה נשמרה');return;}
    status('טוען פריט…');
    const result=await request('/api/short/open',{revision:state.revision,item_id:id,view});
    state=result.state;currentItem=result.item;currentPosition=routeIds.indexOf(id);dirty=false;lastError=null;$('saveNext').disabled=false;$('detailsButton').disabled=false;renderItem();
    status('העבודה השמורה נטענה · שינויים נשמרים אוטומטית בדיסק');
  }
  async function switchView(nextView){await saveBatchDraft();await flush();batchDirty=false;skipped.clear();view=nextView;routeIds=queueIds(state,view).slice();if(tableMode&&view!=='filtered')await openBatch();else{tableMode=false;setTableVisibility();await openItem(routeIds[0]);}}
  function setTableVisibility(){
    $('batchPanel').hidden=!tableMode;
    for(const element of [$('itemView'),document.querySelector('.navigation'),$('reviewScreen').querySelector(':scope > footer')])element.hidden=tableMode;
    $('tableToggle').textContent=tableMode?'מעבר לכרטיס בודד':'תצוגת טבלה';
    document.querySelector('.guidance').textContent=tableMode?'טבלה מרוכזת · עד 20 הקשרים בכל קבוצה · ברירות המחדל אינן תיוג עד לאישור שלך':'שפוט לפי ההקשר · 1 מתאימה · 2 לא מתאימה · 3 לא בטוח · Enter שמירה והבא';
  }
  async function openBatch(restored=null){
    setTableVisibility();currentItem=null;batchDirty=false;draftEdits=new Set(restored?.draft?.edited_item_ids||[]);
    const ids=restored?restored.items.map(i=>i.id):tableQueueIds(state,view).filter(id=>!skipped.has(id)).slice(0,20);
    if(!ids.length){batchItems=[];$('batchTable').innerHTML='<p>אין שורות נוספות בתור הזה. שורות שדילגת עליהן נשארות ללא תיוג; בחירה חוזרת בתור מציגה אותן.</p>';$('batchConfirm').disabled=true;$('batchSkip').disabled=true;updateProgress();showScreen('reviewScreen');return;}
    const result=restored||await request('/api/short/batch-open',{revision:state.revision,view,item_ids:ids});
    state=result.state;batchItems=result.items;batchId=result.batch_id;
    const labels=Object.fromEntries(batchItems.map(item=>[item.id,restored?.draft?restored.draft.labels[item.id]:(Object.values(state.records[item.id]?.judgments||{})[0]?.label||'not_fits')]));
    $('batchTable').innerHTML=renderBatchTable(batchItems,labels);
    function changed(selects){
      batchDirty=true;
      for(const select of selects){draftEdits.add(select.dataset.batchItem);select.closest('tr').classList.add('modified');}
      cacheDraft();status('שומר טיוטה בלבד — אין כאן אישור שיפוטים');run(saveBatchDraft,false);
    }
    $('batchTable').querySelectorAll('select').forEach(select=>select.onchange=()=>changed([select]));
    $('batchTable').querySelectorAll('[data-pair-label]').forEach(button=>button.onclick=()=>{
      const selects=[...button.closest('tbody').querySelectorAll('select[data-batch-item]')];
      for(const select of selects)select.value=button.dataset.pairLabel;
      changed(selects);
    });
    $('batchTable').inert=Boolean(restored?.read_only);$('batchConfirm').disabled=Boolean(restored?.read_only);$('batchSkip').disabled=Boolean(restored?.read_only);$('batchConfirm').textContent=`קראתי — אישור עד ${batchItems.length} שורות והמשך`;
    updateProgress();showScreen('reviewScreen');updateDraftChoices();status(restored?'טיוטה שוחזרה — אינה שיפוט מאושר'+(restored.read_only?' · הזמן הסתיים: עיון בלבד':''):`${batchItems.length} שורות מוצגות · ברירות המחדל טרם אושרו ואין להן שיפוט אנושי`);
  }
  $('tableToggle').onclick=()=>run(async()=>{await saveBatchDraft();await flush();batchDirty=false;tableMode=!tableMode;setTableVisibility();if(tableMode)await openBatch();else{routeIds=queueIds(state,view).slice();await openItem(routeIds[0]);}});
  $('batchConfirm').onclick=()=>run(async()=>{
    await saveBatchDraft();
    const labels=Object.fromEntries([...$('batchTable').querySelectorAll('select')].map(el=>[el.dataset.batchItem,el.value]));
    const result=await request('/api/short/batch-save',{revision:state.revision,view,batch_id:batchId,confirmed:true,labels});
    state=result.state;localStorage.removeItem(draftKey());draftEdits.clear();const count=Object.values(labels).filter(Boolean).length;for(const item of batchItems)skipped.add(item.id);batchDirty=false;
    await openBatch();status(`נשמרו בדיסק ${count} הכרעות שאישרת · הקבוצה הבאה עדיין לא מתויגת`);window.scrollTo({top:0});
  });
  $('batchSkip').onclick=()=>run(async()=>{await saveBatchDraft();for(const item of batchItems)skipped.add(item.id);batchDirty=false;await openBatch();window.scrollTo({top:0});});
  $('batchSummary').onclick=()=>run(async()=>{await saveBatchDraft();batchDirty=false;await openSummary();});
  function renderDetails(details){$('detailsView').innerHTML=renderMaskedDetails(details);}
  function renderSummary(summary,explicitFull=false){
    const full=canShowFull(state,summary,explicitFull);
    $('summaryTitle').textContent=full?'סיכום לאחר חשיפת התוצאות':'סיכום ביניים מוסתר';
    $('summaryIntro').textContent=full?'תמונת המצב שלפני החשיפה נשמרה. עריכות נוספות מסומנות לאחר חשיפה. זהו סיכום איכותני, ללא אומדן לכל המאגר.':'הסיכום מציג התקדמות, שיפוטים, תגיות והערות בלבד. אפשר לשמור ולצאת בלי לחשוף תוצאות.';
    $('summaryView').innerHTML=renderSummaryContent(summary,full);
    $('revealPanel').hidden=isContinuation(state)?explicitFull:Boolean(state.revealed||state.exposed);
    $('fullDownloads').hidden=!Boolean(state.revealed||state.exposed);
  }
  async function openSummary(){
    await saveBatchDraft();await flush();status('מכין סיכום…');
    const result=await request('/api/short/summary',{revision:state.revision});
    state=result.state;renderSummary(result.summary);updateProgress();showScreen('summaryScreen');lastError=null;
    status('הסיכום מוכן · הכרעות מאושרות וטיוטות נשמרות בנפרד');window.scrollTo({top:0});
  }
  async function download(path,name){await flush();const link=document.createElement('a');link.href=path;link.download=name;document.body.appendChild(link);link.click();link.remove();}
  function draftKey(){return `identified-unconfirmed-draft:${state.plan_id}`;}
  function draftPayload(){return {batch_id:batchId,view,edited_item_ids:[...draftEdits],labels:Object.fromEntries([...$('batchTable').querySelectorAll('select')].map(el=>[el.dataset.batchItem,el.value]))};}
  function cacheDraft(){localStorage.setItem(draftKey(),JSON.stringify(draftPayload()));}
  function updateDraftChoices(){
    $('draftChoice').innerHTML=(state.pending_drafts||[]).map(d=>`<option value="${esc(d.batch_id)}">${esc(d.saved_at)} · ${esc(d.view)}</option>`).join('');
    $('draftRestore').disabled=!(state.pending_drafts||[]).length;
    $('reuseOpen').textContent=`נבדקו בעבר (${state.pending_reuse||0} ממתינות לאישור)`;
  }
  async function saveBatchDraft(){
    if(!batchDirty||!batchId)return;
    const payload=draftPayload();cacheDraft();
    const result=await request('/api/short/batch-draft',payload);
    if(JSON.stringify(draftPayload())===JSON.stringify(payload)){batchDirty=false;localStorage.removeItem(draftKey());}
    state.pending_drafts=[...(state.pending_drafts||[]).filter(d=>d.batch_id!==payload.batch_id),{batch_id:payload.batch_id,view:payload.view,saved_at:result.draft.saved_at}];
    updateDraftChoices();status('טיוטה נשמרה בנפרד בדיסק — לא נוספו שיפוטים מאושרים');
  }
  $('draftRestore').onclick=()=>run(async()=>{
    const id=$('draftChoice').value;await saveBatchDraft();
    const restored=await request('/api/short/draft-restore',{revision:state.revision,batch_id:id});
    view=restored.draft.view;tableMode=true;await openBatch(restored);
  });
  async function openReuse(){
    await saveBatchDraft();await flush();
    const result=await request('/api/short/reuse-open',{revision:state.revision});state=result.state;
    $('reuseTable').innerHTML='<table class="review-table"><thead><tr><th>אישור</th><th>הקשר</th><th>ייחוס</th><th>תשובה והכרעה היסטורית</th><th>מקור ועיתוי</th></tr></thead><tbody>'+result.items.map(item=>`<tr><td><input type="checkbox" data-reuse-item="${esc(item.id)}" aria-label="שימוש חוזר ${esc(item.id)}" ${item.reused?'disabled checked':''}>${item.reused?'אושר לשימוש חוזר':''}</td><td>${esc(item.sentence)}</td><td>${esc(item.gold)}</td><td>${esc(item.answers[0].text)}<br><strong>נבדקו בעבר: ${esc(LABELS[item.historical_label])}</strong><br>${item.answers[0].occurrence_count} תשובות חדשות מקושרות</td><td>${item.historical_sources.map(p=>`${esc(p.annotator)} · <span dir="ltr" title="${esc(p.label_updated_at)}">${esc(new Date(p.label_updated_at).toLocaleString('he-IL',{timeZone:'Asia/Jerusalem'}))}</span><br><span dir="ltr">${esc(p.answer_id)}</span><br><span dir="ltr">${esc(p.origin)}</span><br>חשיפה: ${esc(PHASES[p.exposure_evidence?.status]||p.label_phase||'לא ידוע')}`).join('<hr>')}</td></tr>`).join('')+'</tbody></table>';
    updateProgress();updateDraftChoices();showScreen('reusePanel');status('ההכרעות ומקורותיהן מוצגים לאישור שימוש חוזר בלבד · שעון התיוג לא מופעל באישור זה');
  }
  $('reuseOpen').onclick=()=>run(openReuse);
  $('reuseSelectAll').onclick=()=>{$('reuseTable').querySelectorAll('input:not(:disabled)').forEach(el=>el.checked=true);};
  $('reuseConfirm').onclick=()=>run(async()=>{
    const ids=[...$('reuseTable').querySelectorAll('input:checked:not(:disabled)')].map(el=>el.dataset.reuseItem);
    if(!ids.length){status('בחר קבוצות לאישור שימוש חוזר');return;}
    const result=await request('/api/short/reuse-confirm',{revision:state.revision,confirmed:true,item_ids:ids});state=result.state;
    await openReuse();status(`אושר שימוש חוזר ב־${ids.length} קבוצות היסטוריות · לא נוצרו שיפוטים חדשים`);
  });
  $('reuseContinue').onclick=()=>run(async()=>{view='calibration';tableMode=true;await openBatch();});
  async function expireTable(){
    $('saveNext').disabled=true;$('itemView').inert=true;$('batchTable').inert=true;$('batchConfirm').disabled=true;$('batchSkip').disabled=true;
    try{await saveBatchDraft();status('הסתיימו 60 דקות: אין אישור נוסף. ההכרעות המאושרות גובו; טיוטות נשמרו בנפרד לשחזור ולעיון בלבד.');}
    catch(error){status('הזמן הסתיים. הטיוטה לא אושרה; שמירתה בדיסק לא אומתה. העותק המקומי בדפדפן נשמר — העתק הטיוטה בדפדפן יישלח שוב בעת רענון, בלי לאשר אותו.',true);}
  }
  setInterval(()=>{
    if(!state?.manual_session)return;
    if(continuedSession(state)){$('deadlineWarning').hidden=true;return;}
    const seconds=Math.max(0,Math.ceil((Date.parse(state.manual_session.deadline)-Date.now())/1000));
    const remaining=Math.ceil(seconds/60);$('reviewer').textContent=`מתייג/ת: ${state.reviewer} · נותרו עד ${remaining} דקות`;
    $('queueHint').hidden=false;$('queueHint').textContent=remaining>50?'שלב כיול (10 דקות): שיפוטים מגוונים ובדיקת שמירה וחידוש.':remaining>10?'שלב התור המתועדף: בחרו יצירה — ללא אותיות זרות.':'10 הדקות האחרונות: עברו לביקורת חיוביים וללא בטוח; אין צורך לסיים תור.';
    if(seconds<=120){$('deadlineWarning').hidden=false;$('deadlineWarning').textContent=seconds?`נותרו ${seconds} שניות. אשר רק שורות שכבר קראת. טיוטה שלא אושרה תישמר בנפרד; לא יהיה אישור אוטומטי או זמן נוסף.`:'החלון הסתיים. טיוטות אינן שיפוטים מאושרים; השחזור זמין לעיון בלבד.';}
    if(!seconds&&!deadlineHandled){deadlineHandled=true;run(expireTable,false);}
  },1000);
  async function init(){
    try{
      const [initial,session]=await Promise.all([request('/api/short/state'),request('/api/session')]);
      state=initial;$('qaBanner').hidden=!session.qa;if(state.protocol_version==='identified-test-review-v1'){
        $('tableToggle').hidden=false;$('reviewTools').hidden=false;tableMode=true;view=continuedSession(state)?'hebrew':'calibration';updateProgress();updateDraftChoices();
        const cached=localStorage.getItem(draftKey());
        if(cached){try{await request('/api/short/batch-draft',JSON.parse(cached));state=await request('/api/short/state');updateDraftChoices();}catch(error){status('טיוטה מקומית נשמרה בדפדפן אך לא ניתן לקשרה לקבוצה הפעילה; אין למחוק אותה.',true);}}
        if(state.manual_session&&!continuedSession(state)&&Date.now()>=Date.parse(state.manual_session.deadline))await openSummary();
        else if(state.pending_reuse&&!state.manual_session)await openReuse();
        else if(state.pending_drafts?.length){const restored=await request('/api/short/draft-restore',{revision:state.revision,batch_id:state.pending_drafts.at(-1).batch_id});view=restored.draft.view;await openBatch(restored);}
        else await openBatch();return;}view='all';routeIds=queueIds(state,view).slice();updateProgress();const next=firstIncomplete(state);if(isContinuation(state)||next||!state.queue.length)await openItem(next);else await openSummary();
    }catch(error){showError(error);if(!currentItem)showScreen('startupError');}
  }
  $('previous').onclick=()=>run(()=>openItem(routeIds[currentPosition-1]));
  $('queueView').onchange=()=>{const selected=$('queueView').value;run(()=>switchView(selected));};
  $('filteredButton').onclick=()=>run(()=>switchView('filtered'));
  $('saveNext').onclick=()=>run(async()=>{const before=routeIds.slice(),id=currentItem.id;await flush();const remaining=queueIds(state,view);const next=isContinuation(state)?nextQueueId(before,id,remaining):before[before.indexOf(id)+1];routeIds=remaining.slice();if(next)await openItem(next);else await openSummary();});
  $('stopSummary').onclick=()=>run(openSummary);
  $('detailsButton').onclick=()=>run(async()=>{await flush();const result=await request('/api/short/details',{revision:state.revision,item_id:currentItem.id});state=result.state;renderDetails(result.details);showScreen('detailsScreen');status('פתיחת הפרטים תועדה · העבודה נשמרה');window.scrollTo({top:0});});
  $('backDetails').onclick=()=>{showScreen('reviewScreen');window.scrollTo({top:0});};
  $('backSummary').onclick=()=>run(async()=>{if(tableMode){await openBatch();}else if(!currentItem){routeIds=queueIds(state,view).slice();await openItem(routeIds[0]);}else showScreen('reviewScreen');window.scrollTo({top:0});});
  $('exportAnnotations').onclick=()=>run(()=>download('/api/short/export.json','short-review-masked-annotations.json'));
  $('exportSummary').onclick=()=>run(()=>download('/api/short/summary/export.md','short-review-masked-summary.md'));
  $('revealResults').onclick=()=>run(async()=>{await flush();status('שומר תמונת מצב ופותח תוצאות…');const result=await request('/api/short/reveal',{revision:state.revision});state=result.state;renderSummary(result.summary,true);updateProgress();showScreen('summaryScreen');lastError=null;status('תמונת המצב נשמרה · התוצאות נחשפו');window.scrollTo({top:0});});
  $('exportFull').onclick=()=>run(()=>download('/api/short/full-export.json','short-review-full-results.json'));
  $('exportFullSummary').onclick=()=>run(()=>download('/api/short/full-summary.md','short-review-full-summary.md'));
  $('retryLoad').onclick=()=>run(init);
  window.addEventListener('keydown',event=>{
    if(tableMode||$('reviewScreen').hidden||document.querySelector('main').inert)return;
    const intent=shortcutIntent(event,activeAnswerId);if(!intent)return;
    if(intent.type==='next'){if($('saveNext').disabled)return;event.preventDefault();$('saveNext').onclick();return;}
    const rows=[...$('itemView').querySelectorAll('[data-row-answer]')],row=rows.find(row=>row.dataset.rowAnswer===intent.answerId);
    const radio=row?.querySelector(`input[type=radio][value="${intent.label}"]`);if(!radio)return;
    event.preventDefault();radio.checked=true;markDirty();
    const labels=Object.fromEntries(rows.map(r=>[r.dataset.rowAnswer,r.querySelector('input[type=radio]:checked')?.value||'']));
    setActiveAnswer(nextActiveAnswer(rows.map(r=>r.dataset.rowAnswer),labels,intent.answerId),true);
  });
  window.addEventListener('beforeunload',event=>{if(dirty||batchDirty||lastError){event.preventDefault();event.returnValue='';}});
  run(init);
})(typeof globalThis!=='undefined'?globalThis:this);
