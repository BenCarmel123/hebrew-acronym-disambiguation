'use strict';
// Pure helpers shared by the browser and the focused offline tests.
(function (root) {
  const qualities = ['correct', 'wrong', 'partial', 'undecidable', 'no_answer'];
  function annotation(record = {}) { return record.draft || record.reviewed || {}; }
  function answersFor(item, systemId = '') {
    return item.answers.filter(answer => !systemId || answer.system_id === systemId);
  }
  function judgmentsFor(item, record, systemId = '') {
    const saved = annotation(record).answers || {};
    return answersFor(item, systemId).map(answer => saved[answer.id]).filter(Boolean);
  }
  function completion(item, record = {}) {
    const a = annotation(record);
    const judged = judgmentsFor(item, record).filter(j => qualities.includes(j.quality)).length;
    const decided = Boolean((a.item_problems || []).length);
    const eligible = decided && judged === item.answers.length;
    const status = record.completion?.status === 'complete' && eligible ? 'complete' :
      (judged || decided || judgmentsFor(item, record).some(j => j.quality)) ? 'partial' : 'draft';
    return {status, judged_answers: judged, total_answers: item.answers.length, item_decided: decided};
  }
  function matches(item, record = {}, filters = {}) {
    const a = annotation(record), s = filters.status || '', f = filters.finding || '';
    if (filters.source && String(item.source) !== filters.source) return false;
    if (filters.acronym && item.acronym !== filters.acronym) return false;
    if (filters.system && !answersFor(item, filters.system).length) return false;
    const c = completion(item, record), judgments = judgmentsFor(item, record, filters.system);
    if (s === 'new' && (record.draft || record.reviewed)) return false;
    if (['draft', 'partial', 'complete'].includes(s) && (!record.draft && !record.reviewed || c.status !== s)) return false;
    if (['revisit', 'example'].includes(s) && !a[s]) return false;
    if (f === 'mechanical') return filters.system ? answersFor(item, filters.system).some(x => (x.mechanical_flags || []).length) : Boolean((item.findings || []).length);
    if (f === 'disagreement') return judgments.some(x => x.disagrees_auto === 'yes');
    if ([...qualities, 'legacy_partial'].includes(f)) return judgments.some(x => x.quality === f);
    return !f || (a.item_problems || []).includes(f);
  }
  function summary(items, records, systemId = '') {
    const result = {items: items.length, complete: 0, partial: 0, draft: 0, judged: 0, possible: 0, qualities: {}};
    for (const item of items) {
      const record = records[item.id] || {};
      if (record.draft || record.reviewed) result[completion(item, record).status]++;
      result.possible += answersFor(item, systemId).length;
      for (const j of judgmentsFor(item, record, systemId)) {
        if (!qualities.includes(j.quality)) continue;
        result.judged++;
        result.qualities[j.quality] = (result.qualities[j.quality] || 0) + 1;
      }
    }
    return result;
  }
  const api = {qualities, annotation, answersFor, judgmentsFor, completion, matches, summary};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ReviewLogic = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
