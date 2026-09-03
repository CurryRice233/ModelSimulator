/*
 * “同步到仿真器”桥接链路测试
 * 用法：node tests/verify_sync.js（在项目根目录运行）
 * 覆盖：
 *  ① deepseek4.html 点击「⇢ 同步到仿真器」→ 写入 localStorage + 自动跳转 index.html
 *  ② index.html 一次性消费完整 payload（9 个字段应用、提示文案）
 *  ③ 外部预设 fetch 完成后不覆盖已同步的输入（SYNC_APPLIED 守卫）
 */
const fs = require('fs');
const path = require('path');

function buildEnv(preloadedSync, htmlName) {
  const html = fs.readFileSync(path.join(__dirname, '..', htmlName), 'utf8');
  const m = html.match(/<script>([\s\S]*?)<\/script>/);
  let script = m[1];
  const els = {};
  const listeners = {};
  // 元素桩：innerHTML 重建 + 事件捕获
  const g = id => {
    if (!els[id]) {
      const e = { id, value: '', checked: false, textContent: '', innerHTML: '', className: '',
                 classList: { s: new Set(), add(c){ this.s.add(c); }, remove(c){ this.s.delete(c); },
                              toggle(c,f){ f ? this.s.add(c) : this.s.delete(c); }, contains(c){ return this.s.has(c); } } };
      e.appendChild = function (o) {
        if (o) { this.innerHTML += '<option value="' + (o.value||'') + '">' + (o.textContent||'') + '</option>'; }
      };
      e.addEventListener = (ev, fn) => { (listeners[id] = listeners[id] || {})[ev] = fn; };
      els[id] = e;
    }
    return els[id];
  };
  global.document = { getElementById: g, createElement(){ return { value: '', textContent: '' }; }, body: {} };
  global.localStorage = (function () {
    const m = {};
    if (preloadedSync) m['dsv4_sync'] = JSON.stringify(preloadedSync);
    return { getItem(k){ return k in m ? m[k] : null; }, setItem(k,v){ m[k]=String(v); }, removeItem(k){ delete m[k]; }, _m: m };
  })();
  global.fetch = async () => {
    const file = htmlName === 'deepseek4.html' ? 'dsv4_presets.json' : 'presets.json';
    return { ok: true, json: async () => JSON.parse(fs.readFileSync(path.join(__dirname, '..', file), 'utf8')) };
  };
  global.window = { location: { href: '' } };
  script = script.replace(/\}\)\(\);\s*$/, 'globalThis.__env = { listeners, els };\n})();');
  eval(script);
  return { els, listeners, localStorage: global.localStorage };
}

let fail = 0;
function assertEq(name, actual, expected) {
  if (actual !== expected) { console.error('[FAIL]', name, '=>', JSON.stringify(actual), 'expected', JSON.stringify(expected)); fail = 1; }
  else console.log('[OK]', name, '=', actual);
}
function assertContains(name, actual, needle) {
  if (!String(actual).includes(needle)) { console.error('[FAIL]', name, 'does not contain', needle, '=>', actual); fail = 1; }
  else console.log('[OK]', name, 'contains', JSON.stringify(needle));
}

(async function () {
  /* Pass S1: deepseek4.html 点击同步 → localStorage 写入 + 自动跳转 */
  let env = buildEnv(null, 'deepseek4.html');
  await new Promise(r => setTimeout(r, 80));  // 等外部预设 fetch 完成、默认渲染
  const syncBtn = env.listeners['btn-sync'] && env.listeners['btn-sync']['click'];
  assertEq('btn-sync 监听存在', !!syncBtn, true);
  syncBtn.call(env.els['btn-sync'], {});
  const payload = JSON.parse(env.localStorage._m['dsv4_sync']);
  assertEq('已写入 localStorage', !!payload, true);
  assertEq('跳转 index.html', global.window.location.href, './index.html');
  assertEq('payload.attn', payload.attn, 0.301268992);
  assertEq('payload.exp', payload.exp, 0.088080384);
  assertEq('payload.layers', payload.layers, 96);
  assertEq('payload.nExp', payload.nExp, 1024);
  assertEq('payload.act', payload.act, 7.3671875);
  console.log('--- Pass S1 OK ---\n');

  /* Pass S2: index.html 一次性消费完整 payload，且 fetch 完成后不被覆盖 */
  delete global.window;
  env = buildEnv({
    attn: 0.301268992, exp: 0.088080384, act: 7.3671875,
    layers: 96, nExp: 1024, topk: 16, moeHidden: 4096, hidden: 7168, seqLen: 4096
  }, 'index.html');
  assertEq('attn 应用', env.els['attn'].value, 0.301268992);
  assertEq('exp 应用', env.els['exp'].value, 0.088080384);
  assertEq('act 应用', env.els['act'].value, 7.3671875);
  assertEq('layers 应用', env.els['layers'].value, 96);
  assertEq('nExp 应用', env.els['nExp'].value, 1024);
  assertEq('topk 应用', env.els['topk'].value, 16);
  assertEq('moeHidden 应用', env.els['moeHidden'].value, 4096);
  assertEq('seqLen 应用', env.els['seqLen'].value, 4096);
  assertContains('提示文案', env.els['cfg-msg'].textContent, '已从 DeepSeekV4 页同步');
  assertContains('提示含 96 层', env.els['cfg-msg'].textContent, '96 层');
  assertEq('同步键已清除', env.localStorage.getItem('dsv4_sync'), null);
  /* await：微任务（fetch presets.json then）之后 */
  await new Promise(r => setTimeout(r, 80));
  assertEq('fetch 后 layers 仍为 96（不被 default 61 覆盖）', env.els['layers'].value, 96);
  assertContains('fetch 后同步提示仍保留', env.els['cfg-msg'].textContent, '已从 DeepSeekV4 页同步');
  console.log('--- Pass S2 OK ---\n');

  if (fail) { console.error('TESTS FAILED'); process.exit(1); }
  console.log('ALL TESTS PASSED');
})();