/*
 * DeepSeekV4 参数量与动态激活显存 · 回归测试
 * 用法：node tests/verify_dsv4.js（在项目根目录运行）
 * 校验 deepseek4.html：default / v4pro 预设的参数量、激活参数量
 * 与动态激活显存分解（参照 DeepSeek4_参数量及MFU计算.xlsx / dsk4_显存分析.xlsx 口径推导）。
 * 默认值来自 dsv4_presets.json，校准自 DeepSeek4_参数量及MFU计算.xlsx 的 v4pro 页：
 * Lm=96、TopK=16、fe=4096、nr=1024、ns=1、Li=30、Lc4=30、Lc128=31，
 * 总参数量 8698.184466432 B、激活参数量 173.59568896 B（与 Excel 一致）。
 */
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'deepseek4.html'), 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) throw new Error('no <script> block found');
let script = m[1];

new Function(script);
console.log('[OK] JS syntax');

/* 所有 $("...") 引用的 id 必须在 HTML 中存在 */
const idsUsed = [...script.matchAll(/\$\("([^"]+)"\)/g)].map(x => x[1]);
const missing = idsUsed.filter(id => !html.includes('id="' + id + '"'));
if (missing.length) { console.error('[FAIL] missing HTML ids:', missing); process.exit(1); }
console.log('[OK] all', idsUsed.length, 'JS-referenced ids exist in HTML');

function makeEl(id) {
  return { id, value: '', checked: false, textContent: '', innerHTML: '', className: '',
           options: [], selectedIndex: 0, addEventListener() {}, style: {} };
}
const els = {};
class CList {
  constructor(){ this.s = new Set(); }
  add(c){ this.s.add(c); } remove(c){ this.s.delete(c); } contains(c){ return this.s.has(c); }
  toggle(c, force){ if (force) this.s.add(c); else this.s.delete(c); }
}
global.document = {
  getElementById(id) {
    if (!els[id]) { const e = makeEl(id); e.classList = new CList(); els[id] = e; }
    return els[id];
  },
  createElement(tag){ return { tag, href: '', download: '', click(){} }; },
  body: { appendChild(){}, removeChild(){} }
};

script = script.replace(/\}\)\(\);\s*$/, 'globalThis.__t = { compute, applyPreset, setDefaults, getSync: function () { return LAST; } };\n})();');
eval(script);

const readNum = id => els[id].textContent;
let fail = 0;
function assertEq(name, actual, expected) {
  if (actual !== expected) { console.error('[FAIL]', name, '=>', JSON.stringify(actual), 'expected', JSON.stringify(expected)); fail = 1; }
  else console.log('[OK]', name, '=', actual);
}

/* Pass A: default 口径 = Excel v4pro（Lm=96、k=16、fe=4096、nr=1024、ns=1） */
__t.setDefaults();
assertEq('总参数量', readNum('kpi-params'), '8,698.184 B（8.7 T）');
assertEq('激活参数量', readNum('kpi-act'), '173.596 B');
assertEq('单层动态', readNum('kpi-per'), '7.3672 GiB');
assertEq('总动态', readNum('kpi-dyn'), '707.25 GiB');
assertEq('params-total', readNum('params-total'), '8,698.184 B（8.7 T）');
assertEq('act-total', readNum('act-total'), '173.596 B');
assertEq('dyn-attn-total', readNum('dyn-attn-total'), '2.4063 GiB');
assertEq('dyn-moe-total', readNum('dyn-moe-total'), '4.9609 GiB');
assertEq('params-list 行数', (els['params-list'].innerHTML.match(/<li>/g)||[]).length + '', '7');
assertEq('act-list 行数', (els['act-list'].innerHTML.match(/<li>/g)||[]).length + '', '5');
assertEq('dyn-attn 行数', (els['dyn-attn-list'].innerHTML.match(/<li>/g)||[]).length + '', '6');
assertEq('dyn-moe 行数', (els['dyn-moe-list'].innerHTML.match(/<li>/g)||[]).length + '', '8');
assertEq('dyn-extra 行数', (els['dyn-extra-list'].innerHTML.match(/<li>/g)||[]).length + '', '5');
console.log('--- Pass A OK ---\n');

/* Pass B: v4pro 预设 = Excel v4pro（Lm=96、k=16、Lc4/Lc128=30/31、fe=4096、nr=1024、ns=1） */
__t.applyPreset('v4pro');
assertEq('v4pro Lm', els['Lm'].value, 96);
assertEq('v4pro k', els['k'].value, 16);
assertEq('v4pro Lc4', els['Lc4'].value, 30);
assertEq('v4pro 总参数量', readNum('kpi-params'), '8,698.184 B（8.7 T）');
assertEq('v4pro 激活参数量', readNum('kpi-act'), '173.596 B');
assertEq('v4pro 单层动态', readNum('kpi-per'), '7.3672 GiB');
assertEq('v4pro 总动态', readNum('kpi-dyn'), '707.25 GiB');
console.log('--- Pass B OK ---\n');

/* Pass C: TopK 对激活参数量与动态显存的影响（default 口径，k 默认 16） */
__t.setDefaults();
els['k'].value = '6';
__t.compute();
assertEq('k6 激活参数量', readNum('kpi-act'), '89.039 B');
assertEq('k6 单层动态', readNum('kpi-per'), '4.75 GiB');
assertEq('k6 总动态', readNum('kpi-dyn'), '456 GiB');
els['k'].value = '16';
__t.compute();
assertEq('恢复 k16 单层动态', readNum('kpi-per'), '7.3672 GiB');
console.log('--- Pass C OK ---\n');

/* Pass D: 同步到仿真器数据（单层 DSA Attn 参数量 / 单专家 / 单层动态激活 GB） */
__t.setDefaults();
var g = __t.getSync();
assertEq('sync attn（= DSA Attn 单层含 mhc）', Math.abs(g.attn - 0.301268992) < 1e-9 ? true : 'attn=' + g.attn, true);
assertEq('sync exp（= 单个 MoE 专家 fe=4096）', Math.abs(g.exp - 0.088080384) < 1e-9 ? true : 'exp=' + g.exp, true);
assertEq('sync act（= 单层动态激活 GiB）', Math.abs(g.act - 7.3671875) < 1e-3 ? true : 'act=' + g.act, true);
assertEq('sync layers（= 96）', g.layers, 96);
assertEq('sync nExp（= 1024）', g.nExp, 1024);
assertEq('sync topk（= 16）', g.topk, 16);
assertEq('sync moeHidden（= 4096）', g.moeHidden, 4096);
assertEq('sync hidden（= 7168）', g.hidden, 7168);
assertEq('sync seqLen（= 4096）', g.seqLen, 4096);
console.log('--- Pass D OK ---\n');

/* Pass E: 外部结构预设文件 dsv4_presets.json 与页面内置 fallback 一致性 */
const dp = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'dsv4_presets.json'), 'utf8'));
assertEq('dsv4 预设数', (dp.presets || []).length + '', '1');
assertEq('dsv4 default.Lm', dp.default.Lm, 96);
assertEq('dsv4 default.k', dp.default.k, 16);
assertEq('dsv4 default.fe', dp.default.fe, 4096);
assertEq('dsv4 default.Lmtp', dp.default.Lmtp, 0);
assertEq('dsv4 default.nr', dp.default.nr, 1024);
assertEq('dsv4 default.Lc4', dp.default.Lc4, 30);
assertEq('dsv4 预设0 id', dp.presets[0].id, 'v4pro');
assertEq('dsv4 预设0 Lm', dp.presets[0].values.Lm, 96);
assertEq('dsv4 预设0 k', dp.presets[0].values.k, 16);
assertEq('dsv4 预设0 Lc4', dp.presets[0].values.Lc4, 30);
assertEq('dsv4 预设0 Lc128', dp.presets[0].values.Lc128, 31);
assertEq('dsv4 预设0 fe', dp.presets[0].values.fe, 4096);
assertEq('dsv4 预设0 nr', dp.presets[0].values.nr, 1024);
__t.applyPreset('v4pro');
assertEq('外部预设 v4pro Lm 生效', els['Lm'].value, 96);
assertEq('外部预设 v4pro k 生效', els['k'].value, 16);
console.log('--- Pass E OK ---\n');

(async function () {
  /* Pass F: 浏览器路径 —— fetch 加载外部 json（预设键名 values）后点击预设必须生效 */
  const h2 = fs.readFileSync(path.join(__dirname, '..', 'deepseek4.html'), 'utf8');
  let s2 = h2.match(/<script>([\s\S]*?)<\/script>/)[1];
  const els2 = {};
  const lsn = {};
  const g2 = id => {
    if (!els2[id]) {
      const e = { id, value: '', checked: false, textContent: '', innerHTML: '', className: '',
                 classList: { add(){}, remove(){}, toggle(){}, contains(){ return false; } } };
      e.addEventListener = (ev, fn) => { (lsn[id] = lsn[id] || {})[ev] = fn; };
      els2[id] = e;
    }
    return els2[id];
  };
  global.document = { getElementById: g2, createElement(){ return { appendChild(){}, textContent: '', value: '' }; }, body: {} };
  global.fetch = async () => ({ ok: true, json: async () => JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'dsv4_presets.json'), 'utf8')) });
  s2 = s2.replace(/\}\)\(\);\s*$/, 'globalThis.__t2 = { compute, applyPreset };\n})();');
  eval(s2);
  await new Promise(r => setTimeout(r, 80));
  assertEq('Fetch 后默认 Lm', els2['Lm'].value, 96);
  /* 用户手动改 Lm=77 后点击预设，应恢复为 Excel 校准值 96 */
  els2['Lm'].value = '77';
  els2['preset'].value = 'v4pro';
  const fn = lsn['preset'] && lsn['preset']['change'];
  assertEq('Fetch 后 preset change 监听存在', !!fn, true);
  fn.call(els2['preset'], {});
  await new Promise(r => setTimeout(r, 20));
  assertEq('Fetch 后点击预设 Lm=96', els2['Lm'].value, 96);
  assertEq('Fetch 后点击预设 k=16', els2['k'].value, 16);
  console.log('--- Pass F OK ---\n');

  if (fail) { console.error('TESTS FAILED'); process.exit(1); }
  console.log('ALL TESTS PASSED');
})();