// Render every extracted formula with the same pinned MathJax used by the site.
const fs = require('node:fs');
const formulas = JSON.parse(fs.readFileSync(process.argv[2] || 'site/math-formulas.json', 'utf8'));
if (!formulas.length) throw new Error('No formulas found in this mathematics site');
// GitHub's live renderer, inspected 2026-09-19:
// https://github.githubassets.com/assets/chunk-lazy-element-math-renderer-def98b14253cbdbc.js
// Its filter uses substring matching, before MathJax parses the expression.
const githubBlockedMacros = [
  'DeclareMathOperator', 'DeclarePairedDelimiters', 'renewtagform', 'newtagform',
  'colorbox', 'fcolorbox', 'hphantom', 'vphantom', 'phantom', 'operatorname',
  'Newextarrow', 'definecolor', 'mathchoice', 'unicode', 'mmlToken'
];
require('mathjax').init({loader: {load: ['input/tex', 'output/svg']}}).then(async MathJax => {
  const failures = [];
  const githubPageCost = new Map();
  for (const formula of formulas) {
    // Wiki export adds empty groups when protecting inline vertical bars.
    const wikiTex = formula.display ? formula.tex : formula.tex.replace(/\\\|/g, '\\Vert{}').replace(/(?<!\\)\|/g, '\\vert{}');
    const cost = wikiTex.split('{').length;
    const pageCost = (githubPageCost.get(formula.source) || 0) + cost;
    githubPageCost.set(formula.source, pageCost);
    if (cost > 1000 || pageCost > 2000) {
      failures.push({source: formula.source, errors: ['GitHub formula/page complexity budget exceeded'], cost, pageCost});
    }
    const blocked = githubBlockedMacros.filter(name => ['\\' + name, '\\$' + name, '\\${' + name].some(pattern => wikiTex.includes(pattern)));
    if (blocked.length) {
      failures.push({source: formula.source, tex: formula.tex, errors: blocked.map(name => 'GitHub does not allow ' + name)});
      continue;
    }
    const node = await MathJax.tex2svgPromise(formula.tex, {display: formula.display});
    const output = MathJax.startup.adaptor.outerHTML(node);
    if (output.includes('data-mjx-error')) {
      failures.push({source: formula.source, tex: formula.tex, errors: [...output.matchAll(/data-mjx-error="([^"]*)"/g)].map(match => match[1])});
    }
  }
  console.log(`Rendered ${formulas.length} formulas; ${failures.length} errors`);
  if (failures.length) {
    console.error(JSON.stringify(failures, null, 2));
    process.exitCode = 1;
  }
}).catch(error => { console.error(error.message); process.exitCode = 1; });
