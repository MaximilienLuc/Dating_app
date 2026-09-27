// Recalcul des scores dans le navigateur (identique à Python, vérifié par verify_score.js)
function xgbProba(x, M) {
  let m = M.base_margin;
  for (const t of M.trees) {
    let i = 0;
    while (t[i][0] >= 0) {
      const n = t[i], v = x[n[0]];
      i = (v === null || v === undefined || Number.isNaN(v)) ? n[4]
        : (Math.fround(v) < Math.fround(n[1]) ? n[2] : n[3]);
    }
    m += t[i][5];
  }
  return 1 / (1 + Math.exp(-m));
}
function logitProba(x, L) {
  let z = L.intercept;
  for (let j = 0; j < L.idx.length; j++) {
    let v = x[L.idx[j]];
    if (v === null || v === undefined) v = L.median[j];
    z += L.coef[j] * (v - L.mean[j]) / L.scale[j];
  }
  return 1 / (1 + Math.exp(-z));
}
if (typeof module !== "undefined") module.exports = { xgbProba, logitProba };
