// Hero: scattered "human" nodes drift; a hidden campaign periodically synchronises and lights up.
(() => {
  const c = document.getElementById("net");
  const ctx = c.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let W, H, dpr, nodes = [], t = 0;
  const INK = "22,21,15", ACC = "232,85,45";

  function init() {
    dpr = Math.min(devicePixelRatio || 1, 2);
    W = c.clientWidth; H = c.clientHeight;
    c.width = W * dpr; c.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.round(Math.min(110, W * H / 12000));
    const cx = W * (W > 900 ? 0.74 : 0.6), cy = H * 0.52, r = Math.min(W, H) * 0.16;
    nodes = Array.from({ length: n }, (_, i) => {
      const bot = i < 16;
      const a = (i / 16) * Math.PI * 2;
      return {
        bot, x: Math.random() * W, y: Math.random() * H,
        hx: bot ? cx + Math.cos(a) * r * (0.5 + Math.random() * 0.6) : null,
        hy: bot ? cy + Math.sin(a) * r * (0.5 + Math.random() * 0.6) : null,
        vx: (Math.random() - .5) * .3, vy: (Math.random() - .5) * .3, s: bot ? 3.2 : 2 + Math.random() * 1.6,
      };
    });
  }

  function frame() {
    t += 1;
    const phase = (Math.sin(t / 220) + 1) / 2;            // 0 = scattered, 1 = coordinated
    const k = Math.max(0, (phase - 0.35) / 0.65);
    ctx.clearRect(0, 0, W, H);
    for (const p of nodes) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
      if (p.bot) { p.x += (p.hx - p.x) * 0.02 * k; p.y += (p.hy - p.y) * 0.02 * k; }
    }
    for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      const both = a.bot && b.bot;
      const lim = both ? 260 : 110;
      if (d < lim) {
        const al = (1 - d / lim) * (both ? 0.55 * k : 0.12);
        if (al < 0.01) continue;
        ctx.strokeStyle = `rgba(${both ? ACC : INK},${al})`;
        ctx.lineWidth = both ? 1.1 : 0.7;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }
    for (const p of nodes) {
      ctx.fillStyle = p.bot ? `rgba(${ACC},${0.35 + 0.65 * k})` : `rgba(${INK},0.35)`;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.s, 0, Math.PI * 2); ctx.fill();
    }
    if (!reduce) requestAnimationFrame(frame);
  }
  addEventListener("resize", init);
  init(); frame();

  // nav border + reveal on scroll
  const nav = document.querySelector(".nav");
  addEventListener("scroll", () => nav.classList.toggle("scrolled", scrollY > 20), { passive: true });
  const io = new IntersectionObserver(es => es.forEach(e => e.isIntersecting && (e.target.classList.add("in"), io.unobserve(e.target))), { threshold: .12 });
  document.querySelectorAll(".section > *").forEach(el => { el.classList.add("reveal"); io.observe(el); });
})();
