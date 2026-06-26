import { getApiBase } from './api.js';
import { teacherLogoutApi } from './auth.js';

let teacherView        = "dash";
let teacherPollInterval = null;
let teacherTimerInt    = null;
let localStudents      = [];
let localAuditLog      = [];
let localSnapshots     = [];
let localEventsCache   = [];

function pad(n) { return String(n).padStart(2, '0'); }

function toast(msg, type) {
  const t = document.getElementById("toast");
  if (!t) return;
  t.textContent = msg;
  t.className = "toast show" + (type ? " " + type : "");
  clearTimeout(t._t);
  t._t = setTimeout(() => t.className = "", 3500);
}

function showErr(id, msg) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = msg || "";
  el.style.display = msg ? "block" : "none";
}

function hideErr(id) { showErr(id, ""); }

export function padTime(n) { return pad(n); }

export function showPage(id) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  const page = document.getElementById(id);
  if (page) page.classList.add("active");
}

export function spinner(on) {
  const s = document.getElementById("spinner");
  if (s) s.classList.toggle("show", on);
}

/* ── Live Feed ────────────────────────────────────────── */

function startLiveFeed() {
  const img = document.getElementById("liveFeedImg");
  if (!img) return;

  function tryLoad() {
    img.src = getApiBase() + "/proctor/video_feed?t=" + Date.now();
  }
  tryLoad();

  setInterval(() => {
    if (parseFloat(img.style.opacity) < 0.5) tryLoad();
  }, 8000);

  async function pollStatus() {
    try {
      const r = await fetch(getApiBase() + "/proctor/status");
      const d = await r.json();
      const lbl = document.getElementById("liveFeedLabel");
      const st  = document.getElementById("liveFeedStatus");
      const badge = document.getElementById("liveFeedBadge");
      if (!lbl || !st || !badge) return;
      if (d.student_name && d.student_name !== "Unknown") {
        lbl.textContent = d.student_name;
        lbl.style.color = "#7ec8e3";
        st.textContent  = d.status_text || "Live";
        badge.textContent = "Live";
        badge.style.cssText = "padding:2px 8px;border-radius:12px;background:rgba(53,237,126,0.15);color:#35ed7e";
      } else {
        lbl.textContent = "No candidate";
        lbl.style.color = "#aaa";
        st.textContent  = "Waiting\u2026";
        badge.textContent = "Idle";
        badge.style.cssText = "padding:2px 8px;border-radius:12px;background:rgba(255,255,255,0.08);color:#888";
      }
    } catch {
      const badge = document.getElementById("liveFeedBadge");
      if (badge) { badge.textContent = "Offline"; badge.style.color = "#666"; }
    }
  }
  pollStatus();
  setInterval(pollStatus, 3000);
}

/* ── Data Fetching ────────────────────────────────────── */

async function fetchStudents() {
  try {
    const [sr, ss] = await Promise.all([
      fetch(getApiBase() + "/api/teacher/students",  { credentials: "include" }),
      fetch(getApiBase() + "/api/proctor/summary",   { credentials: "include" }),
    ]);
    if (sr.ok) {
      const sd = await sr.json();
      localStudents = sd.students.map(s => ({
        ...s,
        score:  s.risk_score,
        events: s.event_count,
        status: s.last_event,
      }));
      const tc = document.getElementById("totalCandidates");
      if (tc) tc.textContent = sd.total;
    }
    if (ss.ok) {
      const sm = await ss.json();
      const sa = document.getElementById("s-active");
      const sf = document.getElementById("s-flagged");
      const sw = document.getElementById("s-warnings");
      const se = document.getElementById("s-events");
      if (sa) sa.textContent = sm.logged_in;
      if (sf) sf.textContent = sm.flagged;
      if (sw) sw.textContent = sm.warnings;
      if (se) se.textContent = sm.total_events;
    }
  } catch {}

  renderStudentCards();
  renderCharts();
}

async function fetchAuditEvents() {
  try {
    const res = await fetch(getApiBase() + "/api/proctor/events", { credentials: "include" });
    const d   = await res.json();
    if (res.ok) {
      localAuditLog = d.events.map(e => ({
        time:    e.time,
        student: e.student_name,
        type:    e.event_type,
        level:   e.level,
        conf:    e.confidence,
        impact:  e.impact,
        note:    e.note || "",
        id:      e.id,
      }));
    }
  } catch {}

  if (teacherView === "audit") renderAuditTable();
  renderLatestFlags();
}

async function fetchSnapshots() {
  try {
    const res = await fetch(getApiBase() + "/api/proctor/snapshots", { credentials: "include" });
    if (res.ok) {
      const d = await res.json();
      localSnapshots = d.snapshots || [];
      const ss = document.getElementById("s-snaps");
      if (ss) ss.textContent = localSnapshots.length;
      if (teacherView === "snapshots") renderSnapshotView();
    }
  } catch {}
}

export async function refreshAll() {
  await Promise.all([fetchStudents(), fetchAuditEvents(), fetchSnapshots()]);
}

/* ── Render Helpers ───────────────────────────────────── */

export function renderStudentCards() {
  const grid = document.getElementById("studentGrid");
  if (!grid) return;

  if (!localStudents.length) {
    grid.innerHTML = '<div class="p-5 text-center" style="color:rgba(255,255,255,0.4);font-size:14px">No candidates online yet.</div>';
    return;
  }

  grid.innerHTML = localStudents.map(s => {
    const cc = s.score >= 70 ? "danger" : s.score >= 30 ? "warn" : "";
    let badge;
    if (s.submitted) {
      badge = '<span class="badge badge-green">Submitted</span>';
    } else if (!s.started) {
      badge = '<span class="badge" style="background:rgba(255,255,255,0.08);color:rgba(255,255,255,0.5)">Waiting</span>';
    } else if (s.score >= 70) {
      badge = '<span class="badge" style="background:rgba(231,76,60,0.2);color:#e74c3c">' + s.status + '</span>';
    } else if (s.score >= 30) {
      badge = '<span class="badge badge-orange">' + s.status + '</span>';
    } else {
      badge = '<span class="badge badge-green">Normal</span>';
    }

    const snapCount = localSnapshots.filter(x => x.app_number === s.app_number).length;
    const snapBtn = snapCount > 0
      ? `<button onclick="window.openStudentSnapshots('${s.app_number}','${s.name}')"
           style="font-size:11px;padding:4px 10px;background:rgba(88,101,242,0.15);color:#5865f2;border:none;border-radius:12px;cursor:pointer;margin-top:6px">
           ${snapCount} snap${snapCount > 1 ? 's' : ''}</button>`
      : "";

    return `<div class="student-card ${cc}">
              <div class="student-card-body">
                <div class="student-card-name">${s.name}</div>
                ${badge}
                <div class="student-card-meta">Risk: ${s.score}/100 \u00b7 ${s.events} events</div>
                ${snapBtn}
              </div>
            </div>`;
  }).join("");
}

function renderCharts() {
  /* Events by type */
  const typeCounts = {};
  localAuditLog.filter(e => e.level !== "info").forEach(e => {
    typeCounts[e.type] = (typeCounts[e.type] || 0) + 1;
  });
  const maxT = Math.max(...Object.values(typeCounts), 1);
  const tc = document.getElementById("typeChart");
  if (tc) {
    tc.innerHTML = Object.entries(typeCounts)
      .sort((a, b) => b[1] - a[1])
      .map(([t, c]) =>
        `<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:13px">
          <div style="width:130px;color:rgba(255,255,255,0.7);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${t}</div>
          <div class="chart-bar-track">
            <div class="chart-bar-fill ${c <= 2 ? 'warning' : ''}" style="width:${Math.round(c/maxT*100)}%"></div>
          </div>
          <div style="width:24px;text-align:right;color:rgba(255,255,255,0.4);font-size:12px">${c}</div>
        </div>`)
      .join("") || '<div style="color:rgba(255,255,255,0.4);font-size:13px">No events yet.</div>';
  }

  /* Risk leaderboard */
  const sorted = [...localStudents].sort((a, b) => b.score - a.score);
  const maxR   = Math.max(...sorted.map(s => s.score), 1);
  const rc = document.getElementById("riskChart");
  if (rc) {
    rc.innerHTML = sorted.slice(0, 6).map(s =>
      `<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:13px">
        <div style="width:130px;color:rgba(255,255,255,0.7);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${s.name}</div>
        <div class="chart-bar-track">
          <div class="chart-bar-fill ${s.score >= 70 ? 'danger' : s.score >= 30 ? 'warning' : ''}" style="width:${Math.round(s.score/maxR*100)}%"></div>
        </div>
        <div style="width:30px;text-align:right;font-weight:600;color:${s.score >= 70 ? '#e74c3c' : s.score >= 30 ? '#f0ad4e' : '#35ed7e'}">${s.score}</div>
      </div>`)
      .join("") || '<div style="color:rgba(255,255,255,0.4);font-size:13px">No data yet.</div>';
  }
}

function renderLatestFlags() {
  const body = document.getElementById("latestBody");
  if (!body) return;
  const flags = localAuditLog.filter(e => e.level !== "info").slice(0, 8);
  if (!flags.length) {
    body.innerHTML = '<tr><td colspan="3" style="text-align:center;color:rgba(255,255,255,0.4);padding:20px;font-size:13px">No flags yet.</td></tr>';
    return;
  }
  body.innerHTML = flags.map((e, i) =>
    `<tr class="violation-entry ${i === 0 ? 'highlight' : ''}">
      <td style="color:rgba(255,255,255,0.5);font-size:12px">${e.time}</td>
      <td style="font-size:13px;font-weight:500">${e.student}</td>
      <td>
        <span class="badge" style="background:${e.level === 'danger' ? 'rgba(231,76,60,0.2)' : 'rgba(240,173,78,0.2)'};color:${e.level === 'danger' ? '#e74c3c' : '#f0ad4e'}">
          ${e.type}
        </span>
      </td>
    </tr>`
  ).join("");
}

export function renderAuditTable() {
  const body = document.getElementById("auditBody");
  if (!body) return;
  const search = (document.getElementById("auditSearch")?.value || "").toLowerCase();
  const filter = document.getElementById("auditType")?.value || "";
  const list   = localAuditLog.filter(e =>
    e.level !== "info" &&
    e.student.toLowerCase().includes(search) &&
    (!filter || e.type === filter)
  );

  if (!list.length) {
    body.innerHTML = '<tr><td colspan="6" style="text-align:center;color:rgba(255,255,255,0.4);padding:30px;font-size:13px">No events found.</td></tr>';
    return;
  }

  body.innerHTML = list.map(e =>
    `<tr class="violation-entry">
      <td style="color:rgba(255,255,255,0.5);font-size:12px">${e.time}</td>
      <td style="font-weight:500">${e.student}</td>
      <td>
        <span class="badge" style="background:${e.level === 'danger' ? 'rgba(231,76,60,0.2)' : 'rgba(240,173,78,0.2)'};color:${e.level === 'danger' ? '#e74c3c' : '#f0ad4e'}">
          ${e.type}
        </span>
        ${e.note ? '<div style="font-size:11px;color:rgba(255,255,255,0.4);margin-top:2px">' + e.note + '</div>' : ""}
      </td>
      <td style="font-size:12px;color:rgba(255,255,255,0.6)">${e.conf}%</td>
      <td style="font-size:12px;color:#f0ad4e">+${e.impact}</td>
      <td style="font-size:12px;color:rgba(255,255,255,0.4)">${e.note || '\u2014'}</td>
    </tr>`
  ).join("");
}

export function renderSnapshotView() {
  const grid = document.getElementById("snapGrid");
  if (!grid) return;
  const search = (document.getElementById("snapSearch")?.value || "").toLowerCase();
  const filter = document.getElementById("snapType")?.value || "";
  const list   = localSnapshots.filter(s =>
    s.student_name.toLowerCase().includes(search) &&
    (!filter || s.event_type === filter)
  );

  if (!list.length) {
    grid.innerHTML = '<div style="grid-column:1/-1;text-align:center;color:rgba(255,255,255,0.4);padding:20px;font-size:13px">No snapshots yet.</div>';
    return;
  }
  grid.innerHTML = list.map(s =>
    `<div class="snap-card" onclick="window.openSnapshotModal('${s.id}')">
      <img src="${s.image}" loading="lazy" alt="Snapshot"/>
      <div style="padding:8px;background:#000">
        <div style="color:#e74c3c;font-size:11px;font-weight:600">${s.event_type}</div>
        <div style="font-weight:600;font-size:12px;color:#ddd;margin-top:2px">${s.student_name}</div>
        <div style="font-size:10px;color:rgba(255,255,255,0.4);margin-top:2px">${s.time}</div>
      </div>
    </div>`
  ).join("");
}

/* ── Modal Helpers ────────────────────────────────────── */

window.openStudentSnapshots = function(appNo, name) {
  const list = localSnapshots.filter(s => s.app_number === appNo);
  const mt = document.getElementById("modalTitle");
  const mm = document.getElementById("modalMeta");
  const eg = document.getElementById("evidenceGrid");
  if (mt) mt.textContent = "Snapshots \u2014 " + name;
  if (mm) mm.textContent = list.length + " snapshot(s)";
  if (eg) {
    eg.innerHTML = list.map(s =>
      `<div style="border-radius:16px;overflow:hidden;background:#000">
        <img src="${s.image}" style="width:100%;height:140px;object-fit:cover;display:block" loading="lazy"/>
        <div style="padding:8px;font-size:12px">
          <div style="color:#e74c3c;font-weight:600">${s.event_type}</div>
          <div style="color:rgba(255,255,255,0.5);font-size:11px">${s.time}</div>
        </div>
      </div>`
    ).join("");
  }
  const em = document.getElementById("evidenceModal");
  if (em) em.classList.add("show");
};

window.openSnapshotModal = function(id) {
  const s = localSnapshots.find(x => x.id === id);
  if (!s) return;
  const mt = document.getElementById("modalTitle");
  const mm = document.getElementById("modalMeta");
  const eg = document.getElementById("evidenceGrid");
  if (mt) mt.textContent = s.event_type + " \u2014 " + s.student_name;
  if (mm) mm.textContent = "Captured at " + s.time;
  if (eg) {
    eg.innerHTML =
      `<div style="grid-column:1/-1;border-radius:16px;overflow:hidden;background:#000">
        <img src="${s.image}" style="width:100%;max-height:420px;object-fit:contain;display:block"/>
        <div style="padding:8px;font-size:12px">
          <div style="color:#e74c3c;font-weight:600">${s.event_type}</div>
          <div>${s.student_name} \u00b7 ${s.time}</div>
        </div>
      </div>`;
  }
  const em = document.getElementById("evidenceModal");
  if (em) em.classList.add("show");
};

window.closeEvidence = function() {
  const em = document.getElementById("evidenceModal");
  if (em) em.classList.remove("show");
};

/* ── Teacher Navigation ──────────────────────────────── */

window.switchTeacherView = function(view) {
  teacherView = view;
  document.querySelectorAll(".t-page").forEach(p => p.classList.remove("active"));
  const p = document.getElementById("t-" + view);
  if (p) p.classList.add("active");
  if (view === "audit")     renderAuditTable();
  if (view === "snapshots") renderSnapshotView();
  if (view === "dash") {
    renderStudentCards();
    renderCharts();
    renderLatestFlags();
  }
};

window.teacherLogout = async function() {
  clearInterval(teacherPollInterval);
  clearInterval(teacherTimerInt);
  localStudents  = [];
  localAuditLog  = [];
  localSnapshots = [];
  await teacherLogoutApi();
  showPage("loginPage");
  toast("Logged out");
};

window.refreshAll = function() { refreshAll(); };

window.exportAudit = async function() {
  try {
    const res = await fetch(getApiBase() + "/api/proctor/export", { credentials: "include" });
    if (res.ok) {
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "audit-report.csv";
      a.click();
      return;
    }
  } catch {}
  const rows = [["Time", "Student", "Event", "Confidence", "Impact", "Note"]];
  localAuditLog.filter(e => e.level !== "info")
    .forEach(e => rows.push([e.time, e.student, e.type, e.conf + "%", "+" + e.impact, e.note]));
  const csv = rows.map(r => r.join(",")).join("\n");
  const a = document.createElement("a");
  a.href = "data:text/csv," + encodeURIComponent(csv);
  a.download = "audit-report.csv";
  a.click();
};

export function startTeacherDashboard() {
  function tick() {
    const n = new Date();
    const t = document.getElementById("timer");
    if (t) t.textContent = pad(n.getHours()) + ":" + pad(n.getMinutes()) + ":" + pad(n.getSeconds());
  }
  tick();
  teacherTimerInt = setInterval(tick, 1000);

  startLiveFeed();
  refreshAll();
  teacherPollInterval = setInterval(refreshAll, 4000);
}

/* ── Tab switcher for login page ─────────────────────── */

window.switchTab = function(n) {
  document.querySelectorAll(".tab").forEach((t, i) =>
    t.classList.toggle("active", (i === 0) === (n === "candidate"))
  );
  const tc = document.getElementById("tab-candidate");
  const tt = document.getElementById("tab-teacher");
  if (tc) tc.classList.toggle("active", n === "candidate");
  if (tt) tt.classList.toggle("active", n === "teacher");
};
