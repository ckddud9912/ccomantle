let lastCorrectKey = null;
let pollId = null;
let finalShown = false;  // final-overlay 가 현재 표시 중인지 — 재시작 감지에 사용

const MY_TEAM_KEY = "ccomantle.myTeam";
const MY_COLOR_KEY = "ccomantle.myTeamColor";

const wordInput = document.getElementById("word");
const teamInput = document.getElementById("team");
const teamColorInput = document.getElementById("teamColor");

// 입력란 초기값: 이 탭의 sessionStorage 우선, 없으면 localStorage 폴백.
// session 이 없는 새 탭은 폴백으로 입력란만 채워주고, "내 팀" 식별 자체는
// 이 탭에서 실제로 제출하기 전까진 비어있다 (다른 탭의 팀이 새 탭에 흘러들지 않게).
const initialTeam =
  sessionStorage.getItem(MY_TEAM_KEY) || localStorage.getItem(MY_TEAM_KEY);
if (initialTeam) teamInput.value = initialTeam;

const initialColor =
  sessionStorage.getItem(MY_COLOR_KEY) || localStorage.getItem(MY_COLOR_KEY);
if (initialColor) teamColorInput.value = initialColor;

// 한글 IME 조합 중에는 Enter를 무시 (조합 미완성 상태로 제출되는 것 방지)
wordInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.isComposing && e.keyCode !== 229) {
    sendGuess();
  }
});

// "내 팀" = 이 탭에서 마지막으로 제출한 팀 (sessionStorage 라 탭별 격리).
// 입력란을 단순히 바꾸는 것만으론 바뀌지 않음 — 같은 브라우저 다른 탭의
// 다른 팀이 흘러들거나, 오타로 잠깐 바뀐 값이 하이라이트에 반영되는 걸 막기 위함.
function getMyTeam() {
  return (sessionStorage.getItem(MY_TEAM_KEY) || "").trim();
}

// rank/is_answer 기반으로 막대 색상 결정 (가까울수록 초록, 멀수록 빨강)
function barColorFor(row) {
  if (row.is_answer) return "var(--accent)";
  if (!row.rank || row.rank > 1000) return "var(--border-input)";
  // rank 1 → t=1 (초록 hue 130), rank 1000 → t=0 (빨강 hue 0)
  const t = 1 - (row.rank - 1) / 999;
  const hue = Math.round(t * 130);
  return `hsl(${hue}, 70%, 48%)`;
}

document.getElementById("final-close").addEventListener("click", () => {
  document.getElementById("final-overlay").style.display = "none";
});

function showLatest(text, kind = "info") {
  const el = document.getElementById("latest");
  el.innerText = text;
  el.dataset.kind = kind;  // info | error | success
  if (kind === "error") {
    wordInput.classList.add("input-shake");
    setTimeout(() => wordInput.classList.remove("input-shake"), 500);
  }
}

async function sendGuess() {
  const team = document.getElementById("team").value.trim();
  const word = wordInput.value.trim();
  const team_color = document.getElementById("teamColor").value;

  if (!team) return showLatest("팀 이름을 먼저 입력해주세요.", "error");
  if (!word) return showLatest("단어를 입력해주세요.", "error");

  sessionStorage.setItem(MY_TEAM_KEY, team);
  localStorage.setItem(MY_TEAM_KEY, team);
  sessionStorage.setItem(MY_COLOR_KEY, team_color);
  localStorage.setItem(MY_COLOR_KEY, team_color);

  try {
    const res = await fetch("/guess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ team, word, team_color }),
    });
    const data = await res.json();

    if (data.error) {
      showLatest(data.error, "error");
      return;
    }

    if (data.result === "duplicate") {
      showLatest("이미 이 라운드에서 제출했습니다.", "error");
      return;
    }

    if (data.entry) {
      const e = data.entry;
      const msg =
        `[${e.round}R][${e.team}] ${e.word} → 유사도 ${e.similarity?.toFixed(3) ?? "-"}` +
        (e.rank ? ` / ${e.rank}위` : "");
      showLatest(msg, data.result === "correct" ? "success" : "info");

      if (data.result === "correct") {
        lastCorrectKey = `${e.round}::${e.team}::${e.word}`;
      }
    }

    wordInput.value = "";
    await loadBoard();
  } catch (err) {
    console.error(err);
    showLatest("네트워크 오류. 잠시 후 다시 시도해주세요.", "error");
  }
}

async function loadBoard() {
  try {
    const res = await fetch("/leaderboard");
    const data = await res.json();

    if (data.finished) {
      // 한 번만 final 결과 로드. 폴링은 멈추지 않음 (재시작 감지를 위해)
      if (!finalShown) {
        finalShown = true;
        await loadFinalResult();
      }
      return;
    }

    // 종료 상태였다가 풀려난 경우 = 어드민이 재시작함. 오버레이 닫고 보드 정리
    if (finalShown) {
      finalShown = false;
      lastCorrectKey = null;
      document.getElementById("final-overlay").style.display = "none";
    }

    const currentRound = data.current_round;
    const maxRounds = data.max_rounds;
    const rounds = data.rounds || {};

    document.getElementById("round-label").innerText =
      `현재 라운드: ${currentRound} / ${maxRounds}`;
    document.getElementById("current-round-text").innerText = `${currentRound} 라운드`;
    renderRoundProgress(currentRound, maxRounds);

    document.getElementById("sim-top1").innerText =
      data.sim_top1 ? data.sim_top1.toFixed(3) : "-";
    document.getElementById("sim-top20").innerText =
      data.sim_top20 ? data.sim_top20.toFixed(3) : "-";
    document.getElementById("sim-top1000").innerText =
      data.sim_top1000 ? data.sim_top1000.toFixed(3) : "-";

    document.getElementById("answer-label").innerText = data.answer
      ? "현재 정답 단어는 어드민에만 표시됩니다."
      : "정답이 아직 설정되지 않았습니다.";

    renderCurrentRound(rounds[String(currentRound)] || [], currentRound);
    renderPastRounds(rounds, currentRound);
  } catch (err) {
    console.error(err);
  }
}

function renderCurrentRound(rows, roundNo) {
  const tbody = document.getElementById("tbody-current");
  tbody.innerHTML = "";
  const myTeam = getMyTeam();

  rows.forEach((row, idx) => {
    const tr = document.createElement("tr");
    if (myTeam && row.team === myTeam) tr.classList.add("row-mine");

    const tdRank = document.createElement("td");
    tdRank.textContent = idx + 1;
    tr.appendChild(tdRank);

    const tdTeam = document.createElement("td");
    tdTeam.textContent = row.team;
    tdTeam.className = "row-team";
    tdTeam.style.color = row.team_color || "#3b82f6";
    tr.appendChild(tdTeam);

    const tdWord = document.createElement("td");
    tdWord.textContent = row.word;
    tr.appendChild(tdWord);

    const tdSim = document.createElement("td");
    if (typeof row.similarity === "number") {
      tdSim.textContent = row.similarity.toFixed(3);
      tdSim.className = row.similarity >= 0.6 ? "sim-good" : "sim-bad";
    } else {
      tdSim.textContent = "-";
    }
    tr.appendChild(tdSim);

    const tdRankInfo = document.createElement("td");
    const pill = document.createElement("span");
    pill.className = "rank-pill";
    pill.textContent = formatRank(row);
    if (row.is_answer) pill.classList.add("rank-answer");
    tdRankInfo.appendChild(pill);
    tr.appendChild(tdRankInfo);

    const tdBar = document.createElement("td");
    const box = document.createElement("div");
    box.className = "bar-box";
    const fill = document.createElement("div");
    fill.className = "bar-fill";

    let pct = 0;
    if (row.is_answer) {
      pct = 100;
    } else if (row.rank && row.rank <= 1000) {
      pct = Math.max(5, ((1000 - row.rank + 1) / 1000) * 100);
    }
    fill.style.width = pct + "%";
    fill.style.background = barColorFor(row);

    box.appendChild(fill);
    tdBar.appendChild(box);
    tr.appendChild(tdBar);

    const key = `${roundNo}::${row.team}::${row.word}`;
    if (lastCorrectKey && key === lastCorrectKey) {
      tr.style.outline = "2px solid #facc15";
      tr.style.boxShadow = "0 0 10px rgba(250, 204, 21, 0.8)";
      setTimeout(() => {
        tr.style.outline = "none";
        tr.style.boxShadow = "none";
        lastCorrectKey = null;
      }, 1200);
    }

    tbody.appendChild(tr);
  });

  if (rows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 6;
    td.className = "no-data";
    td.textContent = "아직 제출된 단어가 없습니다.";
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
}

function formatRank(row) {
  if (row.is_answer) return "정답!";
  if (!row.rank || row.rank > 1000) return "1000위 이상";
  return `${row.rank}위`;
}

function renderPastRounds(rounds, currentRound) {
  const container = document.getElementById("past-all");
  container.innerHTML = "";
  const myTeam = getMyTeam();

  // 최신 라운드를 위로 (역순)
  let any = false;
  for (let r = currentRound - 1; r >= 1; r--) {
    const list = rounds[String(r)] || [];
    if (!list.length) continue;
    any = true;

    const wrapper = document.createElement("div");
    wrapper.className = "past-table-wrapper";

    const title = document.createElement("div");
    title.style.cssText = "display:flex;justify-content:space-between;align-items:baseline;";

    const h = document.createElement("h3");
    h.textContent = `${r} 라운드`;
    title.appendChild(h);

    const sub = document.createElement("span");
    sub.style.cssText = "font-size:11px;color:var(--text-muted);";
    sub.textContent = `${list.length}개 제출`;
    title.appendChild(sub);

    wrapper.appendChild(title);

    const table = document.createElement("table");
    const thead = document.createElement("thead");
    thead.innerHTML = `
      <tr>
        <th>순위</th><th>팀</th><th>단어</th><th>유사도</th><th>유사도 순위</th>
      </tr>`;
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    list.forEach((row, idx) => {
      const tr = document.createElement("tr");
      if (myTeam && row.team === myTeam) tr.classList.add("row-mine");

      const tdRank = document.createElement("td");
      tdRank.textContent = idx + 1;
      tr.appendChild(tdRank);

      const tdTeam = document.createElement("td");
      tdTeam.textContent = row.team;
      tdTeam.style.color = row.team_color || "#3b82f6";
      tr.appendChild(tdTeam);

      const tdWord = document.createElement("td");
      tdWord.textContent = row.word;
      tr.appendChild(tdWord);

      const tdSim = document.createElement("td");
      tdSim.textContent =
        typeof row.similarity === "number" ? row.similarity.toFixed(3) : "-";
      tr.appendChild(tdSim);

      const tdRankInfo = document.createElement("td");
      tdRankInfo.textContent = formatRank(row);
      tr.appendChild(tdRankInfo);

      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    wrapper.appendChild(table);
    container.appendChild(wrapper);
  }

  if (!any) {
    const n = document.createElement("div");
    n.className = "no-data";
    n.textContent = "아직 종료된 라운드가 없습니다.";
    container.appendChild(n);
  }
}

function renderRoundProgress(currentRound, maxRounds) {
  const dots = document.querySelectorAll("#round-progress .rp-dot");
  dots.forEach((el, i) => {
    el.classList.remove("rp-done", "rp-active");
    if (i + 1 < currentRound) el.classList.add("rp-done");
    else if (i + 1 === currentRound) el.classList.add("rp-active");
  });
}

async function loadFinalResult() {
  try {
    const res = await fetch("/final_result");
    const data = await res.json();

    // 정답 단어 공개
    const titleEl = document.getElementById("final-answer");
    if (titleEl) {
      titleEl.textContent = data.answer ? `정답: ${data.answer}` : "";
    }

    const teams = data.result || [];

    renderFinalWordRanking(teams);
    renderFinalTeamRanking(teams);

    document.getElementById("final-overlay").style.display = "flex";
  } catch (e) {
    console.error(e);
  }
}

// 섹션 1: 모든 제출 단어를 점수순으로
function renderFinalWordRanking(teams) {
  const tbody = document.getElementById("final-words-tbody");
  tbody.innerHTML = "";

  // 모든 팀의 모든 submission 을 평탄화
  const allWords = [];
  teams.forEach((t) => {
    (t.submissions || []).forEach((sub) => {
      allWords.push({
        team: t.team,
        team_color: t.team_color,
        round: sub.round,
        word: sub.word,
        similarity: sub.similarity,
      });
    });
  });

  // 유사도 내림차순
  allWords.sort((a, b) => (b.similarity ?? 0) - (a.similarity ?? 0));

  if (!allWords.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.className = "no-data";
    td.textContent = "제출된 단어가 없습니다.";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  allWords.forEach((row, idx) => {
    const tr = document.createElement("tr");
    if (idx === 0) tr.classList.add("row-top1");

    appendCell(tr, idx + 1);
    appendCell(tr, row.word);
    appendTeamCell(tr, row.team, row.team_color);
    appendCell(tr, `R${row.round}`);
    appendCell(
      tr,
      typeof row.similarity === "number" ? row.similarity.toFixed(3) : "-"
    );

    tbody.appendChild(tr);
  });
}

// 섹션 2: 팀별 평균 + 베스트 단어 한 개
function renderFinalTeamRanking(teams) {
  const tbody = document.getElementById("final-teams-tbody");
  tbody.innerHTML = "";

  teams.forEach((row, idx) => {
    // 팀의 베스트 submission 찾기
    const submissions = row.submissions || [];
    const best = submissions.reduce(
      (acc, s) =>
        s.similarity != null && (acc == null || s.similarity > acc.similarity)
          ? s
          : acc,
      null
    );

    const tr = document.createElement("tr");
    if (idx === 0) tr.classList.add("row-top1");

    appendCell(tr, idx + 1);
    appendTeamCell(tr, row.team, row.team_color);
    appendCell(tr, (row.avg || 0).toFixed(3));

    const tdBest = document.createElement("td");
    if (best) {
      tdBest.textContent = `${best.word} (R${best.round}, ${best.similarity.toFixed(3)})`;
    } else {
      tdBest.textContent = "-";
    }
    tr.appendChild(tdBest);

    tbody.appendChild(tr);
  });
}

function appendCell(tr, text) {
  const td = document.createElement("td");
  td.textContent = text;
  tr.appendChild(td);
}

function appendTeamCell(tr, team, color) {
  const td = document.createElement("td");
  td.textContent = team;
  td.style.color = color || "#3b82f6";
  td.style.fontWeight = "600";
  tr.appendChild(td);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

loadBoard();
pollId = setInterval(loadBoard, 1500);
