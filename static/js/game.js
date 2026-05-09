let lastCorrectKey = null;
let pollId = null;

const wordInput = document.getElementById("word");

// 한글 IME 조합 중에는 Enter를 무시 (조합 미완성 상태로 제출되는 것 방지)
wordInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.isComposing && e.keyCode !== 229) {
    sendGuess();
  }
});

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
      if (pollId) {
        clearInterval(pollId);
        pollId = null;
      }
      await loadFinalResult();
      return;
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

  rows.forEach((row, idx) => {
    const tr = document.createElement("tr");

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

    // 정답 단어 공개 (게임 끝났을 때 가장 궁금한 정보)
    const titleEl = document.getElementById("final-answer");
    if (titleEl) {
      titleEl.textContent = data.answer ? `정답: ${data.answer}` : "";
    }

    const tbody = document.getElementById("final-tbody");
    tbody.innerHTML = "";

    (data.result || []).forEach((row, idx) => {
      const tr = document.createElement("tr");

      const tdRank = document.createElement("td");
      tdRank.textContent = idx + 1;
      tr.appendChild(tdRank);

      const tdTeam = document.createElement("td");
      tdTeam.textContent = row.team;
      tdTeam.style.color = row.team_color || "#3b82f6";
      tr.appendChild(tdTeam);

      const tdAvg = document.createElement("td");
      tdAvg.textContent = (row.avg || 0).toFixed(3);
      tr.appendChild(tdAvg);

      // 라운드별 제출 단어 + 유사도 칩 형태로
      const tdWords = document.createElement("td");
      tdWords.className = "final-words";
      (row.submissions || []).forEach((sub) => {
        const item = document.createElement("span");
        item.className = "fw-item";
        item.innerHTML =
          `<span class="fw-round">R${sub.round}</span>` +
          `<span class="fw-word">${escapeHtml(sub.word)}</span>` +
          `<span class="fw-sim">${
            typeof sub.similarity === "number" ? sub.similarity.toFixed(3) : "-"
          }</span>`;
        tdWords.appendChild(item);
      });
      tr.appendChild(tdWords);

      tbody.appendChild(tr);
    });

    document.getElementById("final-overlay").style.display = "flex";
  } catch (e) {
    console.error(e);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

loadBoard();
pollId = setInterval(loadBoard, 1500);
