const answerInput = document.getElementById("answer-input");
const answerStatus = document.getElementById("answer-status");
const roundStatus = document.getElementById("round-status");
const gameStatus = document.getElementById("game-status");

document.getElementById("btn-set-answer").addEventListener("click", setAnswer);
document.getElementById("btn-top1000").addEventListener("click", openTop1000);
document.getElementById("btn-end-game").addEventListener("click", endGame);

document.querySelectorAll(".round-buttons button").forEach((btn) => {
  btn.addEventListener("click", () => changeRound(parseInt(btn.dataset.round, 10)));
});

document.getElementById("top1000-close").addEventListener("click", () => {
  document.getElementById("top1000-overlay").style.display = "none";
});

async function setAnswer() {
  const answer = answerInput.value.trim();
  if (!answer) return;

  try {
    const res = await fetch("/set_answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer }),
    });
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    answerStatus.textContent = `현재 정답: ${data.answer}`;
    gameStatus.textContent = "경기 진행 중";
  } catch (e) {
    console.error(e);
  }
}

async function changeRound(round) {
  try {
    const res = await fetch("/set_round", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ round }),
    });
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    roundStatus.textContent = `현재 라운드: ${data.current_round}`;
  } catch (e) {
    console.error(e);
  }
}

async function endGame() {
  if (!confirm("정말로 경기를 종료하시겠습니까?")) return;
  try {
    const res = await fetch("/end_game", { method: "POST" });
    const data = await res.json();
    if (data.status === "finished") {
      gameStatus.textContent = "경기 종료됨 (게임 페이지에서 최종 결과가 표시됩니다)";
      alert("경기가 종료되었습니다. 게임 페이지에서 최종 결과가 표시됩니다.");
    }
  } catch (e) {
    console.error(e);
  }
}

async function openTop1000() {
  try {
    const res = await fetch("/top1000");
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      return;
    }

    document.getElementById("top1000-title").textContent =
      `정답 단어: ${data.answer ?? "-"}`;

    const tbody = document.getElementById("top1000-tbody");
    tbody.innerHTML = "";
    (data.top1000 || []).forEach((row) => {
      const tr = document.createElement("tr");

      const tdRank = document.createElement("td");
      tdRank.textContent = row.rank;
      tr.appendChild(tdRank);

      const tdWord = document.createElement("td");
      tdWord.textContent = row.word;
      tr.appendChild(tdWord);

      const tdSim = document.createElement("td");
      tdSim.textContent = row.similarity.toFixed(4);
      tr.appendChild(tdSim);

      tbody.appendChild(tr);
    });

    document.getElementById("top1000-overlay").style.display = "flex";
  } catch (e) {
    console.error(e);
  }
}

// 초기 상태 동기화
(async function initStatus() {
  try {
    const res = await fetch("/leaderboard");
    const data = await res.json();
    if (data.answer) {
      answerStatus.textContent = `현재 정답: ${data.answer}`;
    }
    roundStatus.textContent = `현재 라운드: ${data.current_round ?? "-"}`;
    gameStatus.textContent = data.finished ? "경기 종료됨" : "경기 진행 중";
  } catch (e) {
    console.error(e);
  }
})();
