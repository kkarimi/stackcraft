"use strict";

// Python supplies every legal landing cell. This client never simulates a move.
const $ = (id) => document.getElementById(id);
const colors = { I: 1, O: 2, T: 3, S: 4, Z: 5, J: 6, L: 7 };
// Icons only: these coordinates are never used to calculate placements.
const icons = {
  I: [[0, 0], [1, 0], [2, 0], [3, 0]], O: [[1, 0], [2, 0], [1, 1], [2, 1]],
  T: [[1, 0], [0, 1], [1, 1], [2, 1]], S: [[1, 0], [2, 0], [0, 1], [1, 1]],
  Z: [[0, 0], [1, 0], [1, 1], [2, 1]], J: [[0, 0], [0, 1], [1, 1], [2, 1]],
  L: [[2, 0], [0, 1], [1, 1], [2, 1]],
};
let live = null;
let selected = null;
let busy = false;
let stateSynced = false;
let frames = null;
let frameIndex = 0;
let replayArtifact = null;
let replayTimer = null;
let activeSeed = 42;
let raceVisible = false;
let raceLoading = false;
let racePlayers = null;
let raceIndex = 0;
let raceMax = 0;
let raceTimer = null;
const cells = Array.from({ length: 200 }, () => {
  const cell = document.createElement("div");
  cell.className = "cell";
  $("board").append(cell);
  return cell;
});

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const result = await response.json();
  if (!response.ok) {
    const detail = typeof result.detail === "string" ? result.detail : "The request could not be completed. Check the input and try again.";
    throw new Error(detail);
  }
  return result;
}

function reportError(error) {
  $("error").textContent = error.message || "Something went wrong. Please try again.";
  $("error").hidden = false;
}

async function operation(task) {
  if (busy) return;
  busy = true;
  $("error").hidden = true;
  renderControls();
  try { await task(); } catch (error) { reportError(error); }
  finally { busy = false; renderControls(); }
}

function currentState() { return frames ? frames[frameIndex] : live; }
function legal() { return live?.legal_actions || []; }
function chooseNearest(rotation = 0, x = 3) {
  const sameRotation = legal().filter((action) => action.rotation === rotation);
  const choices = sameRotation.length ? sameRotation : legal();
  selected = choices.reduce((best, action) => !best || Math.abs(action.x - x) < Math.abs(best.x - x) ? action : best, null);
}

function renderPreview(id, piece) {
  const preview = $(id);
  preview.replaceChildren();
  preview.setAttribute("aria-label", `${id === "current-preview" ? "Current" : "Next"} piece: ${piece || "none"}`);
  const grid = document.createElement("div");
  grid.className = "mini-grid";
  grid.style.setProperty("--piece-color", `var(--piece-${colors[piece] || 1})`);
  for (let y = 0; y < 2; y++) {
    for (let x = 0; x < 4; x++) {
      const cell = document.createElement("span");
      cell.className = `mini-cell${icons[piece]?.some(([px, py]) => px === x && py === y) ? " filled" : ""}`;
      grid.append(cell);
    }
  }
  preview.append(grid);
}

function renderControls() {
  const playable = !busy && stateSynced && live && !live.terminal && !frames && selected;
  for (const id of ["move-left", "move-right", "rotate", "drop"]) $(id).disabled = !playable;
  for (const id of ["restart", "play-again", "load-replay", "exit-replay"]) $(id).disabled = busy;
  $("download-replay").disabled = busy || (!live && !replayArtifact);
  $("replay-prev").disabled = busy || !frames || frameIndex === 0;
  $("replay-next").disabled = busy || !frames || frameIndex >= frames.length - 1;
  $("replay-play").disabled = busy || !frames || frames.length <= 1;
  $("replay-position").disabled = busy;
  $("play-controls").hidden = Boolean(frames);
  $("replay-controls").hidden = !frames;
}

function render() {
  const state = currentState();
  if (!state) return renderControls();
  const ghost = !frames && !state.terminal ? new Set(selected?.cells.map(([x, y]) => y * 10 + x)) : new Set();
  for (let i = 0; i < cells.length; i++) {
    const value = state.board[Math.floor(i / 10)][i % 10];
    cells[i].className = `cell${value ? " filled" : ghost.has(i) ? " ghost" : ""}`;
    cells[i].style.setProperty("--piece-color", `var(--piece-${value || colors[state.current] || 1})`);
  }
  $("score").textContent = state.score.toLocaleString();
  $("lines").textContent = state.lines.toLocaleString();
  $("pieces").textContent = state.pieces.toLocaleString();
  $("current-name").textContent = state.current;
  $("move-counter").textContent = `MOVE ${String(state.pieces + (state.terminal ? 0 : 1)).padStart(3, "0")}`;
  $("board-mode").textContent = frames ? "SAVED REPLAY" : "YOUR BOARD";
  $("player-status").textContent = frames ? "Watching a recorded game" : !stateSynced ? "Connection lost · restart to continue" : state.terminal ? "Run complete" : "Ready for your next move";
  $("terminal-overlay").hidden = !state.terminal || Boolean(frames);
  $("placement-label").textContent = frames ? `Recorded move ${frameIndex}` : state.terminal ? "No legal placements" : selected ? `COL ${selected.x + 1} / ROT ${selected.rotation + 1}` : "Choose your landing spot";
  $("board").setAttribute("aria-label", `${frames ? "Replay" : "Game"} board. ${state.pieces} pieces placed, ${state.lines} lines, score ${state.score}.${state.terminal ? " Game over." : selected && !frames ? ` ${state.current} piece at column ${selected.x + 1}, rotation ${selected.rotation + 1}. Use left and right to move, up to rotate, space to drop.` : ""}`);
  renderPreview("current-preview", state.current);
  renderPreview("next-preview", state.next_piece);
  if (frames) {
    $("replay-position").max = frames.length - 1;
    $("replay-position").value = frameIndex;
    $("replay-progress").textContent = `${frameIndex} / ${frames.length - 1}`;
  }
  renderControls();
}

function stopReplay() {
  if (replayTimer !== null) clearInterval(replayTimer);
  replayTimer = null;
  $("replay-play").textContent = "Play";
  $("replay-play").setAttribute("aria-label", "Play replay");
}

async function restart() {
  const text = $("seed").value.trim();
  const seed = Number(text);
  if (!text || !Number.isSafeInteger(seed) || seed < 0 || seed > 2147483647) {
    reportError(new Error("Choose a whole-number seed between 0 and 2,147,483,647."));
    return;
  }
  await operation(async () => {
    const snapshot = await api("/api/games", { method: "POST", body: JSON.stringify({ seed }) });
    stopReplay();
    frames = null;
    replayArtifact = null;
    live = snapshot;
    stateSynced = true;
    activeSeed = seed;
    chooseNearest();
    render();
    $("announcement").textContent = `New game, seed ${seed}. ${live.current} piece ready.`;
  });
}

function move(direction) {
  if (busy || !stateSynced || frames || !selected || live.terminal) return;
  const row = legal().filter((action) => action.rotation === selected.rotation).sort((a, b) => a.x - b.x);
  const index = row.findIndex((action) => action.id === selected.id);
  selected = row[Math.max(0, Math.min(row.length - 1, index + direction))];
  render();
}

function rotate() {
  if (busy || !stateSynced || frames || !selected || live.terminal) return;
  const rotations = [...new Set(legal().map((action) => action.rotation))].sort((a, b) => a - b);
  const nextRotation = rotations[(rotations.indexOf(selected.rotation) + 1) % rotations.length];
  chooseNearest(nextRotation, selected.x);
  render();
}

async function drop() {
  if (!live || !stateSynced || live.terminal || !selected || frames) return;
  await operation(async () => {
    const oldLines = live.lines;
    const sessionPath = `/api/games/${encodeURIComponent(live.id)}`;
    try {
      live = await api(`${sessionPath}/moves`, {
        method: "POST",
        body: JSON.stringify({ action_id: selected.id, expected_pieces: live.pieces }),
      });
      stateSynced = true;
    } catch (error) {
      // A failed response can follow an accepted move. Never retry from stale state.
      try {
        live = await api(sessionPath);
        stateSynced = true;
        chooseNearest();
        render();
      } catch {
        stateSynced = false;
        selected = null;
        render();
        throw new Error("Could not confirm the game state. Play is disabled to avoid repeating a move. Reconnect and start a new game.");
      }
      throw error;
    }
    chooseNearest();
    render();
    const clear = live.lines - oldLines;
    $("announcement").textContent = live.terminal ? `Game over. ${live.lines} lines cleared. Score ${live.score}.` : `${clear ? `${clear} ${clear === 1 ? "line" : "lines"} cleared! ` : ""}${live.current} piece ready. Score ${live.score}.`;
  });
}

// Pointer users can immediately continue with arrow/space shortcuts. Keyboard
// activation retains button focus so Tab/Enter navigation stays conventional.
function placementControl(event, action) {
  action();
  if (event.detail > 0) $("board").focus({ preventScroll: true });
}
$("move-left").addEventListener("click", (event) => placementControl(event, () => move(-1)));
$("move-right").addEventListener("click", (event) => placementControl(event, () => move(1)));
$("rotate").addEventListener("click", (event) => placementControl(event, rotate));
$("drop").addEventListener("click", (event) => placementControl(event, drop));
$("restart").addEventListener("click", restart);
$("play-again").addEventListener("click", restart);
$("seed").addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); restart(); } });
document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, select, button, a, summary") || event.ctrlKey || event.metaKey || event.altKey) return;
  if (raceVisible || frames || busy || !stateSynced || !live || live.terminal) return;
  const handler = { ArrowLeft: () => move(-1), ArrowRight: () => move(1), ArrowUp: rotate, " ": drop, Enter: drop }[event.key];
  if (handler) { event.preventDefault(); if (!event.repeat || event.key.startsWith("Arrow")) handler(); }
});
$("board").addEventListener("click", (event) => {
  if (frames || busy || !stateSynced || !selected || live.terminal) return;
  const rect = $("board").getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) / (rect.width / 10));
  chooseNearest(selected.rotation, x);
  render();
  $("board").focus({ preventScroll: true });
});
$("download-replay").addEventListener("click", () => operation(async () => {
  const artifact = frames ? replayArtifact : await api(`/api/games/${encodeURIComponent(live.id)}/replay`);
  const url = URL.createObjectURL(new Blob([JSON.stringify(artifact, null, 2) + "\n"], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `stackcraft-seed-${artifact.seed ?? activeSeed}-replay.json`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}));
$("load-replay").addEventListener("click", () => $("replay-file").click());
$("replay-file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  operation(async () => {
    if (file.size > 5 * 1024 * 1024) throw new Error("Choose a replay smaller than 5 MB.");
    let artifact;
    try { artifact = JSON.parse(await file.text()); } catch { throw new Error("This file is not valid JSON. Choose a Stackcraft replay."); }
    const result = await api("/api/replays", { method: "POST", body: JSON.stringify(artifact) });
    if (!Array.isArray(result.frames) || !result.frames.length) throw new Error("The replay contains no frames.");
    stopReplay();
    frames = result.frames;
    replayArtifact = artifact;
    frameIndex = 0;
    render();
    $("announcement").textContent = `Replay loaded. ${frames.length - 1} recorded moves.`;
  });
});
function seek(index) {
  if (!frames) return;
  frameIndex = Math.max(0, Math.min(frames.length - 1, index));
  render();
}
$("replay-position").addEventListener("input", (event) => { stopReplay(); seek(Number(event.target.value)); });
$("replay-prev").addEventListener("click", () => { stopReplay(); seek(frameIndex - 1); });
$("replay-next").addEventListener("click", () => { stopReplay(); seek(frameIndex + 1); });
$("replay-play").addEventListener("click", () => {
  if (replayTimer !== null) return stopReplay();
  if (!frames || frames.length <= 1) return;
  if (frameIndex === frames.length - 1) seek(0);
  $("replay-play").textContent = "Pause";
  $("replay-play").setAttribute("aria-label", "Pause replay");
  replayTimer = setInterval(() => { seek(frameIndex + 1); if (frameIndex >= frames.length - 1) stopReplay(); }, 450);
});
$("exit-replay").addEventListener("click", () => {
  stopReplay();
  frames = null;
  replayArtifact = null;
  render();
  if (!live) restart();
});

// Race state is separate from the human session and its imported replay.
function stopRace() {
  if (raceTimer !== null) clearInterval(raceTimer);
  raceTimer = null;
  $("race-play").textContent = "Play race ▶";
  $("race-play").setAttribute("aria-label", "Play recorded race");
}

function raceCard(player, index) {
  const card = document.createElement("article");
  card.className = "race-card";
  const heading = document.createElement("div");
  heading.className = "race-card-heading";
  const name = document.createElement("h3");
  name.id = `race-player-${index}`;
  name.textContent = player.name;
  const status = document.createElement("span");
  status.className = "race-player-status";
  heading.append(name, status);
  const wrap = document.createElement("div");
  wrap.className = "board-wrap";
  const board = document.createElement("div");
  board.className = "board race-board";
  board.setAttribute("role", "img");
  const boardCells = Array.from({ length: 200 }, () => {
    const cell = document.createElement("div");
    cell.className = "cell";
    board.append(cell);
    return cell;
  });
  wrap.append(board);
  const metrics = document.createElement("dl");
  metrics.className = "race-metrics";
  const values = {};
  for (const [key, label] of [["lines", "LINES"], ["score", "SCORE"], ["pieces", "PLACED"]]) {
    const group = document.createElement("div");
    const term = document.createElement("dt");
    term.textContent = label;
    const value = document.createElement("dd");
    values[key] = value;
    group.append(term, value);
    metrics.append(group);
  }
  const details = document.createElement("p");
  details.className = "race-piece-detail";
  const revision = document.createElement("p");
  revision.className = "race-revision";
  revision.textContent = player.revision;
  revision.title = `Player revision: ${player.revision}`;
  card.setAttribute("aria-labelledby", name.id);
  card.append(heading, wrap, metrics, details, revision);
  $("race-boards").append(card);
  return { ...player, board, boardCells, status, values, details };
}

function renderRace() {
  if (!racePlayers) return;
  for (const player of racePlayers) {
    const index = Math.min(raceIndex, player.frames.length - 1);
    const state = player.frames[index];
    for (let i = 0; i < player.boardCells.length; i++) {
      const value = state.board[Math.floor(i / 10)][i % 10];
      player.boardCells[i].className = `cell${value ? " filled" : ""}`;
      player.boardCells[i].style.setProperty("--piece-color", `var(--piece-${value || 1})`);
    }
    for (const key of ["lines", "score", "pieces"]) player.values[key].textContent = state[key].toLocaleString();
    const ended = index === player.frames.length - 1;
    player.status.textContent = state.terminal ? "TOPPED OUT" : ended ? "RECORDING ENDED" : "RECORDED";
    player.status.classList.toggle("ended", ended);
    player.details.textContent = state.terminal ? "No legal placements remain" : `Current ${state.current} · Next ${state.next_piece}`;
    player.board.setAttribute("aria-label", `${player.name} recorded board. ${state.pieces} pieces placed, ${state.lines} lines, score ${state.score}.${state.terminal ? " Topped out." : ""}`);
  }
  $("race-position").value = raceIndex;
  $("race-position").max = raceMax;
  $("race-progress").textContent = `${raceIndex} / ${raceMax}`;
  $("race-prev").disabled = raceIndex === 0;
  $("race-next").disabled = raceIndex === raceMax;
  $("race-play").disabled = raceMax === 0;
  $("race-restart").disabled = raceMax === 0;
}

async function loadRace() {
  if (racePlayers || raceLoading) return;
  raceLoading = true;
  $("race-loading").hidden = false;
  $("race-loading").textContent = "Loading recorded baseline games…";
  try {
    const manifest = await api("/static/baseline-demo.json");
    if (manifest.schema_version !== 1 || !Number.isSafeInteger(manifest.seed) || !Number.isSafeInteger(manifest.max_pieces) || manifest.max_pieces < 1 || !Array.isArray(manifest.players) || manifest.players.length < 2 || manifest.players.length > 4) {
      throw new Error("The recorded race manifest has an unsupported format.");
    }
    for (const player of manifest.players) {
      if (typeof player.name !== "string" || typeof player.revision !== "string" || player.replay?.seed !== manifest.seed || !Array.isArray(player.replay?.actions) || player.replay.actions.length > manifest.max_pieces) {
        throw new Error("Race players must use the same seed and the declared episode cap.");
      }
    }
    const loaded = await Promise.all(manifest.players.map(async (player) => {
      const result = await api("/api/replays", { method: "POST", body: JSON.stringify(player.replay) });
      if (!Array.isArray(result.frames) || result.frames.length !== player.replay.actions.length + 1) throw new Error("A race replay has an invalid frame count.");
      return { ...player, frames: result.frames };
    }));
    $("race-boards").replaceChildren();
    racePlayers = loaded.map(raceCard);
    raceMax = Math.max(...racePlayers.map((player) => player.frames.length - 1));
    $("race-label").textContent = typeof manifest.label === "string" ? manifest.label : "Recorded development baseline";
    $("race-context").textContent = `Seed ${manifest.seed} · ${manifest.max_pieces}-piece episode cap · Same seven-bag sequence`;
    $("race-loading").hidden = true;
    $("race-transport").hidden = false;
    renderRace();
  } catch (error) {
    $("race-loading").textContent = `Race unavailable: ${error.message} Switch to Play and back to retry.`;
    $("race-context").textContent = "No comparison results are available yet.";
  } finally { raceLoading = false; }
}

function setView(watchRace) {
  raceVisible = watchRace;
  stopReplay();
  stopRace();
  $("play-view").hidden = watchRace;
  $("race-view").hidden = !watchRace;
  for (const [id, active] of [["play-tab", !watchRace], ["race-tab", watchRace]]) {
    $(id).classList.toggle("active", active);
    $(id).setAttribute("aria-selected", String(active));
    $(id).tabIndex = active ? 0 : -1;
  }
  if (watchRace) loadRace();
}
$("play-tab").addEventListener("click", () => setView(false));
$("race-tab").addEventListener("click", () => setView(true));
for (const id of ["play-tab", "race-tab"]) {
  $(id).addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const watchRace = event.key === "End" || (event.key !== "Home" && !raceVisible);
    setView(watchRace);
    $(watchRace ? "race-tab" : "play-tab").focus();
  });
}
function seekRace(index) {
  if (!racePlayers) return;
  raceIndex = Math.max(0, Math.min(raceMax, index));
  renderRace();
}
$("race-position").addEventListener("input", (event) => { stopRace(); seekRace(Number(event.target.value)); });
$("race-restart").addEventListener("click", () => { stopRace(); seekRace(0); });
$("race-prev").addEventListener("click", () => { stopRace(); seekRace(raceIndex - 1); });
$("race-next").addEventListener("click", () => { stopRace(); seekRace(raceIndex + 1); });
$("race-play").addEventListener("click", () => {
  if (raceTimer !== null) return stopRace();
  if (!racePlayers || raceMax === 0) return;
  if (raceIndex === raceMax) seekRace(0);
  $("race-play").textContent = "Pause race Ⅱ";
  $("race-play").setAttribute("aria-label", "Pause recorded race");
  raceTimer = setInterval(() => {
    seekRace(raceIndex + 1);
    if (raceIndex === raceMax) stopRace();
  }, 160);
});
restart();
