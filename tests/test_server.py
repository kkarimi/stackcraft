from fastapi.testclient import TestClient

from stackcraft.server import create_app


def test_game_action_replay_round_trip():
    with TestClient(create_app()) as client:
        first = client.post("/api/games", json={"seed": 42}).json()
        other = client.post("/api/games", json={"seed": 42}).json()
        assert first["board"] == other["board"]
        assert first["current"] == other["current"]
        assert "seed" not in first
        game_id = first["id"]
        result = client.post(
            f"/api/games/{game_id}/moves",
            json={"action_id": first["legal_actions"][0]["id"], "expected_pieces": 0},
        )
        assert result.status_code == 200
        assert result.json()["pieces"] == 1
        assert client.get(f"/api/games/{other['id']}").json()["pieces"] == 0
        replay = client.get(f"/api/games/{game_id}/replay").json()
        restored = client.post("/api/replays", json=replay)
        assert restored.status_code == 200
        frames = restored.json()["frames"]
        assert len(frames) == 2
        assert frames[-1] == {k: v for k, v in result.json().items() if k != "id"}


def test_invalid_move_preserves_game_and_unknown_sessions_fail():
    with TestClient(create_app()) as client:
        original = client.post("/api/games", json={"seed": 7}).json()
        path = f"/api/games/{original['id']}"
        assert (
            client.post(
                path + "/moves", json={"action_id": "r9x99", "expected_pieces": 0}
            ).status_code
            == 422
        )
        assert client.get(path).json() == original
        assert client.get("/api/games/missing").status_code == 404


def test_retry_after_lost_response_cannot_place_another_piece():
    with TestClient(create_app()) as client:
        state = client.post("/api/games", json={"seed": 42}).json()
        path = f"/api/games/{state['id']}"
        request = {"action_id": "r0x3", "expected_pieces": 0}
        accepted = client.post(path + "/moves", json=request)
        assert accepted.status_code == 200
        retry = client.post(path + "/moves", json=request)
        assert retry.status_code == 409
        assert client.get(path).json() == accepted.json()


def test_input_validation_and_health():
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        for seed in [-1, True, "42", 2**53]:
            assert client.post("/api/games", json={"seed": seed}).status_code == 422
        assert client.post("/api/replays", json={"actions": "not a list"}).status_code == 422
        assert client.post("/api/replays", json={"actions": ["r0x0"] * 2001}).status_code == 422


def test_packaged_race_replays_validate_against_current_engine():
    with TestClient(create_app()) as client:
        response = client.get("/static/baseline-demo.json")
        assert response.status_code == 200
        race = response.json()
        assert [player["id"] for player in race["players"]] == ["base", "trained", "heuristic"]
        for player in race["players"]:
            assert player["replay"]["seed"] == race["seed"]
            replay = client.post("/api/replays", json=player["replay"])
            assert replay.status_code == 200
            final = replay.json()["frames"][-1]
            assert final["pieces"] <= race["max_pieces"]
            assert final["score"] == player["replay"]["final"]["score"]
