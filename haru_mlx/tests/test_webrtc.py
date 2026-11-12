import pytest

webrtc = pytest.importorskip("haru_mlx.webrtc", reason='needs the optional "webrtc" extra')


def test_client_ice_servers_must_be_cloudflare_turn_credentials():
    good = [{"urls": ["turn:turn.cloudflare.com:3478?transport=udp"], "username": "u", "credential": "c"}]
    assert webrtc.parse_client_ice_servers(good) == good
    assert webrtc.parse_client_ice_servers([{"urls": "stun:stun.cloudflare.com:3478"}]) is None  # no relay
    assert webrtc.parse_client_ice_servers([{"urls": ["turn:example.com:3478"], "username": "u", "credential": "c"}]) is None
    assert webrtc.parse_client_ice_servers([]) is None
    assert webrtc.parse_client_ice_servers("turn:turn.cloudflare.com") is None


def test_generated_ice_servers_drop_port_53():
    payload = {"iceServers": [{"urls": ["stun:stun.cloudflare.com:53", "turn:turn.cloudflare.com:3478?transport=udp"], "username": "u", "credential": "c"}]}
    assert webrtc._normalize_ice_servers(payload) == [
        {"urls": ["turn:turn.cloudflare.com:3478?transport=udp"], "username": "u", "credential": "c"}
    ]


def test_static_ice_servers(tmp_path, monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_TURN_KEY_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_TURN_KEY_API_TOKEN", raising=False)
    assert webrtc.IceServers().has_turn is False
    path = tmp_path / "ice.json"
    path.write_text('[{"urls": ["turn:turn.example.org:3478"], "username": "u", "credential": "c"}]')
    assert webrtc.IceServers(str(path)).has_turn is True
