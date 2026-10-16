from mobile_world.runtime.utils.e2e import (
    build_agent_env,
    extract_host_port,
    score_success,
)


def test_extract_host_port():
    ports = {
        "6800/tcp": [{"HostPort": "6800"}],
        "5556/tcp": [{"HostPort": "5556"}],
    }
    assert extract_host_port(ports, "6800") == "6800"
    assert extract_host_port(ports, "5556") == "5556"
    assert extract_host_port(ports, "9999") is None


def test_score_success():
    assert score_success(1.0) is True
    assert score_success(0.0) is False
    assert score_success(0.5) is False


def test_build_agent_env_sets_serial_and_keeps_keys():
    env = build_agent_env(5556, overrides={"ANTHROPIC_API_KEY": "k1", "MIDSCENE_MODEL_API_KEY": "k2"})
    assert env["ANDROID_SERIAL"] == "localhost:5556"
    assert env["ANTHROPIC_API_KEY"] == "k1"
    assert env["MIDSCENE_MODEL_API_KEY"] == "k2"
    assert "MIDSCENE_MODEL_NAME" in env
