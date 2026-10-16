from streamlit.testing.v1 import AppTest


def run_app():
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_app_starts_and_shows_context():
    at = run_app()
    assert "AI-детектив" in at.title[0].value
    assert len(at.tabs) == 4


def test_app_modes_and_stress_controls():
    at = run_app()
    for mode in ["A", "B", "C"]:
        at.sidebar.radio[0].set_value(mode).run()
        assert not at.exception
    at.sidebar.checkbox[0].check().run()
    at.sidebar.selectbox[1].set_value("C05").run()
    at.sidebar.selectbox[2].set_value("X-Q03").run()
    assert not at.exception
    assert any("Вариант досье" in i.value for i in at.info)
