from pgx import hello


def test_helo():
    assert hello() == "hello world"
