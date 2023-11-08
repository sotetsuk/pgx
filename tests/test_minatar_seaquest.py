def test_api():
    import pgx
    env = pgx.make("minatar-seaquest")
    pgx.api_test(env, 3, use_key=True)
