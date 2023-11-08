def test_api():
    import pgx
    env = pgx.make("minatar-breakout")
    pgx.api_test(env, 3, use_key=True)
