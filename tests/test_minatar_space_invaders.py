def test_api():
    import pgx
    env = pgx.make("minatar-space_invaders")
    pgx.api_test(env, 3, use_key=True)
