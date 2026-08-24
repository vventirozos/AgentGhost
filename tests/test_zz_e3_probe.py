def test_which_module_wins():
    import src.ghost_agent.tools.execute as viasrc
    import ghost_agent.tools.execute as viapkg
    print("\n  src. form  ->", viasrc.__file__)
    print("  pkg form   ->", viapkg.__file__)
