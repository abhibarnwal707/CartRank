from cartrank.serve.sysinfo import info, rss_mb


def test_sysinfo():
    assert rss_mb() > 10 and {"platform", "python", "rss_mb"} <= set(info())
