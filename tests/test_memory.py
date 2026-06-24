from fairmix_audit.memory import _rss_mb_from_proc_status_text, rss_mb, trim_process_memory


def test_proc_status_rss_parser_reads_current_linux_rss():
    text = "\n".join(
        [
            "Name:\tpython",
            "VmSize:\t  123456 kB",
            "VmRSS:\t    2048 kB",
            "Threads:\t1",
        ]
    )

    assert _rss_mb_from_proc_status_text(text) == 2.0


def test_proc_status_rss_parser_handles_missing_or_invalid_values():
    assert _rss_mb_from_proc_status_text("Name:\tpython\n") is None
    assert _rss_mb_from_proc_status_text("VmRSS:\tnot-a-number kB\n") is None


def test_memory_helpers_are_safe_on_current_platform():
    assert rss_mb() > 0
    assert isinstance(trim_process_memory(), bool)
