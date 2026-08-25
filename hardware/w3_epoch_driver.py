#!/usr/bin/env python3
"""W3-3 dense-cadence driver. Amendment 1d, 25 Aug 2026, DEVIATION LOGGED.

The registered cadence was four epochs on day one at ~3 h spacing. The
plain-arm anchors of W3-1 / W3-1c / W3-1b measured the round's rotation ramping
at ~0.45 deg/min (7.83 -> 14.51 -> 19.07 deg over 30 min, one recalibration at
11:40:34 UTC and continued drift after it). A 3 h cadence cannot resolve a
30 min ramp, so epochs are chained back to back instead: each is submitted as
soon as the previous returns, giving ~10-12 min sampling. Author-approved
deviation, recorded here and in the ledger.

Arm sequence 4 x both + 5 x plain = 279 s, projected total 521 s of 600.
Every submission still passes through rung8's CAP_TOTAL guard.
"""
import sys, time, json, datetime
sys.path.insert(0, ".")
import rung8_epoch as R8
import submit_core as S

ARMS = ["plain", "both", "plain", "both", "plain", "both", "plain", "plain"]
LOG = "w3_epoch_driver.log"


def log(m):
    line = "[%s] %s" % (datetime.datetime.now(datetime.timezone.utc)
                        .strftime("%H:%M:%S"), m)
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def wait_last(timeout_s=1500):
    """Block until the newest unfetched epoch's job is terminal."""
    led = R8.load_ledger()
    pend = [e for e in led["epochs"] if not e.get("fetched")]
    if not pend:
        return True
    jid = pend[-1]["job_id"]
    svc = S.service()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = str(svc.job(jid).status())
        if any(k in st for k in ("DONE", "Completed")):
            return True
        if any(k in st for k in ("ERROR", "CANCELLED", "Failed")):
            log("job %s terminal-bad: %s" % (jid, st))
            return False
        time.sleep(20)
    log("job %s timed out waiting" % jid)
    return False


def main():
    for i, arm in enumerate(ARMS):
        if not wait_last():
            log("stopping: previous epoch did not complete cleanly"); break
        try:
            n = R8.fetch_ready()
            log("fetched %d epoch(s)" % n)
        except Exception as e:
            log("fetch error (continuing): %s" % str(e)[:160])
        u = S.service().usage()
        log("usage %s s consumed, %s s remaining; next arm=%s"
            % (u["usage_consumed_seconds"], u["usage_remaining_seconds"], arm))
        try:
            jid = R8.submit_epoch(arm=arm)
        except Exception as e:
            log("submit error: %s" % str(e)[:200]); break
        if jid is None:
            log("budget guard refused; stopping the series cleanly"); break
    wait_last()
    try:
        R8.fetch_ready()
    except Exception as e:
        log("final fetch error: %s" % str(e)[:160])
    u = S.service().usage()
    log("DRIVER DONE. usage %s s consumed, %s s remaining"
        % (u["usage_consumed_seconds"], u["usage_remaining_seconds"]))
    open("W3_EPOCHS_DONE", "w").write("done\n")


if __name__ == "__main__":
    main()
