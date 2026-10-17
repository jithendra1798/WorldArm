"""Contact guard: smooth gravity drift is not contact; a lasting jump in J2/J3/J5 within reach is."""

import asyncio

from worldarm.contact import ContactGuard


def run(zs, loads_at, **kw):
    """Descend through `zs`; loads_at(z, read_no) gives the loads of the read_no-th read at height z."""
    state = {"z": None, "n": 0}

    async def read():
        state["n"] += 1
        return loads_at(state["z"], state["n"])

    guard = ContactGuard(read, settle_s=0.0, **kw)

    async def go():
        hits = []
        for z in zs:
            state["z"], state["n"] = z, 0
            hits.append(await guard(z))
            if hits[-1]:
                break  # the descent stops at contact
        return hits

    return asyncio.run(go()), guard


ZS = [200, 175, 150, 125, 100, 97, 94, 91, 88, 85]


def drift(z, n=1):  # loads change smoothly with height, as gravity does
    return [0.0, -4.3 + 0.004 * (200 - z), -15.4 - 0.006 * (200 - z), 0.0, -1.9 + 0.002 * (200 - z), 0.0]


def test_smooth_drift_is_not_contact():
    hits, guard = run(ZS, drift)
    assert not any(hits) and guard.contact_z is None


def test_a_push_on_j3_is_contact_at_that_height():
    def touched(z, n):
        loads = drift(z)
        if z <= 91:
            loads[2] += 2.0  # fingers on the table
        return loads
    hits, guard = run(ZS, touched)
    assert hits.index(True) == ZS.index(91) and guard.contact_z == 91


def test_j6_wobble_is_ignored():
    def wobble(z, n):
        loads = drift(z)
        loads[5] += 5.0 if z < 120 else 0.0
        return loads
    hits, _ = run(ZS, wobble)
    assert not any(hits)


def test_ringing_after_a_step_is_not_contact_and_does_not_bend_the_trend():
    def ringing(z, n):
        loads = drift(z)
        if z == 97 and n == 1:
            loads[1] += 6.0  # first read right after the step: J2 still swinging
        if z <= 88:
            loads[2] += 2.0  # the real touch, later
        return loads
    hits, guard = run(ZS, ringing)
    assert hits.index(True) == ZS.index(88) and guard.contact_z == 88
    assert all(h[1][1] < -3.0 for h in guard.history)  # the swinging read never joined the trend


def test_nothing_is_contact_above_reach():
    def shifted(z, n):
        loads = drift(z)
        if z <= 125:
            loads[2] += 2.0  # a lasting shift up high (cable, speed change): the fingers touch nothing there
        return loads
    hits, guard = run(ZS, shifted, active_below=110)
    assert not any(hits) and guard.contact_z is None
    hits, _ = run(ZS, shifted)  # without the reach limit the same shift would stop the descent
    assert any(hits)


def test_two_readings_are_not_a_trend():
    def early(z, n):
        loads = drift(z)
        if z == 150:
            loads[2] += 2.0
        return loads
    hits, _ = run(ZS[:3], early)
    assert not any(hits)
