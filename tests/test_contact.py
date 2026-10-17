"""Contact guard: smooth gravity drift is not contact; a jump in J2/J3/J5 is."""

import asyncio

from worldarm.contact import ContactGuard


def run(zs, loads_at):
    seq = iter(loads_at(z) for z in zs)

    async def read():
        return next(seq)

    guard = ContactGuard(read, settle_s=0.0)

    async def go():
        hits = []
        for z in zs:
            hits.append(await guard(z))
        return hits

    return asyncio.run(go()), guard


ZS = [200, 175, 150, 125, 100, 97, 94, 91, 88, 85]


def drift(z):  # loads change smoothly with height, as gravity does
    return [0.0, -4.3 + 0.004 * (200 - z), -15.4 - 0.006 * (200 - z), 0.0, -1.9 + 0.002 * (200 - z), 0.0]


def test_smooth_drift_is_not_contact():
    hits, guard = run(ZS, drift)
    assert not any(hits) and guard.contact_z is None


def test_a_push_on_j3_is_contact_at_that_height():
    def touched(z):
        loads = drift(z)
        if z <= 91:
            loads[2] += 2.0  # fingers on the table
        return loads
    hits, guard = run(ZS, touched)
    assert hits.index(True) == ZS.index(91) and guard.contact_z == 91


def test_j6_wobble_is_ignored():
    def wobble(z):
        loads = drift(z)
        loads[5] += 5.0 if z < 120 else 0.0
        return loads
    hits, _ = run(ZS, wobble)
    assert not any(hits)
