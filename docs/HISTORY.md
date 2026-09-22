# The program this reconstructs

## ECHO 1

In 1975 **Denys Overholser**, a radar specialist and mathematician at Lockheed's
Skunk Works, wrote a computer program he called **ECHO 1**. It predicted the
radar cross section of a shape — the first time that could be done to useful
accuracy — with one restriction that turned out to matter more than anything
else about it. Lockheed's own account of the period:

> The heart of the proposal was a computer program dubbed "Echo 1," which
> allowed its aircraft designers to predict a radar return. […] For the first
> time, we were able to develop a computer program to precisely predict the
> radar cross section (RCS) return of any given shape — provided it could be
> made from flat panels. In 1975, DARPA was persuaded to accept an unsolicited
> Lockheed proposal based on this groundbreaking software.

## Where the physics came from

ECHO 1 rested on a paper nobody in the West had thought to apply to aircraft:
**Pyotr Ufimtsev**'s 1962 monograph *Method of Edge Waves in the Physical Theory
of Diffraction*, published in Moscow and translated by the USAF Foreign
Technology Division in 1971. Ufimtsev had worked out how to compute the field
scattered from an *edge* — the correction that physical optics, which only knows
about surfaces, leaves out.

That correction is the difference between a code that works and one that does
not. Physical optics alone predicts enormous returns where a surface happens to
face the radar, and almost nothing in between; it gives a designer no way to
tell a genuinely quiet aspect from an artefact of the approximation. The
comparison in `validation/` puts a number on it: on a flat plate at 70–86° off
normal, physical optics alone is **14 dB low**, and the edge waves are the
missing 14 dB.

## Why the F-117 looks like that

ECHO 1 could only analyse shapes made of flat panels, because the closed-form
scattering integral it used only exists for a flat polygon. So the aircraft was
built to suit the code rather than the other way round: the "Hopeless Diamond",
then Have Blue, then the F-117A Nighthawk — faceted, because a facet was what
could be computed.

The design rules that fall out of a code like this are visible in the two
aircraft shapes in `echo1.shapes`, and you can reproduce them:

**Tilt every panel.** A flat panel returns `4 pi A^2 / lambda^2` when the radar
looks down its normal, and that is a colossal number — a 30 cm square panel at
10 GHz gives +20 dBsm, roughly the RCS of a truck. Tilt it and the return
collapses. So no surface faces any direction a radar is likely to be.

**Then align the edges.** Once every panel is tilted, the edges dominate, and
edges cannot be tilted away — each one throws its energy into a fan
perpendicular to itself, and some direction always catches it. What you *can*
do is make all the edges parallel to a small number of directions, so the fans
pile up into a few narrow spikes with quiet sectors between them. That is
planform alignment, and it is why the F-117's wing, tail, intakes and even its
door edges run along the same handful of lines.

`echo1.shapes.faceted_delta` is built to that rule — every planform edge lies
along one of two directions — and `examples/signatures.py` plots it against the
un-aligned hopeless diamond.

## What this package is and is not

This is **not** Lockheed's code, which has never been public, and it makes no
claim to reproduce its numbers. It is an independent implementation of the
physics ECHO 1 was built on, from the open literature, checked against
closed-form results and an independent numerical solver.

Two things are worth keeping in perspective. ECHO 1 ran in 1975 on a mainframe,
took hours per shape, and was essentially two-dimensional — the three-dimensional
picture was assembled from plane cuts. A 1440-aspect sweep here takes a second
on a laptop. And a modern signature problem is at least as much about materials,
apertures, seams and inlet ducts as about shape; this models shape alone.

The historical claims above are from Lockheed Martin's published account and
Air & Space Forces Magazine, both linked from the README.

## Designing one, rather than analysing one

`examples/faceted_fighter.py` closes the loop the other way round. Everything
above is about taking a shape and finding its signature; the reason Lockheed
wanted the code was the reverse — to state a shaping rule, build to it, and
find out at the drawing stage whether the rule had actually been obeyed.

Two predictors answer in closed form, with no solve at all:
`shapes.spike_azimuths` gives the azimuths the edges will throw into, and
`shapes.specular_aspects` gives the aspect each flat panel mirrors at and how
loud it is there. Both are elementary — an edge radiates perpendicular to
itself, a panel returns `4πA²/λ²` along its own normal — and both are things a
designer can check against a drawing in a minute. The solver's job is then to
disagree with them, and every disagreement in this exercise turned out to be a
fault in the geometry rather than in the prediction:

- a skin fanned from a peak the planform could not see all of had folded
  through itself, leaving right-angle re-entrant corners nobody drew;
- fins swept the conventional way added two edge directions to a planform that
  had been carefully built from two;
- fins given thickness the obvious way left a 12 cm flat strip round the rim,
  and a 12 cm strip at 10 GHz is four wavelengths of mirror;
- and a fin's cant angle turned out to *be* the elevation its remaining
  mirror points at, which is the whole of why stealth aircraft cant their fins
  rather than shrink them.

That last one is the clearest statement of what shaping is. Canting a fin from
vertical to 20 degrees does not make its 45 dBsm specular smaller. It moves it
16 degrees below the horizon, where nothing is standing.
