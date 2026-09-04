# Outreach drafts

Three emails, ready to send. They are drafts rather than sent messages on
purpose: these go to named researchers under your name, and the wording of a
first approach to the people whose data you have just published a negative
result about is yours to own.

Read [`releasing.md`](releasing.md) step 4 first for why this is the highest-value
step. Send them **after** the DOI exists and ideally before the arXiv attempt,
since the endorsement you need for step 3 is a plausible side effect of these.

Before sending, fill in the two placeholders: `https://doi.org/10.5281/zenodo.22215142` and, if you have
posted it, `<ARXIV LINK>`. Delete the arXiv line if you have not.

---

## 1. Geert Verdoolaege (the HDB5 maintainer)

Lead author of the 2021 *Nucl. Fusion* paper describing the database, and the
person most likely to care that a validation-methodology result was computed on
it. Find the current address on the Ghent University department page rather than
from an older paper.

**Subject:** `Model ranking on HDB5 inverts under leave-one-tokamak-out`

> Dear Professor Verdoolaege,
>
> I have been working with the ITPA global H-mode database (STD5, v5.2.3) and
> found something about model validation that I think is directly relevant to
> the group that maintains it.
>
> Under grouped cross-validation by discharge, a random forest beats IPB98(y,2)
> by 41% in RMSLE. Under leave-one-tokamak-out on the same features and the same
> models, the ranking inverts: the forest is worse than a log-linear power law on
> 13 of 13 machines, paired gap +0.251 [+0.157, +0.342]. The mechanism is
> measurable rather than a matter of tuning. Per-machine error correlates with
> distance from the training data at rho = +0.85 for the forest and -0.06 for the
> power law, and when JET is held out, 48% of its rows lie above the highest
> confinement time in the remaining machines, which a tree ensemble cannot output
> at all.
>
> The result I would most value your view on is the positive one. Imposing the
> Connor-Taylor constraints as linear equality constraints on the exponents gives
> 0.183 RMSLE at a size cut matched to ITER's 1.82x jump, which beats every
> blind model I built and also beats the analytic law fitted with those machines
> included. It has no hyperparameter. As a check on the derivation, IPB98(y,2)
> itself lands on the Kadomtsev surface at a distance of 0.00096.
>
> Everything regenerates from your published OSF file, which is pinned by SHA-256
> and verified on load rather than redistributed, so the numbers are tied to
> specific bytes:
>
> https://doi.org/10.5281/zenodo.22215142
> https://github.com/lsalcedo100/FusionFlux
>
> I would be glad of any correction, particularly on the physics.
>
> With thanks,
> Liam Salcedo

---

## 2. The ITPA Confinement Database and Modelling Topical Group

The body that would actually act on a validation-methodology finding. Route it
through the current chair, listed on the ITPA pages; if you cannot find a chair
address, Verdoolaege is the better single point of contact and this one can wait
for his reply.

**Subject:** `A validation-methodology result on the ITPA H-mode database`

> Dear <name>,
>
> I am writing about a result on the ITPA global H-mode confinement database
> that concerns how confinement models are compared rather than which model
> wins.
>
> Cross-validation grouped by discharge leaves every machine in the training
> fold, so it measures interpolation within known devices. Scored that way, a
> random forest beats IPB98(y,2) by 41%. Scored by holding out an entire
> tokamak, the same forest loses to a plain log-linear power law on 13 of 13
> machines. The standard comparison does not merely overstate the gain; it
> reverses the ordering, and it does so for a reason that is visible in advance
> from the held-out machine's distance from the training data.
>
> Two things make me think this is worth the group's attention rather than just
> mine. It reproduces on 5358 rows the standard set does not contain and in a
> different confinement regime against a different published law. And it points
> at a cheap fix: constraining the exponents to satisfy Connor-Taylor gives the
> best blind score I obtained at an ITER-matched size cut, with nothing to tune.
>
> The work is independent and not affiliated with any laboratory. Everything
> regenerates from the published OSF file:
>
> https://doi.org/10.5281/zenodo.22215142
> https://github.com/lsalcedo100/FusionFlux
>
> I would welcome correction, and would be glad to present it if that is useful.
>
> Yours sincerely,
> Liam Salcedo

---

## 3. An arXiv endorsement request

Only if step 3 of the release checklist needs one, which for a first submission
to `physics.plasm-ph` without an institutional address it will. Ask someone who
has already replied to you: an endorsement request to a stranger is far weaker
than one to a person who has read the work.

**Subject:** `arXiv endorsement for physics.plasm-ph`

> Dear <name>,
>
> Thank you for your reply about the HDB5 validation result. arXiv requires an
> endorsement for a first submission to physics.plasm-ph, and I am writing to
> ask whether you would be willing to provide one.
>
> The paper is the work you have seen: the ranking inversion under
> leave-one-tokamak-out, its three mechanisms, and the dimensional-analysis
> constraint that repairs it. It is a negative result about validation
> methodology with a positive fix, and it has no institutional affiliation
> behind it, which is why I need to ask.
>
> The endorsement code is <CODE> and the link is <LINK>. If you would rather
> not, I completely understand, and I am grateful for the time you have already
> given it.
>
> With thanks,
> Liam Salcedo

---

## What not to do

- **Do not attach the PDF.** A link is enough and an attachment from an unknown
  sender is a spam filter's favourite thing.
- **Do not lead with the request.** Every one of these leads with the finding.
  The ask, where there is one, is at the end.
- **Do not send all three at once.** The endorsement request is only strong
  after someone has replied.
- **Do not overstate it.** The reversal is a property of these models on this
  database and its replication is not independent in provenance; the paper's
  limitations section says so, and the email should not say more than the paper.
