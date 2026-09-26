# SoftwareX manuscript — author-review draft

The draft follows the **Original Software Publication**, not software-update,
template retrieved from SoftwareX on **26 September 2026**. The paper documents
software **0.3.0**, frozen at
[b3d2076689725777b4581e36a5c9ce7d12c507ae](https://github.com/AlexM3020/accessible-routing-openstreetmap/tree/b3d2076689725777b4581e36a5c9ce7d12c507ae).
Later commits adding manuscript files do not change that scientific reference.

**This is not submission-ready until the author completes the items below.**
No Erlangen results, ethics approvals, funding status, conflicts, code license,
archival DOI, adoption statistics or working hosted-routing service have been invented.

## Contents

- [main.tex](main.tex): complete five-section software paper, title/affiliation,
  abstract, C1–C8 metadata, declarations and bibliography integration.
- [erlangen.tex](erlangen.tex): clearly labelled **evaluation plan and placeholders**,
  not a description of completed testing. Replace with what was actually done.
- [references.bib](references.bib): ten references checked against Crossref or
  official project documentation; Neis uses the 2015 print issue, pp. 188–201.
- [highlights.txt](highlights.txt): five separate submission highlights.
- [generate_example.py](generate_example.py): reproducible synthetic figure/table
  generation, guarded by the documented implementation fingerprint.
- [figures/synthetic-comparison.pdf](figures/synthetic-comparison.pdf): compact
  vector figure generated from the actual software fixture, not an image model.
- [figures/example-table.tex](figures/example-table.tex): numeric LaTeX table
  generated from the same run.
- [figures/example-provenance.json](figures/example-provenance.json): profiles,
  parameters, exact traces, comparisons, versions and figure/table checksums.

The manuscript and its artifacts are separate from the application's five-file
result contract. No routing, scoring, data or website code is changed by this paper.
The root README's deleted MongoDB instructions remain deleted.

## Build

The included figure and table allow compilation without running Python or
accessing any map/database service. Use TeX Live, MiKTeX or Overleaf with
`elsarticle`, BibTeX and the standard packages declared by the manuscript.
The official template's `preprint,12pt,a4paper` settings are retained; there are
no custom page margins or compressed fonts to evade length requirements.

From this folder, with `latexmk` installed:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Alternatively, Tectonic handles the bibliography and required repeat passes:

```powershell
tectonic --untrusted --keep-logs --keep-intermediates --outdir build main.tex
```

Create the build directory first if your tool requires it. Tectonic may download
its standard TeX bundle on first use. The compiled PDF and auxiliary files stay
in the ignored build directory. A portable Tectonic compiler was used for local
verification; no compiler binaries or third-party class files are committed.

For Overleaf, upload the contents of this folder, preserving the figure subfolder,
and select [main.tex](main.tex) as the main document. A PDF preview is not a
replacement for the editable source required by the journal.

## Regenerate the synthetic evidence

From the **repository root**, using the research environment with project dependencies:

```powershell
python -m paper.generate_example --output paper/build/regenerated
```

The destination must not contain previously generated target files. The generator
never overwrites them silently. It verifies software version 0.3.0 and the full
implementation fingerprint, runs the built-in 12-node/30-directed-edge fixture,
validates the saved experiment, and derives the figure, numeric table and provenance.
No OSM download, MongoDB connection, live demo, field study or participant information
is used. The manuscript figure uses a declared 60 m display-only corridor to fit
the journal page; application routing and the default 150 m map radius are unaffected.

Numerical values and geometry should agree when the software is unchanged.
Timing fields and environment records can differ. Bit-identical rendering across
different Matplotlib/PROJ/GEOS/font versions is not promised. Keep the dependency
versions recorded with the evidence when reproducing it. Changing the reference
software requires reviewing the manuscript, generator guard and numerical statements,
not just removing the guard to force the script to run.

## Author actions before submission

1. **Erlangen evidence:** replace [erlangen.tex](erlangen.tex) with the actual study.
   Supply extraction date/bounds/checksum, graph size, route selection, successful
   and failed requests, profiles, audit methodology, paired outcomes, repeated
   timings, hardware and uncertainty. Distinguish mapped from independently
   observed features. Include null or unfavourable results. Update the abstract,
   impact and conclusion only as warranted by those results.
2. **Ethics and privacy:** if people, personal mobility information or identifiable
   locations are involved, verify the applicable approval/exemption, consent and
   sharing requirements. Do not infer an exemption from the fact that OSM is public.
3. **Code license:** none is declared at the reference revision. SoftwareX requires
   an open-source distribution; the official template specifically calls for a
   license file alongside the README. Choose a license you have authority to apply,
   add its actual text, and complete C3. A placeholder file is not a license.
   This task has **not** licensed the code or manuscript on the author's behalf.
  After adding the approved license, update the permanent code link to the
  licensed release/commit as well; a link to an older unlicensed snapshot is
  not repaired simply by changing the label in C3.
4. **Author details:** confirm sole authorship, the interpretation of “M.Sc. Data
   Science” as the programme/affiliation, the institution where the work was done,
   and the full appropriate postal address. No department, street address or ORCID
   has been guessed. The supplied name and email are Alex Martinelli and
   alex.martinelli@fau.de.
5. **Declarations:** approve CRediT roles, funding, competing interests and any
   acknowledgements. Employment and other relationships may need disclosure;
   nothing has been inferred from account names or email addresses.
6. **AI disclosure:** retain and review the factual GitHub Copilot disclosure.
   Check every scientific claim and reference, edit the draft into the author's
   own final work, and confirm responsibility before submission. The draft does
   not falsely state that this review has already happened.
7. **Companion demo:** the link is
   [Roam](https://roam-um3s.onrender.com/). It is a separate deployed application,
   **not** the frozen scientific package. On 26 September 2026 the frontend was
   reachable, but its backend returned HTTP 503; a working route calculation was
   not verified. Check or repair availability before advertising a working demo.
   No suspended service was resumed and no paid compute settings were changed.
8. **Final source and preservation:** confirm the version-specific GitHub link,
   keep the documented version available, and consider a release archive/DOI after
   choosing the license. Do not invent an ElsevierSoftwareX fork URL or DOI before
   one exists. The journal archives accepted code separately.
9. **Remove draft queries:** search for `AuthorQuery`, `Pending`, and “to be completed”.
   Replace the content, rather than hiding unresolved items with a style switch.
   Do a fresh word/figure count after adding the Erlangen results.

## Journal-specific checks

Guidance checked on 26 September 2026:

- [SoftwareX Guide for Authors](https://www.sciencedirect.com/journal/softwarex/publish/guide-for-authors).
- [Official Original Software Publication LaTeX template](https://legacyfileshare.elsevier.com/promis_misc/softwarex-osp-template.tex).
- The current template requires five sections: **Motivation and significance;
  Software description; Illustrative examples; Impact; Conclusions**.
- It specifies **4,000 words**, a six-page main-text target excluding metadata,
  tables, figures and references (word limit takes priority), and at most **six figures**.
  Keep this distinction when judging the longer preprint PDF, which also includes
  metadata, tables, a figure, pending-author notes, declarations and references.
- The template suggests an abstract of about 100 words; the general guide permits
  at most 250. This draft uses a short, citation-free abstract.
- The metadata table retains the current template's C1–C8 descriptions and a
  version-specific **GitHub** code link. It does not substitute an unrelated repository.
- Highlights are a separate editable file: three to five bullets, no more than
  85 characters per bullet including spaces.
- References use numbered citations in order of appearance and DOI links where available.
- Complete the journal's declarations workflow separately at submission. A LaTeX
  paragraph alone may not replace the required declaration-form upload.
- Recheck the live guide before submitting: journal requirements can change.

The paper makes no claim of measured navigation safety, novel A*, third-party
adoption or empirical accessibility improvements before the Erlangen data are added.
Statements about the software and synthetic example are complete; the remaining
queries concern facts or decisions only the author can supply.

## Local verification

- The software suite was rerun during manuscript preparation: **343 passed,
  1 Windows symlink-permission skip**, approximately **93% combined statement/
  branch-aware coverage**. The paper does not describe this as branch coverage alone.
- The reference implementation and synthetic values were checked by the generator;
  its figure is vector artwork with embedded fonts and no raster images.
- The LaTeX document and numbered bibliography compile with **Tectonic 0.17.0**.
  Citation and cross-reference resolution, the 250-word abstract ceiling and
  the 85-character highlight limits were checked. The abstract has **97 words**;
  the longest highlight has **78 characters** excluding its bullet marker.
- A conservative extraction of the main-section span was below **4,000 words**,
  even including intervening table/caption/formula text and the Erlangen scaffold.
  This is an approximate PDF-text check, not a replacement for the journal's final
  word count after you insert results.
- PDF title/metadata pages and the synthetic illustration were visually checked.
  The retained Elsevier class emits a small (2.6 pt) first-page output-box warning
  with this engine; no clipped content was observed. No unresolved citations or
  substantive manuscript text-overflow warnings remain.
- The compiled manuscript is a local preview only and is not committed. Source,
  bibliography, highlights and the required figure/table/provenance assets are included.