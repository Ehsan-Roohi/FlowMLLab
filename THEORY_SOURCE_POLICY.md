# Theory source and originality policy

FlowMLLab may use an external course syllabus or topic list to identify broad
subjects worth teaching. It must not import restricted teaching expression or
student work into the public repository.

## Allowed inputs

- Public course catalogs and syllabus-level topic names.
- Peer-reviewed papers, standards, and books that can be cited publicly.
- Public software documentation and independently derived equations.
- FlowMLLab's own solvers, datasets, figures, tests, and retained evidence.

## Excluded inputs

- Homework, solution sets, grading material, or student submissions.
- Instructor slides, annotated handouts, recordings, transcripts, figures, or
  examples unless they have an explicit reuse license and are cited under it.
- Text, code, diagrams, exercises, or narrative structure copied or lightly
  paraphrased from a restricted source.
- Unpublished datasets or results without documented permission.

Excluded material may not be pasted into prompts, source files, notebooks,
issues, or commit messages. Seeing that a topic appears in a course does not
grant permission to reuse the course's treatment of that topic.

## Independent-authoring workflow

1. Record only the generic topic to be covered, such as Gaussian-process
   regression or posterior predictive calibration.
2. Define a FlowMLLab-specific learning question and physical example before
   reading implementation details elsewhere.
3. Derive the explanation and code from public primary or canonical sources.
4. Use FlowMLLab-owned data and create new figures, prompts, and exercises.
5. Cite the public source that supports each nontrivial method or scientific
   claim.
6. Review the finished material for distinctive phrase, example, figure, and
   exercise overlap before release.

## Provenance record for a theory extension

Every extension should state:

- its scientific question and intended learner;
- the FlowMLLab dataset or exact synthetic generator it uses;
- the public references used for equations and interpretation;
- its baseline, split, calibration metrics, and claim boundary; and
- a declaration that no restricted course text, solution, figure, or code was
  incorporated.

The provenance record is part of the scientific evidence. A topic map can
motivate an extension, but only public references and independently generated
assets can support its public content.
