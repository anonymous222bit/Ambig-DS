# Third-Party Licenses

This repository builds on, vendors, or references third-party software and
data. Their original licenses and notices are reproduced below and apply to
the corresponding portions of this codebase and to the Hugging Face
datasets (`anonymous222bit/Ambig-DS-M`, `anonymous222bit/Ambig-DS-T`)
that include any of these components.

---

## DSBench

Source: <https://github.com/LiqiangJing/DSBench>
Used in: target-ambiguity benchmark construction (`create_datasets/ambig_ds_target/pipeline_DSBench/`),
target-ambiguity evaluator (`evaluate/ambig_ds_target/`, including per-task
`eval.py` files redistributed in the `Ambig-DS-T` HF dataset), and the
50 base tabular tasks underlying Ambig-DS-T.

License (code): MIT.

```
MIT License

Copyright (c) 2024 Liqiang Jing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

DSBench's repository additionally states:

> **NOTE:** This license applies to the code in this repository, but not the
> external datasets and files that may be downloaded while using this
> package.

DSBench's data disclaimer (paraphrased from the upstream repository):

- Strictly for **non-commercial** educational and research use.
- No guarantees of accuracy, completeness, or timeliness.
- Users must comply with applicable privacy and data-protection laws.
- DSBench claims no ownership of the original data; original creators'
  rights are preserved.

We propagate the **non-commercial** restriction to the prompts and
manifests we redistribute that are derived from DSBench tasks.

---

## MLE-bench

Source: <https://github.com/openai/mle-bench>
Used in: metric-ambiguity benchmark construction
(`create_datasets/ambig_ds_metric/pipeline/`), metric-ambiguity evaluator
(`evaluate/ambig_ds_metric/`, which depends on `mlebench` as a Python
package; per-competition `description.md` and `grader.py` files are
referenced by the `Ambig-DS-M` HF dataset).

License (code): MIT.

```
MIT License

Copyright (c) 2024 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

MLE-bench's repository additionally states:

> **NOTE:** This license applies to the code in this repository, but not the
> external datasets and files that may be downloaded while using this
> package.

---

## Underlying Kaggle competition data

Both Ambig-DS-M and Ambig-DS-T derive from public Kaggle competitions.
**We do not redistribute the raw Kaggle competition data** (training
data, test data, labels, sample submissions, or any media bundled with
the competitions). Users obtain the data themselves via the Kaggle CLI
and the upstream `mlebench prepare` / DSBench data-preparation
pipelines, after accepting each competition's rules on kaggle.com. Each
competition's original terms of use, copyright, and licensing apply.

Original task descriptions ("prompts") are reproduced in our HF datasets
in unmodified or minimally edited form for reproducibility, following
the precedent set by MLE-bench and DSBench. Users redistributing these
prompts onward must respect the upstream non-commercial constraint
(DSBench) and the original Kaggle competition terms.
