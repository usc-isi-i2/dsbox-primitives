## Dsbox Primitive Unit Test
#### Test pipeline
How To add more testing pipeline (e.g. for new primitive):
1. In `library.py`, following the existed format (like `DefaultClassificationTemplate`) to create a new `Template` class.
2. If the added new pipeline is for new `runType` (you can also check it to ensure). Add the corresponding mapping In dict `DATASET_MAPPER` at line 416 on file `template.py`, follow the format. Ensure it is correct, otherwise the system would failed on finding correct dataset to run.
3. ~~Go to `generate-pipelines-json.py` and add the new class for the import (line 8) part.~~ The system should now import all templates.
4. Add it to `TEMPLATE_LIST` on `generate-pipelines-json.py`.
5. Then, the unit test system will automatically run the new template and generate corresponding `pipeline.json` file that can used to upload as sample pipeline.

#### primitives that do not have pipelines now
1. data preprocessing: `label_encoder`, `greedy_imputation`, `multitable_featurization`
2. `column_fold` and `unfold`
3. Video classification: `LSTM`, `inceptionV3`, 
4. concat related: `horizontal concat`,
5. Dataset splitter: `splitter`
