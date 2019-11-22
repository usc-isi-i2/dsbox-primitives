## Dsbox Primitive Unit Test
#### Test pipeline
How To add more testing pipeline (e.g. for new primitive):
1. In `library.py`, following the existed format (like `DefaultClassificationTemplate`) to create a new `Template` class.
2. If the added new pipeline is for new task type problem(you can also check it to ensure). Add the corresponding mapping In dict `DATASET_MAPPER` at line 416 on file `template.py`, follow the format.
3. Go to `generate-pipelines-json.py` and add the new class for the import (line 8) part.
4. Add it to `TEMPLATE_LIST`.
5. Then, the unit test system will automatically run the new template and generate corresponding `pipeline.json` file that can used to upload as sample pipeline.

#### primitives that do not have pipelines now
1. `unary_encoder`
2. `column_fold` and `unfold`