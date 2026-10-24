# AI Usage and Critical Reflection

**Project:** MMA3001 – Automated Pork Packaging Quality Inspection  
**AI Tool:** OpenAI ChatGPT  
**Year:** 2026

## 1. Overview of AI Use

OpenAI ChatGPT was used as a supporting tool throughout the development, debugging, evaluation and documentation of this project.

AI assistance made a **substantial contribution** to the initial development of Python scripts, experiment planning, troubleshooting and written documentation. However, AI-generated outputs were not treated as automatically correct. They were reviewed against the project requirements, executed where applicable, and checked using recorded experimental results.

## 2. How AI Was Used

### Experimental Design and Implementation

ChatGPT assisted with designing seven YOLO11 experiments to investigate the effects of network depth, width and initial learning rate.

AI assistance was used to develop and refine parts of the following scripts:

- `src/train.py` – initial training workflow.
- `src/run_architecture_experiments.py` – architecture generation, training configurations and results summarisation.
- `src/plot_report_figures.py` – visualisation of experimental results.
- `main.py` – demonstration commands for checking project files, displaying results, running predictions and executing tests.

These scripts were reviewed and used in the student's local Python development environment.

### Debugging and Troubleshooting

ChatGPT was used to help interpret error messages, investigate configuration issues and explain aspects of YOLO11 training and inference.

Assistance included understanding the difference between COCO-pretrained YOLO11 weights and the project-trained model checkpoints, as well as explaining the influence of detection-confidence thresholds on displayed predictions.

### Automated Testing

ChatGPT assisted with drafting automated tests for experiment configurations, dataset YAML handling, generated architecture files and results summarisation.

The tests were executed using pytest in the local development environment. The recorded test run completed successfully with **7 passed and 0 failed**.

Test evidence is available in:

- [`test_architecture_experiments.py`](../tests/test_architecture_experiments.py)
- [`test_report.txt`](test_report.txt)
- [`test_report.xml`](test_report.xml)

These tests verify selected software behaviours rather than the model's real-world detection accuracy.

### Report Writing and Documentation

ChatGPT made a substantial contribution to drafting and restructuring the written project report, preparing the README, improving technical explanations and organising references.

The drafted material was compared with the project's recorded experiments and evaluation results. The student remained responsible for checking the accuracy of submitted statements and ensuring that the documentation reflected the implemented work.

## 3. Why AI Was Used

AI was used to improve development efficiency, assist with unfamiliar library functionality and provide explanations of object-detection methods.

It was particularly useful when constructing repeatable experiments, understanding evaluation metrics and preparing clear technical documentation.

AI assistance also helped reduce time spent on repetitive coding and formatting tasks, allowing more attention to be directed towards executing experiments and interpreting the results.

## 4. Verification of AI-Generated Outputs

AI-generated suggestions were checked using several methods:

1. **Configuration verification:** The seven proposed architectures were constructed in preview mode and their parameter counts inspected.
2. **Experimental execution:** The seven controlled YOLO11 configurations were trained locally using Dataset V1.
3. **Results verification:** Training logs and CSV files were examined to distinguish peak validation performance from final-epoch performance.
4. **Checkpoint evaluation:** Saved model checkpoints were loaded and evaluated to verify their detection performance.
5. **Held-out testing:** The selected baseline model was evaluated on the separate test subset.
6. **Automated software testing:** Seven pytest tests were executed successfully.
7. **Documentation review:** Reported metrics, model configurations and limitations were compared against recorded experimental outputs.

The experimental results were generated through actual model execution, not through AI-generated estimates.

## 5. Limitations and Errors Encountered

AI-generated advice was not always directly applicable to the project.

Some early suggestions involved conventional fully connected neural-network approaches that did not suit the selected YOLO11 object-detection framework. These suggestions required reconsideration and adaptation.

Another important issue was distinguishing the original COCO-pretrained `yolo11n.pt` weights from the project's trained `best.pt` checkpoints. This distinction was checked before selecting the model used in the final evaluation.

AI explanations and code suggestions could also overlook project-specific paths, existing files or experimental constraints. Consequently, outputs required practical verification rather than direct acceptance.

These experiences demonstrated the importance of checking whether AI-generated guidance is technically correct and relevant to the implemented system.

## 6. Student Responsibility and Engineering Decisions

The student remained responsible for the submitted project, including:

- Running the training and evaluation procedures.
- Reviewing experiment configurations and saved outputs.
- Checking the reported numerical results.
- Selecting the final model based on validation evidence.
- Evaluating performance on held-out test images.
- Interpreting limitations and identifying future improvements.
- Reviewing and submitting the final code, report and documentation.

In particular, the selection of the baseline configuration was supported by its observed validation mAP@50–95 and computational complexity, rather than by an unverified AI recommendation.

## 7. Critical Reflection

AI assistance improved the efficiency of the development and reporting process and helped clarify concepts such as model architecture, training initialisation, precision, recall, mAP and confidence thresholds.

However, the project demonstrated that AI-generated material must be checked against actual software behaviour and experimental evidence. Successful code execution does not necessarily establish model accuracy, and an apparently reasonable suggestion may not be appropriate for a specific engineering problem.

The main lesson from using AI was the importance of combining automated assistance with independent verification, critical interpretation and responsibility for engineering decisions.

AI was therefore used as a development and learning aid, not as a substitute for experimental validation or engineering judgement.

---

*This document supplements the AI Use and Critical Reflection section of the submitted MMA3001 project report.*