## LLM Finetuning Q&A - Cybersecurity, IT Support Dataset

A comprehensive, high-quality dataset of 11,000 question-and-answer (Q&A) pairs designed to fine-tune large language models (LLMs) for specialised domains like cybersecurity, IT support, and cloud computing. This dataset is ideal for researchers and developers working on building sophisticated and accurate conversational AI and question-answering systems.

The dataset is meticulously structured with two primary columns: "Question" and "Answer." It addresses a wide array of topics, from fundamental cybersecurity concepts like "What is a VPN and why is it used?" to more technical inquiries such as "What is the difference between IDS and IPS?". The content is designed to be a valuable resource for training LLMs to comprehend and generate nuanced and contextually relevant responses in highly specialised fields.

The creation of this dataset was a multi-faceted process. Initially, real-world data was gathered from approximately 7,000 customer support tickets from METCLOUD, a UK-based cybersecurity and cloud computing services company. These tickets, logged via phone and email, captured authentic client interactions. A team of cybersecurity experts and IT specialists then manually pre-processed this raw data, transforming unstructured support tickets into standardised question-and-answer pairs. This involved a rigorous cleaning process to remove irrelevant information, correct errors, and standardise technical terms.

To enrich the dataset, domain-specific knowledge was extracted from authoritative online sources, including documentation from Microsoft, Hewlett-Packard Enterprise (HPE), MITRE ATT&CK, and Cisco. Furthermore, synthetic data was generated using state-of-the-art LLMs like OpenAI's GPT-4o, guided by a systematically created topic tree to ensure comprehensive coverage of subtopics. This synthetic data generation was crucial in expanding the dataset while maintaining relevance and mitigating privacy concerns.

Before its final compilation, the entire dataset underwent a thorough quality assurance review by cybersecurity engineers and IT specialists. This expert-led validation ensures the technical accuracy and alignment of the answers with current industry best practices.
The dataset is provided in a user-friendly format, making it readily available for integration into various model training pipelines. This public release aims to foster advancements in the development of domain-specific AI applications.

### File Information

- File Name: `dataset.xlsx`
- Format: Excel (.xlsx)
- Total Entries: 11,000
- Columns: Question, Answer

### Dataset availability

The dataset is available on IEEE DataPort DOI: https://doi.org/10.21227/3gnj-b288

## Citation
If you use this dataset in your project, please cite it as follows:

```bibtex
@data{3gnj-b288-25,
  doi       = {10.21227/3gnj-b288},
  url       = {https://dx.doi.org/10.21227/3gnj-b288},
  author    = {Chaithanya Vamshi Sai and Nouh Sabri Elmitwally and Iain Rice and Haitham Mahmoud and Ian Vickers and Xavier Schmoor},
  publisher = {IEEE Dataport},
  title     = {LLM Finetuning Q\&A - Cybersecurity, IT Support Dataset},
  year      = {2025}
}
