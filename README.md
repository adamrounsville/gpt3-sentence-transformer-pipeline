# GPT3 Sentence Transformer Pipeline
A Python pipeline to generate responses for a dataset of 250 datapoints of type `gender, age, ethnicity` using GPT3 (`text-davinci-003`), map them to a 768 dimensional dense vector space using the [T5 XXL](https://huggingface.co/sentence-transformers/sentence-t5-xxl) sentence transformer, use [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) and [UMAP](https://umap-learn.readthedocs.io/en/latest/) dimensionality-reduction methods to reduce the dimensionality of the data set, and then provide visualizations using [Plotly](https://github.com/plotly/plotly.py) and sentiment analysis using [TextBlob](https://textblob.readthedocs.io/en/dev/]).

## Instructions 
#### Setup
1) Add OpenAI key to `keys.py`
2) Run `pip install matplotlib seaborn umap-learn sentence_transformers openai plotly textblob`
3) Run `python3 gen.py`, changing the `PROMPT` global variable if you want to change the dataset in `fake_data/fake_people.csv`
4) Change the `STORY_START` and `STORY_END` global variables in `stories.py` to account for what answers you want GPT3 to generate

#### Pipeline
1) Run `python3 stories.py`
2) Run `python3 strans.py` (CAUTION: This will likely use significant GPU resources while the sentence transformer is running)
3) Run `python3 vis.py` to generate Plotly and sentence sentiment analysis graphs
