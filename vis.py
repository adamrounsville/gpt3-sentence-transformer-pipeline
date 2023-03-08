import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from textblob import TextBlob
from sklearn.cluster import KMeans, DBSCAN
import plotly.express as px

plt.ion()

FEATURE_TO_SORT_ON = 'ethnicity'
OUTPUT_FILE_NAMES  = {
    'plotly_graph':'plotly-graph.html',
    'plotly_graph_clusters':'plotly-graph-clusters.html',
    'plotly_graph_clusters_dbscan':'plotly-graph-clusters-dbscan.html',
    'graphs-pdf':'output-graphs.pdf'
}

df = pd.read_csv("./results_simple_multipass_text-davinci-003.csv")

# ============================= Get Sentiment Data =============================

pdf = matplotlib.backends.backend_pdf.PdfPages(f"./result_viz/{OUTPUT_FILE_NAMES['graphs-pdf']}")

# Calculate sentiment polarity
def sentiment_polarity(text, include_neutral=False):
    sentiment = TextBlob(text).sentiment.polarity

    if sentiment < 0:
        return "Negative"
    elif sentiment == 0 and include_neutral:
        return "Neutral"
    else:
        return "Positive"

stories = {}
sentiment_means = {}
ethnicities = df['ethnicity'].unique()

for e in ethnicities:
    stories[e] = None
for e in ethnicities:
    stories[e] = df['response'].loc[(df['ethnicity'] == e)]
for e in ethnicities:
    s = []
    
    for response in stories[e]:
        s.append(TextBlob(response).sentiment.polarity)

    sentiment_means[e] = np.mean(s)

# ============================= Plot Sentiment Data ============================

# Matplotlib barchart
data = sentiment_means
ind = np.arange(len(data))
fig = plt.figure()

plt.bar(ind, list(data.values()))
plt.xticks(ind, list(data.keys()))
plt.show()

pdf.savefig(fig)

# Pie charts
for ethnicity in stories.keys():
  sentiment_list = [sentiment_polarity(story) for story in stories[ethnicity]]
  sentiment_keys = ['Positive', 'Negative']

  if len(sentiment_list) == 0:
    print('Sentiment list empty: ', ethnicity)
    continue
  
  values = [sentiment_list.count('Positive') / len(sentiment_list), sentiment_list.count('Negative') / len(sentiment_list)]

  # Plotting the results as a pie chart
  fig = plt.figure()

  plt.pie(values, labels=sentiment_keys, startangle=90, counterclock=False, autopct='%1.1f%%', shadow=True)
  plt.axis('equal')
  plt.title(f'Sentiment Analysis Results: {ethnicity}')
  plt.show()

  pdf.savefig(fig)

# Save figures to pdf
pdf.close()

# ==============================================================================

sentences = []

for pid in tqdm(range(len(df))):
    person = df.iloc[pid]
    sentences.append(person['response'])

rid = 0
rids = {}
rrids = {}
rtexts = []
rcodes = []

for pid in tqdm(range(len(df))):
    person = df.iloc[pid]
    ethnicity = person[FEATURE_TO_SORT_ON]

    if type(ethnicity) != str:
        ethnicity = 'Other'

    ethnicity = ethnicity.strip().lower()
    rtexts.append(ethnicity)

    if not ethnicity in rids:
        rids[ethnicity] = rid
        rid += 1

    rcodes.append(rids[ethnicity])

rcodes = np.array(rcodes)

for k, v in rids.items():
    rrids[v] = k

projections = np.load("./simple_multipass_projections.npy")

plt.clf()

legend_entries = []

for r in range(rid):
    inds = rcodes == r
    legend_entries.append(rrids[r])

    plt.scatter(projections[inds, 0], projections[inds, 1], alpha=0.5)

plt.legend(legend_entries)

# ================================= Clustering =================================

# kmeans
def cluster_data(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    return kmeans.labels_

cluster_labels = cluster_data(projections, 5)

# dbscan
dbscan = DBSCAN(eps=0.27, min_samples=2)
dbscan.fit(projections)
dbscan_cluster_labels = dbscan.labels_

n_clusters = len(set(dbscan_cluster_labels)) - (1 if -1 in dbscan_cluster_labels else 0)
print("Clusters found: ", n_clusters)

# =================================== Plotly ===================================

# Add linebreaks
def split_string(string, parts=4):
    n = len(string)
    return [string[i * n // parts:(i + 1) * n // parts] for i in range(parts)]

s1, s2, s3, s4 = [], [], [], []

for s in sentences:
    x = split_string(s, 4)
    s1.append(x[0])
    s2.append(x[1])
    s3.append(x[2])
    s4.append(x[3])

fig = px.scatter(
    projections,
    x=0, y=1,
    color=rtexts,
    color_discrete_sequence=px.colors.qualitative.Prism,
    hover_data=[s1, s2, s3, s4]
)
fig.write_html(f"./result_viz/{OUTPUT_FILE_NAMES['plotly_graph']}")

# Color by cluster kmeans
fig = px.scatter(
    projections,
    x=0, y=1,
    color=cluster_labels,
    color_discrete_sequence=px.colors.qualitative.Prism,
    hover_name=rtexts,
    hover_data=[s1, s2, s3, s4]
)
fig.write_html(f"./result_viz/{OUTPUT_FILE_NAMES['plotly_graph_clusters']}")

# Color by cluster dbscan
fig = px.scatter(
    projections,
    x=0, y=1,
    color=dbscan_cluster_labels,
    color_discrete_sequence=px.colors.qualitative.Prism,
    hover_name=rtexts,
    hover_data=[s1, s2, s3, s4]
)
fig.write_html(f"./result_viz/{OUTPUT_FILE_NAMES['plotly_graph_clusters_dbscan']}")

print('Finished data visualization')
