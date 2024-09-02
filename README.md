# Use Topic modeling techniques to gain insights into the key themes and areas of discussion within a collection of articles related to ChatGPT and BARD
## Introduction to Topic Modeling

In our age of information overload, extracting meaningful insights from vast collections of textual data has become an essential task. Whether it's analyzing customer reviews, exploring a massive archive of news articles, or understanding social media trends, the ability to automatically identify and categorize topics within text is crucial. Topic modeling, a powerful technique in the field of natural language processing (NLP) and machine learning, addresses this need by offering a structured approach to uncovering hidden themes, patterns, and trends within textual data.

At its core, topic modeling is a statistical method used to identify abstract topics or themes that appear in a collection of documents. These documents can range from books and research papers to emails and social media posts. The fundamental concept is that each document is a mixture of multiple topics, and the goal of topic modeling is to discover those latent topics without the need for human supervision. Through this, topic modeling brings automation and efficiency to the process of understanding and organizing textual data, a task that would be overwhelmingly time-consuming and error-prone for humans alone.

One of the most well-known and widely used algorithms for topic modeling is Latent Dirichlet Allocation (LDA). LDA assumes that each document is a mix of topics, and each topic is a mix of words. The algorithm, which is based on a generative probabilistic model, uncovers the hidden structure by iteratively estimating the probability distributions of words in documents and topics. This probabilistic framework allows LDA to discover topics in a way that makes intuitive sense: each topic is a probability distribution over words, and each document is a probability distribution over topics.

Another popular technique is Non-Negative Matrix Factorization (NMF). NMF factorizes a term-document matrix into two lower-dimensional matrices, one representing topics and the other representing document-topic proportions. The main difference from LDA is that NMF enforces non-negativity on its factors, which can lead to more interpretable results, as each element represents the contribution of a term or topic to a document. NMF has found applications in various fields, from text mining to image analysis.

Beyond LDA and NMF, there are other algorithms, such as Latent Semantic Analysis (LSA) and more recent advancements like BERTopic and Gensim's Word2Vec-based methods. Each of these methods has its own strengths and weaknesses, making the choice of algorithm dependent on the specific goals and characteristics of the dataset at hand.

Topic modeling offers a multitude of applications across different domains. In academia, researchers employ topic modeling to explore vast archives of scientific literature, revealing trends and connections between various research areas. In the business world, companies use topic modeling to analyze customer feedback, social media conversations, and reviews to gain insights into consumer sentiment and emerging trends. In journalism, it helps in categorizing and summarizing large volumes of news articles quickly, enabling reporters to focus on stories that matter most. Governments and public institutions use topic modeling to sift through massive text corpora for information retrieval, policy analysis, and regulatory compliance. Even in healthcare, topic modeling is applied to clinical records, enabling medical professionals to detect patterns in patient data.

With the increasing prevalence of topic modeling, various software libraries and tools have emerged to simplify the process. Popular libraries such as Gensim, scikit-learn, and the Natural Language Toolkit (NLTK) in Python provide accessible interfaces for applying topic modeling techniques. This accessibility has made it easier for data scientists, researchers, and analysts to harness the power of topic modeling.

As we delve deeper into this topic, we will explore the key algorithms and their applications, the challenges, and limitations of topic modeling, as well as practical tips for implementing it effectively. Topic modeling, as an indispensable tool for organizing and making sense of textual data, continues to shape the way we interact with information, offering valuable insights that can drive innovation, improve decision-making, and enhance our understanding of the complex world of text.

## **Literature Review**

Literature review of "Smart literature review: a practical topic modelling approach to exploratory literature review" by Claus Ballegaard
Ballegaard's article (2019) proposes a framework for using topic modelling to conduct an exploratory literature review. Topic modelling is a machine learning technique that can be used to identify hidden patterns in text data. In the context of literature reviews, topic modelling can be used to identify the main topics that are being discussed in a collection of papers.

Ballegaard's framework consists of three steps:

**Pre-processing:** This step involves preparing the data for topic modelling. This includes tasks such as removing stop words, stemming or lemmatizing the words, and creating a document-term matrix.

**Topic modelling:** This step involves running a topic modelling algorithm on the document-term matrix. Ballegaard recommends using the Latent Dirichlet Allocation (LDA) algorithm.

**Post-processing:** This step involves interpreting the results of the topic modelling algorithm. This includes tasks such as identifying the most important topics, identifying relationships between topics, and identifying influential papers.

Ballegaard applies his framework to a case study of a literature review on the topic of supply chain management. He finds that topic modelling was able to help him to identify the main topics that are being discussed in the supply chain management literature, as well as the relationships between these topics. He also finds that topic modelling was able to help him to identify influential papers in the field.

**Literature on topic modelling for literature reviews**

The use of topic modelling for literature reviews is a relatively new area of research. However, there is a growing body of literature that suggests that topic modelling can be a valuable tool for researchers.
For example, a study by Chen et al. (2019) found that topic modelling was able to help researchers to identify new and emerging research topics in the field of computer science. Another study by Wang et al. (2020) found that topic modelling was able to help researchers to identify the main topics that are being discussed in the field of medical research.

However, it is important to note that topic modelling is not a perfect tool. One of the main challenges with topic modelling is that it can be difficult to interpret the results of the algorithm. This is because topic modelling algorithms are statistical models, and they do not provide any information about the meaning of the topics that they identify.
Another challenge with topic modelling is that it can be sensitive to the parameters that are used. For example, the number of topics that are identified will depend on the number of topics that are specified in the algorithm.

Benefits and challenges of using topic modelling for literature reviews

**Benefits**
- Topic modelling can help researchers to identify the main topics that are being discussed in their field
- Topic modelling can help researchers to identify relationships between topics
- Topic modelling can help researchers to identify influential papers in their field
- Topic modelling can help researchers to identify gaps in the literature
- Topic modelling can help researchers to generate new research questions
Challenges
- It can be difficult to interpret the results of the topic modelling algorithm
- Topic modelling can be sensitive to the parameters that are used
- Topic modelling requires a large corpus of text data
- Topic modelling can be computationally expensive

**Conclusion**

Overall, Ballegaard's article provides a clear and concise framework for using topic modelling to conduct an exploratory literature review. The framework is easy to follow and can be applied to any field of research.

Topic modelling is a relatively new tool for literature reviews, but it has the potential to be a valuable tool for researchers. Topic modelling can help researchers to identify the main topics that are being discussed in their field, as well as the relationships between these topics. Topic modelling can also help researchers to identify influential papers in their field, identify gaps in the literature, and generate new research questions.

However, it is important to note that topic modelling is not a perfect tool. One of the main challenges with topic modelling is that it can be difficult to interpret the results of the algorithm. Another challenge with topic modelling is that it can be sensitive to the parameters that are used.

Despite these challenges, topic modelling can be a valuable tool for researchers who are conducting literature reviews.

## **Objective of Study**

The primary objective of this study is to employ topic modeling techniques to gain insights into the key themes and areas of discussion within a collection of articles related to ChatGPT and BARD. Specifically, the study aims to achieve the following objectives:

- 1. **Topic Discovery and Categorization:** The study seeks to identify latent topics within the corpus of articles and categorize them to reveal the primary subject areas, issues, and discussions surrounding ChatGPT and BARD. By employing topic modeling algorithms, we aim to uncover the underlying structures that may not be immediately apparent through manual content analysis.

- 2. **Comparative Analysis:** To provide a comparative analysis of the two distinct topics, ChatGPT and BARD, this study aims to identify the relative prominence and prevalence of topics within each dataset. By comparing the distribution of topics in articles related to ChatGPT and BARD, we aim to uncover patterns of interest and areas of emphasis in these domains.

- 3. **Trends and Evolution:** This study aims to track the evolution of topics over time, if applicable, within the articles. By examining the temporal dynamics of topics, we can identify how discussions and interests related to ChatGPT and BARD have evolved, providing insights into the changing landscape of these subjects.

- 4. **Key Insights and Implications:** The study intends to provide actionable insights by interpreting the topics discovered. Through qualitative analysis and topic interpretation, we aim to extract meaningful information, key findings, and implications related to ChatGPT and BARD, which can be of value to researchers, practitioners, and stakeholders in these domains.

- 5. **User-generated Content Analysis:** If the dataset includes user-generated content, such as comments or discussions, the study will examine the sentiment, engagement, and potential ethical considerations within these articles. This analysis will shed light on the public's perception of ChatGPT and BARD and potential issues that have arisen in these discussions.

- 6. **Data-driven Decision Support:** The results of this study will serve as a data-driven decision support tool for individuals and organizations interested in ChatGPT and BARD. By providing a structured overview of topics, we aim to assist in content organization, trend analysis, and informed decision-making.

- 7. **Recommendations for Future Research:** Based on the findings and insights, the study will conclude with recommendations for areas of future research and potential directions for those interested in exploring ChatGPT and BARD further.

In summary, the objective of this study is to apply topic modeling techniques to uncover the latent topics, themes, and discussions within articles related to ChatGPT and BARD. By achieving these objectives, the study will contribute to a deeper understanding of the content landscape surrounding these topics and offer valuable insights that can inform decision-making and future research efforts.

## **Research Design**

Using a news article scraper, 443 news articles were scraped from the internet using several keywords like ChatGPT, Bard, OpenAI, Bard AI, BERT etc.
The main objective of choosing the mentioned keywords was to target the articles which consisted of information about ChatGPT and Google Bard.

After the news articles were collected, several steps were taken before proceeding with the topic modelling.
* 1.	Lowercase – The contents of the articles were converted to lowercase to maintain consistency in the text.
* 2.	Removing punctuation – The punctuation from the articles were removed.
* 3.	Removing numbers – Numbers were removed from the articles to make the text in the articles clearer for analysis.
* 4.	Removing stopwords – Stopwords from the data were removed to highlight the relevant text in the articles.
* 5.	Lemmatization – The articles were lemmatized to retain the real meaning of the word in the data and make the analysis more accurate.
All these steps were part of preprocessing of the data before moving forward with the topic modelling.
The preprocessing of the data was concluded on python and the preprocessed file was saved for topic modelling.

Using the preprocessed file topic modelling was performed individually for articles around ChatGPT and Google Bard.

## **Interpretation of Findings**

**Google Bard**

The topic modeling process was conducted with different numbers of topics, ranging from 2 to 9. Coherence scores were utilized to determine the optimal number of topics. Coherence scores measure the interpretability and meaningfulness of topics; a higher score indicates more coherent and distinct topics.  
Upon evaluating coherence scores, it was observed that the coherence score reaches its maximum value at 4 topics (Coherence Value: 0.349). Hence, the decision was made to proceed with 4 topics for further analysis.

![image](https://github.com/user-attachments/assets/a8422cdf-4003-4d16-808c-2c40e5cbebae)

The LDA model was trained with 4 topics, and the following key themes were extracted:

![image](https://github.com/user-attachments/assets/58d12f68-e162-4740-a942-20f6e0c97e88)

Topic 0 – Customer Experience  
Topic 1 – Conversational AI  
Topic 2 – Customer Support  
Topic 3 – Communication  

The perplexity value was calculated as -7.7092. Additionally, the coherence score for the selected 4 topics was 0.3437, signifying a good level of interpretability and coherence among the topics.

## 

**ChatGpt**

In our topic modeling analysis, we explored a range of topics from 2 to 9, evaluating their coherence scores to identify the optimal number of topics for our dataset. The coherence scores measure the interpretability and meaningfulness of topics, helping us choose the most suitable configuration for our analysis.

![image](https://github.com/user-attachments/assets/fe11d720-8e43-4fe3-9ddf-917104a3a251)

Upon analyzing the coherence scores, we observed that the highest coherence value was achieved when the number of topics was set to 6, with a coherence score of 0.355. This indicates that the words within each topic were closely related and formed coherent clusters, making them interpretable. Therefore, we proceeded with 6 topics, which provided a balance between coherence and granularity in our analysis.


Upon generating the topics using the LDA model with 4 topics, we observed distinct keyword patterns within each topic. Here are the identified topics along with their prominent keywords:

![image](https://github.com/user-attachments/assets/9cc10ee0-f044-4926-8b93-03150c8db978)

Topic 0 – Productivity.  
Topic 1 – Innovation.  
Topic 2 – Efficiency.  
Topic 3 – Creative.  

Additionally, the perplexity score, a measure of how well the model predicts the data, was calculated to be approximately -8.20. A lower perplexity score suggests better predictability of the model.

## **Conclusion**  
We employed topic modeling techniques to gain insights into the key themes and areas of discussion within a collection of articles related to ChatGPT and BARD. The results of our analysis suggest that there is a strong focus on the following topics:  
- Customer Experience  
- Conversational AI  
- Customer Support  
- Communication  
-  Productivity  
- Innovation  
- Efficiency   
- Creativity   
These findings suggest that ChatGPT and BARD are being used to develop new technologies and applications that can help businesses and individuals improve their productivity, creativity, and communication. Additionally, there is a growing interest in using these models to provide better customer support and create more engaging customer experiences. 

Overall, the results of this study provide valuable insights into the key trends and areas of innovation in the field of conversational AI. The findings suggest that ChatGPT and BARD are being used to develop powerful new tools and applications that have the potential to revolutionize the way we interact with computers and with each other.

