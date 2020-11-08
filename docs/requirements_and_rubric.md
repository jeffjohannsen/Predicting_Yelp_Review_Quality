# Capstone II

For most, Capstone II is a start on your final capstone project.  Quality data,
EDA, and feature engineering take time and are as important in this Capstone II
as Capstone I.  They set you up for success.  

## Capstone II goals

* An MVP that demonstrates supervised or unsupervised learning (and maybe both).
* In the case of supervised learning, picking an appropriate metric to quantify 
  performance, and then use of that metric in cross-validation to arrive at
  a model that generalizes as well as possible on unseen data.  Be prepared 
  for the request: "Describe the process you used to ensure your model
  was properly fit."
* In the case of unsupervised learning, picking an appropriate clustering 
  method and metric to guide the choice of the number of hard or soft clusters.
  This should be followed up with a thoughtful discussion as to why the 
  chosen number of clusters is "correct."
* In all cases, discussion of the cleaning and featurization pipeline and how 
  raw data were transformed to the data trained on.  Text processing especially
  requires discussion.  
* In the case of classification and class imbalance, discussion of how the
  class imbalance was addressed.  Did you use a default decision threshold,
  or did you pick a different threshold through out-of-model knowledge (e.g.
  a cost-benefit matrix and a profit curve.)


## Capstone II proposal  

You need to write-up a Capstone II proposal and submit it to your instructors for
review.  Please submit a **one-page PDF** file (no text files, markdowns, or 
Word Documents).

The proposal should:

1) State your question.  What is it that you are curious about?  What are you looking 
for in the data?

2) Demonstrate that you have looked at your data.  What are your columns?  Are they
numerical/catagorical?  How many Nan's are there?  Have you made a couple of plots? 
We only have a week for this capstone. It's very hard to do a good capstone when 
you've only had the real dataset for a couple of days.  This can make it challenging to 
work with a company.  Their timescale is different from the DSI.  "Fast" for them is a 
couple of weeks.  You needed the dataset yesterday.

3) State your MVP.  MVP is your Minimum Viable Product.  What's the minimum that you 
hope to accomplish?  Then, feel free to expand on MVP+, and MVP++.  

## Evaluation  
The three capstones account for 40% of your DSI assessment score/grade.  The scores
equally weight each capstone, and equally weight the presentations (~10 minutes) and 
accompanying Github for each capstone.  

The rubrics for Capstone II are similar to Capstone I.

#### Capstone II Github scoring rubric:

|Scoring item                          |Points | Description                                                 |
|:-------------------------------------|:-----:|:------------------------------------------------------------|
|Repo. organization and commit history |   3   | 0: No organization, 1 commit.<br> 1: Poor organization, 3+ commits.<br> 2: Ok organization, 3+ commits.<br>3: scripts in `src`, jupyter notebooks in `notebooks`, data, images, and other files similarly organized and 5+ commits.|
|Appropriate use of scripts/notebooks  |   3   | 0: Everything in a jupyter notebook.<br> 3: Perhaps some EDA presented in notebooks, but data acquisition, cleaning, and important analyses in scripts.|
|Object-oriented programming           |   3   | 0: You are repeating repeating yourself yourself.<br> 2: In functions. <br>3: Appropriate use of classes|
|Code style                            |   3   | 0: It is difficult to understand what your code is doing and what variables signify.<br> 2: Good variable/function/class names, `if __name__ == '__main__'` block, appropriate documentation.<br>  3: All previous and would perfectly pass a [pycodestyle](https://pypi.org/project/pycodestyle/) test.|
|Coding effort (**in scripts**)            |   3   | 0: <= 50 lines of code<br> 1: 51-100<br> 2: 101-150<br> 3: > 150     |

#### Capstone II presentation scoring rubric
|Scoring item                          |Points | Description                                                 |
|:-------------------------------------|:-----:|:------------------------------------------------------------|
|Project question/goal                 |   2   | 0: What are you doing?<br> 1: Theme explained, but not clear question/goal stated.<br> 2: Stated clearly with gusto.        |
|Description of raw data               |   2   | 0: Mentioned in passing - no idea of what the features are, where it came from, how it was obtained.<br>  1: Just a few features, the source described in text (but no images of raw data.) <br>2: Source described, walk through exemplary features and rows, appropriate tables/screenshots.|
|Exploratory Data Analysis             |   3   | 0: Who needs to understand the data, anyway?<br> 1: Perfunctory - general pair-wise scatter matrix that says..what?<br> 2:  Documentation of interesting relationships between the features and target.<br> 3: All previous and with thoughtful feature engineering.|
|Analysis (e.g cleaning pipeline, modeling, validation of model, presentation of results) |   5   | 0: None<br> 1: Approach invalid/unsuited to problem.<br>  2: Brief description focused on results without any explanation of approach/method.<br> 3: Clearly explained process that my be slightly incomplete or with minor errors.<br>4: Clearly explained with no/few errors.<br> 5: Impressive effort.| 
|README                                |   3   | 0: Missing or useless in describing project.<br>  1: Misspellings, hard to read font, strange formatting, ugly screenshots, inconsistent text sizes, wall-of-text.<br> 2: Generally pleasing that describes project well, good illustrations, a few minor issues. <br>3: Beautiful and an impressive showcase for the project with good organization, appropriate use of text and illustrations, and helpful references.|

