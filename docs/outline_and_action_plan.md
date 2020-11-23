# Outline and Action Plan

* Getting the Data
    * Locating Data Sources
    * Setting Data Collection Methods and Code
    * Bringing in the Data and Data Storage
* EDA
* Data Munging
    * Combine
    * Clean
    * Organize
* Data Visualization
* Data Pipeline Creation - Probably a Class
* Questions and Goals
* Model Selection and Optimization
    * Supervised Learning Model
    * Unsupervised Learning Model
    * Cost Metric Selection
    * How the models were fit and Why they were fit this way.
    * Cross Validation
* Conclusions
* Readme
    * Creation
    * Organization
    * Proof-reading and clean-up
* Code
    * Organization
    * Functions and Pipeline Class
    * Docstrings
    * Proofing with pycodestyle
* Github Repo
    * Organization
    * Proofing and clean-up
* Presentation
    * Outline and Flow that fits timeframe
    * Practice Read-throughs
    * Utilize Public Speaking Template


## Reminders
* Focus on MVP. Create Pipeline of classes and functions that automate as much of the above Action Plan as possible.
* Don't get stuck in data acquisition. Need to have a backup plan if data collection isn't going well enough.
* Spend most of my class time focused on value add work like README additions and improvements and python coding. Research can be focused on during off time if I'm still interested.
* Remember to save time (At least 1 full day) for smoothing out the final project. README touch ups, presentation outline and practice, pycodestyle review, Github repo clean up and organization, etc.

## Expected Time Requirements and Time Budget

|Item                                                           |Time Budget|Actual Time Use|
|:---|:---:|:---:|
|**Getting the Data**                                               |      |     |
|Locating Data Sources                                          |      |     |
|Setting Data Collection Methods and Code                       |      |     |
|Bringing in the Data and Data Storage                          |      |     |
|**EDA**                                                            |      |     |
|Data Munging                                                   |      |     |
|Combine                                                        |      |     |
|Clean                                                          |      |     |
|Organize                                                       |      |     |
|**Data Visualization**                                             |      |     |
|**Data Pipeline Creation - Probably a Class**                      |      |     |
|**Questions and Goals**                                            |      |     |
|**Model Selection and Optimization**                               |      |     |
|Supervised Learning Model                                      |      |     |
|Unsupervised Learning Model                                    |      |     |
|Cost Metric Selection                                          |      |     |
|How the models were fit and Why they were fit this way.        |      |     |
|Cross Validation                                               |      |     |
|**Conclusions**                                                    |      |     |
|**Readme**                                                         |      |     |
|Creation                                                       |      |     |
|Organization                                                   |      |     |
|Proof-reading and clean-up                                     |      |     |
|**Code**                                                           |      |     |
|Organization                                                   |      |     |
|Functions and Pipeline Class                                   |      |     |
|Docstrings                                                     |      |     |
|Proofing with pycodestyle                                      |      |     |
|**Github Repo**                                                    |      |     |
|Organization                                                   |      |     |
|Proofing and clean-up                                          |      |     |
|**Presentation**                                                   |      |     |
|Outline and Flow that fits timeframe                           |      |     |
|Practice Read-throughs                                         |      |     |
|Utilize Public Speaking Template                               |      |     |

## Final Week

## Monday Goals

## Tuesday Goals

## Wednesday Goals

## Thursday Goals

## Notes

Question Selection

Predicting review usefulness at time of posting.
* Business Case
    * Yelp can optimally surface reviews in real time instead of waiting on users to vote on the review usefullness. 
    * Allows Yelp reviews to keep up with changes in the business quality instead of lagging behind it. Like reddit's hot feed instead of the top feed.
    * Bayesian updating of review quality over time from initial review usefulness rating. Updated based on any "useful", "funny", and "cool" votes and time passed. Helps correct for wrong guesses.
    * TARGET: Composite of review "useful", "funny", and "cool". Time discounted.
    * FEATURES:
        * Review:
            * Review text
            * Star Rating
            * Date/Time
        * User
            * Review Count
            * User Yelp Join Date/Time
            * "Elite" - Count and time span since last
            * Friend Count
            * Fans Count
            * Average Ratings - +Difference between review rating and average review rating
            * Compliment counts (Multiple) Composite?
            * Time discounted "useful", "funny", and "cool" counts (User start to review date to current data)
        * Business
            * Stars +Difference between review stars and overall stars
            * Review Count
            * Location
            * Categories
            * Price Range
            * Checkin Count from Checkins

        * Tips could also be evaluated easier (less text)
