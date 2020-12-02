![=Header Image=](images/header_image.jpg)

# Predicting Yelp Review Usefullness

## Table of Contents
* [Introduction](#Introduction)
    * [Motivation](#Motivation)
    * [The Data](#The-Data)
* [Data Pipeline](#Data-Pipeline)
* [Data Analysis](#Data-Analysis)
* [Answering Questions with the Data](#Answering-Questions-with-the-Data)
    * [Supervised Learning](#Supervised-Learning)
    * [Unsupervised Learning](#Unsupervised-Learning)
* [Main Takeaways and Further Work](#Main-Takeaways-and-Further-Work)
    * [Next Steps](#Next-Steps)
    * [What I Learned from this Project](#What-I-Learned-from-this-Project)
* [Photo and Data Credits](#Photo-and-Data-Credits)

<br/><br/>

# Introduction

## Motivation

I have a personal and professional interest in the leisure/tourism industry with a focus on two things:
1. The perceived quality and popularity of places and points of interest. 
    * Examples: Restaurants, Bars/Nightlife, Breweries, Music Venues, Parks and Outdoor Spaces. Very broad focus. Pretty much anything that involves the combination of leisure and being in public.
2. Creating a way to connect people to these places/POIs in a way that is more beneficial to both the places/POIs and the people than current common methods. 

Yelp and other similar companies like Foursquare, Google Places, TripAdvisor, etc. are the current leaders in this space, though I believe there are substantial untapped opportunities in this space.
Working with and better understanding the data available to these companies will help us to more fully take advantage of these opportunities.

### Central Questions and Goals:
### 1. Can the usefullness of a review be determined at the time it was created?
### 2. What model and which inputs are the best for predicting the usefullness of a review?
### 3. Predict the usefullness of a review at the time it is created.

### Business Use Case

* Yelp can optimally surface reviews in real time instead of waiting on users to vote on the review usefullness. 
* Allows Yelp reviews to keep up with changes in the business quality instead of lagging behind it. 
* Like reddit's hot feed instead of the top feed.
* The ability for Yelp to showcase its most useful reviews as they happen allows Yelp to provide the most up to date information to its users which increases user satisfaction, retention, and value to the company. 

## The Data

The data for this project comes from the Yelp Open Dataset.  
[Dataset](https://www.yelp.com/dataset), [Documentation](https://www.yelp.com/dataset/documentation/main), [Yelp Github](https://github.com/Yelp/dataset-examples), [Kaggle](https://www.kaggle.com/yelp-dataset/yelp-dataset)

![Yelp Open Dataset Home Page](images/yelp_dataset_homepage_top.png)

### About Yelp 
> ### *<div align="center">"Yelp connects people with great local businesses. With unmatched local business information, photos and review content, Yelp provides a one-stop local platform for consumers to discover, connect and transact with local businesses of all sizes by making it easy to request a quote, join a waitlist, and make a reservation, appointment or purchase. Yelp was founded in San Francisco in 2004."</div>*

[Yelp News](https://www.yelp-press.com/news/default.aspx), [Fast Facts and Stats](https://www.yelp-press.com/company/fast-facts/default.aspx) <--- Say that 10 times fast.

### About the Data 

![Yelp About the Dataset](images/yelp_dataset_homepage_bottom.png)

This dataset consists of 5 seperate json files totaling ~10GB of data uncompressed. Overall there is a mix of datatypes. The major ones are long text strings, datetimes, booleans, and numerical counts/ratings. Plenty of nan/null values but this is partially offset by the size of the dataset. The five files consist of:
* **Users**- ~2 million rows and 22 features
    * User metadata, list of friends, and stats about interactions with other users.
* **Reviews**- ~8 million reviews with 9 features.
    * 8 review metadata features and the review content.
* **Checkins**- ~175000 businesses represented. Looks like a couple million total date-times. 
    * Dates and times for check-ins for each business. 
* **Businesses**- ~210000 rows with around 30 total features.
    * Business name, address, hours, rating, review count, etc. 
    * Also includes dictionaries of more in depth attributes like cost, accepts credit cards, good for kids, etc.
* **Tips**- ~1.3 million rows with 5 features
    * Kind of like a really short review without as much useful metadata.

<br/><br/>

# Data Analysis



### Original Dataset

=Markdown Table of Head of Original Data=

&nbsp;



<br/><br/>

# Answering Questions with the Data

## Question 1:
## Can the usefullness of a review be determined at the time it was created?



## Question 2:
## What model and which inputs are the best for predicting the usefullness of a review?

## Supervised Learning


## Unsupervised Learning

<br/><br/>

# Conclusions
### 
### 

### =Conclusions Related to Business Use Case=

<br/><br/>

# Main Takeaways and Further Work

## Next Steps

* 
*  

### Future Questions

* 
* 

## What I Learned from this Project



# Photo and Data Credits  
**Cover Photo**:   

**Main Data Sources**:

**Other Credits**:
