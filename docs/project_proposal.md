# Capstone 2 - Project Proposal
## Yelp Data Exploration
**Jeff Johannsen**

## BACKGROUND and MOTIVATION

I have a personal and professional interest in the leisure/tourism industry with a focus on two things:
1. The perceived quality and popularity of places and points of interest. 
    * Examples: Restaurants, Bars/Nightlife, Breweries, Music Venues, Parks and Outdoor Spaces. Very broad focus. Pretty much anything that involves the combination of leisure and being in public.
2. Creating a way to connect people to these places/POIs in a way that is more beneficial to both the places/POIs and the people than current common methods. 

Yelp and other similar companies like Foursquare, Google Places, TripAdvisor, etc. are the current leaders in this space, though I believe there are substantial untapped opportunities in this space.
Working with and better understanding the data available to these companies will help us to more fully take advantage of these opportunities.

## QUESTION, GOAL, and FOCUS
The goal of this project is to determine which Yelp reviews to surface to users in close to real time. The ability for Yelp to showcase its most useful reviews as they happen allows Yelp to provide the most up to date information to its users which increases user satisfaction, retention, and value to the company. 

## DATA

[Yelp Open Dataset](https://www.yelp.com/dataset)

My main dataset is a subset of Yelp's data that they provide to the public for academic purposes.
This dataset consists of 5 seperate json files totaling ~10GB of data uncompressed. Overall there is a mix of datatypes. The major ones are long text strings, datetimes, booleans, and numerical counts/ratings. Plenty of nan/null values but this is partially offset by the size of the dataset. The five files consist of:
* **Users**- ~2 million rows and 22 features
    * User metadata, list of friends, and stats about interactions with other users.
    * Main focus
* **Reviews**- ~8 million reviews with 9 features.
    * 8 review metadata features and the review content.
* **Checkins**- ~175000 businesses represented. Looks like a couple million total date-times. 
    * Dates and times for check-ins for each business. 
* **Businesses**- ~210000 rows with around 30 total features.
    * Business name, address, hours, rating, review count, etc. 
    * Also includes dictionaries of more in depth attributes like cost, accepts credit cards, good for kids, etc.
* **Tips**- ~1.3 million rows with 5 features
    * Kind of like a really short review without as much useful metadata.
    * Probably won't use this file.

## Minimum Viable Product
* Practice, Improve, and Showcase specific data science skills:
    * Data Visualization/Storytelling, Random Forest, Boosted Trees, Data Pipeline Creation, Feature Engineering, Database Management, Cloud Computing (AWS).
* Collect, clean and organize data in a easily searchable database (Postgres) by creating a pipeline.
* Explore and visualize interesting relationships within the data.
* Create a target variable based off a time adjusted number of useful votes on a review.
* Collect, create, and organize features for predicting the target value. Features come from metadata surrounding the review, reviewer (user), and business being reviewed.
* Utilize Random Forest, Boosted Trees, and other machine learning models to predict the target from the features.

## MVP+
* Practice, Improve, and Showcase specific data science skills:
    * Neural Networks, Natural Language Processing (NLP)
* Expand features by exploring review text using NLP techniques.
* Utilize neural networks to model and predict.