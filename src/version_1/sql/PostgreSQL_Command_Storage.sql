
/* Renaming user table to users to avoid Postgres reserved words issue. */

ALTER TABLE "user"
RENAME TO users
;

/* Adding primary keys in Postgres after import from Json file. */
/* No unique identifier (primary key) for tips table. */

ALTER TABLE business 
ADD PRIMARY KEY (business_id)
;

ALTER TABLE users 
ADD PRIMARY KEY (user_id)
;

ALTER TABLE review 
ADD PRIMARY KEY (review_id)
;

ALTER TABLE checkin 
ADD PRIMARY KEY (business_id)
;

/* Create table with only review text. */

SELECT review_id, 
       "text" AS review_text
INTO review_text_only
FROM review
;

ALTER TABLE review_text_only 
ADD PRIMARY KEY (review_id)
;

/* 
Create table of only user friends.
Used to create connectivity factors.
*/

SELECT user_id,
       friends 
INTO user_friends
FROM users
;

ALTER TABLE user_friends 
ADD PRIMARY KEY (user_id)
;

/* 
Create table of all useful non-review-text features.
For use after adding checkin_expanded and
user_friends_full tables.
*/

SELECT
    review.review_id          AS review_id,
    review.user_id            AS user_id,
    review.business_id        AS business_id,
    review.stars              AS review_stars,
    review.useful             AS review_useful,
    review.funny              AS review_funny,
    review.cool               AS review_cool,
    review.date               AS review_date,
    business.latitude         AS business_latitude,
    business.longitude        AS business_longitude,
    business.stars            AS business_avg_stars,
    business.review_count     AS business_review_count,
    business.is_open          AS business_is_open,
    business.categories       AS business_categories,
    checkin_expanded_2.checkin_count       AS business_checkin_count,
    checkin_expanded_2.oldest_checkin      AS business_oldest_checkin,
    checkin_expanded_2.most_recent_checkin AS business_newest_checkin,
    users.average_stars       AS user_avg_stars,
    users.review_count        AS user_review_count,
    users.yelping_since       AS user_yelping_since,
    users.useful              AS user_useful,
    users.funny               AS user_funny,
    users.cool                AS user_cool,
    users.compliment_hot      AS user_compliment_hot,
    users.compliment_more     AS user_compliment_more,
    users.compliment_profile  AS user_compliment_profile,
    users.compliment_cute     AS user_compliment_cute,
    users.compliment_list     AS user_compliment_list,
    users.compliment_note     AS user_compliment_note,
    users.compliment_plain    AS user_compliment_plain,
    users.compliment_cool     AS user_compliment_cool,
    users.compliment_funny    AS user_compliment_funny,
    users.compliment_writer   AS user_compliment_writer,
    users.compliment_photos   AS user_compliment_photos,
    users.fans                AS user_fans,
    user_friends_full.friend_count AS user_friend_count,
    users.elite               AS user_elite
INTO all_features
FROM review
INNER JOIN business
ON review.business_id = business.business_id
INNER JOIN users
ON review.user_id = users.user_id
JOIN checkin
ON review.business_id = checkin.business_id
JOIN user_friends_full
ON users.user_id = user_friends_full.user_id
JOIN checkin_expanded_2
ON business.business_id = checkin_expanded_2.business_id
;

ALTER TABLE all_features 
ADD PRIMARY KEY (review_id)
;

/*
Create sample of 10000 users
that have reviewed a business in Toronto.
Used for testing Graph/Network connectivity.
*/

SELECT user_friends_full.user_id,
       user_friends_full.friends,
       user_friends_full.friend_count
INTO user_friends_10k
FROM user_friends_full
JOIN review
ON review.user_id = user_friends_full.user_id
JOIN business
ON review.business_id = business.business_id
WHERE business.city = 'Toronto'
LIMIT 10000
;

/*
Combine and order text non-text features.
*/

SELECT
    non_nlp_model_data.*,
    review_text_only.review_text
INTO all_model_data
FROM non_nlp_model_data
JOIN review_text_only
ON non_nlp_model_data.review_id = review_text_only.review_id
;

ALTER TABLE all_model_data
ADD PRIMARY KEY (review_id)
;

/*
Randomly splitting all_model_data into working and holdout datasets.
*/

SELECT *
INTO holdout_all_model_data
FROM all_model_data TABLESAMPLE BERNOULLI (20) REPEATABLE (7)
;

ALTER TABLE holdout_all_model_data
ADD PRIMARY KEY (review_id)
;

SELECT * 
INTO working_all_model_data
FROM all_model_data
EXCEPT 
SELECT *
FROM holdout_all_model_data
;

ALTER TABLE working_all_model_data
ADD PRIMARY KEY (review_id)
;

/*
Split features and targets of working and holdout sets
into which questions they are trying to answer;
time-discounted or not.
*/

SELECT
    review_id,
    review_stars,
    review_stars_minus_user_avg,
    review_stars_minus_business_avg,
    review_stars_v_user_avg_sqr_diff,
    review_stars_v_business_avg_sqr_diff,
    business_avg_stars,
    "business_review_count_TD",
    "business_checkin_count_TD",
    "business_checkins_per_review_TD",
    user_avg_stars,
    user_days_active_at_review_time,
    "user_total_ufc_TD",
    "user_review_count_TD",
    "user_friend_count_TD",
    "user_fans_TD",
    "user_compliments_TD",
    "user_elite_count_TD",
    "user_years_since_most_recent_elite_TD",
    "user_ufc_per_review_TD",
    "user_fans_per_review_TD",
    "user_ufc_per_years_yelping_TD",
    "user_fans_per_years_yelping_TD",
    "user_fan_per_rev_x_ufc_per_rev_TD",
    "T1_REG_review_total_ufc",
    "T2_CLS_ufc_>0",
    "T3_CLS_ufc_level",
    "T4_REG_ufc_TD",
    "T5_CLS_ufc_level_TD",
    "T6_REG_ufc_TDBD",
    review_text
INTO holdout_TD_data
FROM holdout_all_model_data
;

SELECT
    review_id,
    review_stars,
    review_stars_minus_user_avg,
    review_stars_minus_business_avg,
    review_stars_v_user_avg_sqr_diff,
    review_stars_v_business_avg_sqr_diff,
    business_avg_stars,
    "business_review_count_TD",
    "business_checkin_count_TD",
    "business_checkins_per_review_TD",
    user_avg_stars,
    user_days_active_at_review_time,
    "user_total_ufc_TD",
    "user_review_count_TD",
    "user_friend_count_TD",
    "user_fans_TD",
    "user_compliments_TD",
    "user_elite_count_TD",
    "user_years_since_most_recent_elite_TD",
    "user_ufc_per_review_TD",
    "user_fans_per_review_TD",
    "user_ufc_per_years_yelping_TD",
    "user_fans_per_years_yelping_TD",
    "user_fan_per_rev_x_ufc_per_rev_TD",
    "T1_REG_review_total_ufc",
    "T2_CLS_ufc_>0",
    "T3_CLS_ufc_level",
    "T4_REG_ufc_TD",
    "T5_CLS_ufc_level_TD",
    "T6_REG_ufc_TDBD",
    review_text
INTO working_TD_data
FROM working_all_model_data
;

ALTER TABLE holdout_TD_data
ADD PRIMARY KEY (review_id)
;

ALTER TABLE working_TD_data
ADD PRIMARY KEY (review_id)
;

SELECT
    review_id,
    review_stars,
    review_stars_minus_user_avg,
    review_stars_minus_business_avg,
    review_stars_v_user_avg_sqr_diff,
    review_stars_v_business_avg_sqr_diff,
    business_avg_stars,
    business_review_count,
    business_checkin_count,
    business_checkins_per_review,
    user_avg_stars,
    user_total_ufc,
    user_review_count,
    user_friend_count,
    user_fans,
    user_compliments,
    user_elite_count,
    user_years_since_most_recent_elite,
    user_days_active_at_review_time,
    user_ufc_per_review,
    user_fans_per_review,
    user_ufc_per_years_yelping,
    user_fans_per_years_yelping,
    user_fan_per_rev_x_ufc_per_rev,
    "T1_REG_review_total_ufc",
    "T2_CLS_ufc_>0",
    "T3_CLS_ufc_level",
    "T4_REG_ufc_TD",
    "T5_CLS_ufc_level_TD",
    "T6_REG_ufc_TDBD",
    review_text                           
INTO holdout_non_TD_data
FROM holdout_all_model_data
;

SELECT
    review_id,
    review_stars,
    review_stars_minus_user_avg,
    review_stars_minus_business_avg,
    review_stars_v_user_avg_sqr_diff,
    review_stars_v_business_avg_sqr_diff,
    business_avg_stars,
    business_review_count,
    business_checkin_count,
    business_checkins_per_review,
    user_avg_stars,
    user_total_ufc,
    user_review_count,
    user_friend_count,
    user_fans,
    user_compliments,
    user_elite_count,
    user_years_since_most_recent_elite,
    user_days_active_at_review_time,
    user_ufc_per_review,
    user_fans_per_review,
    user_ufc_per_years_yelping,
    user_fans_per_years_yelping,
    user_fan_per_rev_x_ufc_per_rev,
    "T1_REG_review_total_ufc",
    "T2_CLS_ufc_>0",
    "T3_CLS_ufc_level",
    "T4_REG_ufc_TD",
    "T5_CLS_ufc_level_TD",
    "T6_REG_ufc_TDBD",
    review_text
INTO working_non_TD_data
FROM working_all_model_data
;

ALTER TABLE holdout_non_TD_data
ADD PRIMARY KEY (review_id)
;

ALTER TABLE working_non_TD_data
ADD PRIMARY KEY (review_id)
;