
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
    checkin_expanded.checkin_count AS business_checkin_count,
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
JOIN checkin_expanded
ON business.business_id = checkin_expanded.business_id
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