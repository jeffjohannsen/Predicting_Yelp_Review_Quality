
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

/* Create table of all useful non-review text features.*/

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
    checkin.date              AS business_checkins,
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
    users.elite               AS user_elite,
    users.friends             AS user_friends
INTO all_non_text_features
FROM review
INNER JOIN business
ON review.business_id = business.business_id
INNER JOIN users
ON review.user_id = users.user_id
JOIN checkin
ON review.business_id = checkin.business_id
;

ALTER TABLE all_non_text_features 
ADD PRIMARY KEY (review_id)
;